import argparse
import os
import os.path as osp
import time
from typing import Any, Dict, Optional, cast

import torch
import torch.distributed.checkpoint as dcp
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from controllable_patching_striding.data import MixedWellDataModule
from controllable_patching_striding.data.well_to_multi_transformer import (
    ChannelsFirstWithTimeFormatter,
)
from controllable_patching_striding.trainer.checkpoints import DummyCheckPointer
from controllable_patching_striding.trainer.training import Trainer
from controllable_patching_striding.utils.distribution_utils import (
    configure_distribution,
    distribute_model,
)


class _ModelOnlyAppState(Stateful):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, [])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            model=self.model,
            optimizers=[],
            model_state_dict=state_dict["model"],
            optim_state_dict=None,
            options=StateDictOptions(strict=False),
        )


def _find_checkpoint_config_path(checkpoint_dir: str) -> Optional[str]:
    candidates = [
        osp.join(checkpoint_dir, "config.yaml"),
        osp.join(checkpoint_dir, "extended_config.yaml"),
        osp.join(checkpoint_dir, "..", "extended_config.yaml"),
        osp.join(checkpoint_dir, "..", "..", "extended_config.yaml"),
    ]
    for p in candidates:
        p = osp.realpath(p)
        if osp.isfile(p):
            return p
    return None


def _normalize_checkpoint_cfg(cfg: Any) -> Any:
    """Normalize a checkpoint config into a usable Hydra config.

    Some runs save a W&B-style config.yaml where each key is stored as:
      key:
        value: <actual-value>
    and may include a top-level _wandb section.
    """
    if not isinstance(cfg, dict):
        # DictConfig -> plain container
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg, dict):
        return OmegaConf.create({})

    cfg = dict(cfg)
    cfg.pop("_wandb", None)

    unwrapped: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and "value" in v and len(v) == 1:
            unwrapped[k] = v["value"]
        else:
            unwrapped[k] = v

    return OmegaConf.create(unwrapped)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load a pretrained checkpoint and run rollout validation to generate videos."
    )
    p.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help=(
            "Path to a checkpoint directory (e.g. .../checkpoints/last or .../checkpoints/step_<N>)."
        ),
    )
    p.add_argument(
        "--well_base_path",
        default="./datasets",
        help="Base path containing The Well datasets.",
    )
    p.add_argument(
        "--output_dir",
        default="./outputs",
        help="Directory where viz outputs (including mp4) will be written.",
    )
    p.add_argument(
        "--hydra_override",
        action="append",
        default=[],
        help=(
            "Additional Hydra override(s), e.g. --hydra_override data.module_parameters.max_samples=20"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.output_dir is None:
        args.output_dir = osp.join(
            os.getcwd(), "viz_from_checkpoint", time.strftime("%Y%m%d_%H%M%S")
        )
    os.makedirs(args.output_dir, exist_ok=True)

    config_dir = osp.join(
        osp.dirname(__file__), "controllable_patching_striding", "configs"
    )

    # Always start from a fully composed base config so required keys exist.
    base_overrides = [
        "server=local",
        "distribution=local",
        "data=TRL_2D",
        f"data.well_base_path={args.well_base_path}",
        "validation_mode=True",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        base_cfg = compose(config_name="config", overrides=base_overrides)
    # Convert to a non-structured config so we can merge checkpoint configs that may contain
    # extra keys not present in the base structured config (e.g. _wandb).
    base_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))

    checkpoint_cfg_path = _find_checkpoint_config_path(args.checkpoint_dir)
    if checkpoint_cfg_path is not None:
        print(f"Loading config from checkpoint: {checkpoint_cfg_path}")
        checkpoint_cfg = cast(Any, OmegaConf.load(checkpoint_cfg_path))
        checkpoint_cfg = _normalize_checkpoint_cfg(checkpoint_cfg)
        cfg = cast(Any, OmegaConf.merge(base_cfg, checkpoint_cfg))
    else:
        cfg = base_cfg

    # Apply any extra user overrides last.
    if args.hydra_override:
        cfg = cast(Any, OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.hydra_override)))

    # Apply runtime overrides on top of loaded config.
    # Some older saved configs may not include distribution_type; default to local.
    with open_dict(cfg):
        if not hasattr(cfg, "distribution") or cfg.distribution is None:
            cfg.distribution = {}
        if (
            not hasattr(cfg.distribution, "distribution_type")
            or cfg.distribution.distribution_type is None
        ):
            cfg.distribution.distribution_type = "local"

    cfg.data.well_base_path = args.well_base_path
    cfg.trainer.video_validation = True
    cfg.trainer.image_validation = False
    if hasattr(cfg, "logger") and hasattr(cfg.logger, "wandb"):
        cfg.logger.wandb = False

    # Some configs refer to the datamodule via a legacy import path.
    if (
        hasattr(cfg.data, "module_parameters")
        and hasattr(cfg.data.module_parameters, "_target_")
        and isinstance(cfg.data.module_parameters._target_, str)
        and cfg.data.module_parameters._target_.startswith("data.")
    ):
        cfg.data.module_parameters._target_ = (
            "controllable_patching_striding.data.multidatamodule.MixedWellDataModule"
        )

    world_size = 1
    rank = 0
    local_rank = 0
    is_distributed = False

    mesh = configure_distribution(cfg)

    datamodule: MixedWellDataModule = instantiate(
        cfg.data.module_parameters,
        world_size=world_size,
        rank=rank,
        data_workers=cfg.data_workers,
        well_base_path=cfg.data.well_base_path,
    )

    total_input_fields = max(datamodule.train_dataset.field_to_index_map.values()) + 1

    model: torch.nn.Module = instantiate(cfg.model, n_states=total_input_fields)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(local_rank)}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model = distribute_model(model, cfg, mesh)

    # NOTE: We only need an optimizer object to satisfy the Trainer constructor.
    # Validation does not use it (no backward/step), and we intentionally do not load optimizer state.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    start_epoch = 1
    val_loss: Optional[float] = None
    if args.checkpoint_dir:
        metadata_file = osp.join(args.checkpoint_dir, "metadata.pt")
        if osp.exists(metadata_file):
            print("loading model from checkpoint")
            meta = torch.load(metadata_file, weights_only=False)
            print(f"loaded metadata: epoch={meta.get('epoch', None)}, val_loss={meta.get('val_loss', None)}")
            epoch = meta.get("epoch", None)
            val_loss = meta.get("val_loss", None)
            start_epoch = 1 if epoch is None else epoch + 1
        state_dict = {"app": _ModelOnlyAppState(model)}
        dcp.load(state_dict=state_dict, checkpoint_id=args.checkpoint_dir)
    wandb_logging = False

    cfg.trainer.max_epoch = start_epoch # Trainer isn't actually used this is just to recover model from checkpoint

    trainer: Trainer = instantiate(
        cfg.trainer,
        experiment_name=osp.basename(args.output_dir),
        viz_folder=args.output_dir,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=None,
        checkpointer=DummyCheckPointer(save_dir=osp.join(args.output_dir, "checkpoints"), rank=rank),
        device=device,
        is_distributed=is_distributed,
        distribution_type=cfg.distribution.distribution_type,
        rank=rank,
        formatter=ChannelsFirstWithTimeFormatter,
        wandb_logging=wandb_logging,
        start_epoch=start_epoch,
        start_val_loss=val_loss,
    )

    # Only run rollout validation (the code path that generates videos when video_validation=True).
    rollout_val_dataloaders = datamodule.rollout_val_dataloaders()
    trainer.validation_loop(
        rollout_val_dataloaders,
        valid_or_test="rollout_valid",
        full=False,
        epoch=start_epoch,
    )

if __name__ == "__main__":
    main()
