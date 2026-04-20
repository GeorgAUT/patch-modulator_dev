import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from hydra.utils import instantiate
import the_well
import wandb


from data_utils.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.trainer.training import expand_mask_to_match
from walrus.trainer.normalization_strat import (
    BaseRevNormalization,
    normalize_target,
)

## Prepare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}", flush=True)

use_wandb = True

## Config and model
config_path = f"./config/tiny_patch_modulator_config_full_data.yaml"
# config_path = f"./config/walrus_extended_config.yaml"
config = OmegaConf.load(config_path)
# checkpoint_path = f"./checkpoints/walrus_checkpoints/walrus.pt"
# checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)


## Instantiate a minimal dataset similar to the notebook path
# well_base_path = "/Volumes/TheWell/thewell/datasets/"
# well_base_path = "./data/polymathic-ai/datasets/"
well_base_path = "./data/polymathic-ai/datasets_mesh/"

# The dataset objects precompute a number of dataset stats on init, so this may take a little while
data_module = instantiate(config.data.module_parameters,
                          well_base_path=well_base_path,
                          world_size=1,
                          rank=0,
                          data_workers=1,
                          field_index_map_override=config.data.get("field_index_map_override", {}), # Use the previous field maps to avoid cycling through the data
                          prefetch_field_names=False)

field_to_index_map = data_module.train_dataset.field_to_index_map
# Retrieve the number of fields in used in the dataset
# from the mapping of field to index and incrementing by 1
total_input_fields = max(field_to_index_map.values()) + 1


## Instantiate model and load weights where appropriate/possible

model: torch.nn.Module = instantiate(
    config.model,
    n_states=total_input_fields,
)
# model.load_state_dict(checkpoint)

# model.load_state_dict(checkpoint)

# Move to the device we want
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


## Dataset preparation

formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)() # This is a functools partial by default


# loader = data_module.rollout_val_dataloaders()[0]
# loader.num_workers = 0  # if you can set it
sampling_rank=0
train_dataloader = data_module.train_dataloader()#sampling_rank)
# train_dataloader = data_module.test_dataloader()#sampling_rank)
train_dataloader.num_workers = 0

## Training loop simplified version

model.train()
criterion = the_well.benchmark.metrics.MAE()
vrmse_metric = the_well.benchmark.metrics.VRMSE()
loss_multiplier=100.0
# criterion1 = nn.MSELoss()
lr = 5e-4
model_epsilon=1e-5
epochs=1000
optimizer = optim.Adam(model.parameters(), lr=lr)
artifact_interval = 100
checkpoint_path = "./checkpoints/patch_modulator_checkpoints/tiny_walrus_more_batches.pt"

if use_wandb:
    wandb.init(
        project="mesh-tusk-training",
        name="tiny-walrus-new-training-payel",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "loss_multiplier": loss_multiplier,
            "model_epsilon": model_epsilon,
            "batch_size": getattr(train_dataloader, "batch_size", None),
            "device": str(device),
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
        },
    )

# from dev_datasets_mesh import WellDataset
# from torch.utils.data import DataLoader
# trainset = WellDataset(
#     well_base_path=well_base_path,
#     well_dataset_name="turbulent_radiative_layer_2D",
#     well_split_name="train"
# )

# train_loader = DataLoader(trainset, batch_size=2, shuffle=True)

model.train()
for epoch in range(1, epochs + 1):
    running = 0.0
    running_vrmse = 0.0
    sample_count = 0
    batch_id = 0 # Batch id to count how far through
    for batch in train_dataloader:
        metadata = batch["metadata"]

        optimizer.zero_grad()
        # Move batch to device
        batch["padded_field_mask"] = batch["padded_field_mask"].to(device)  # We're going to want this out here too
        # Extract mask and move to device for loss eval
        if (
                "mask" in batch["metadata"].constant_field_names[0]  # Assuming all metadata in batch are the same
        ):
            mask_index = batch["metadata"].constant_field_names[0].index("mask")
            mask = batch["constant_fields"][..., mask_index: mask_index + 1]
            mask = mask.to(device, dtype=torch.bool)
        else:
            mask = None


        if "mesh_coords" in batch.keys():
            inputs_mesh, y_ref_mesh, mesh_data = formatter.process_input_mesh(
            batch,
            causal_in_time=model.causal_in_time,
            predict_delta=True,
            train=True,
            )
            mesh_data = list(mesh_data)

            with torch.no_grad():
                normalization_stats_mesh = revin.compute_stats(
                    inputs_mesh, metadata, epsilon=model_epsilon
                )
            normalized_inputs_mesh = inputs_mesh
            normalized_inputs_mesh = revin.normalize_stdmean(
                normalized_inputs_mesh, normalization_stats_mesh
            )
        else:
            mesh_data=None
            inputs_mesh=None
            normalized_inputs_mesh=None
            y_ref_mesh=None

        inputs, y_ref = formatter.process_input(
            batch,
            causal_in_time=model.causal_in_time,
            predict_delta=True,
            train=True,
        )
        inputs = list(inputs)  # Not sure what this is for
        inputs = [x.to(device) for x in inputs]


        # Batch normalisation
        with torch.no_grad():
            normalization_stats = revin.compute_stats(
                inputs[0], metadata, epsilon=model_epsilon
            )
        # NOTE - Currently assuming only [0] (fields) needs normalization
        normalized_inputs = inputs[:]  # Shallow copy
        normalized_inputs[0] = revin.normalize_stdmean(
            normalized_inputs[0], normalization_stats
        )
        print(normalized_inputs[0].shape, flush=True)

        # Move to device
        normalized_inputs = [x.to(device) for x in normalized_inputs]
        y_ref = y_ref.to(device)
        inputs_mesh = normalized_inputs_mesh.to(device) if inputs_mesh is not None else None
        mesh_data = [x.to(device) for x in mesh_data] if mesh_data is not None else None


        # Single step rollout training...
        y_pred = model(
            normalized_inputs[0],
            normalized_inputs[1],
            normalized_inputs[2],
            metadata=metadata)#, mesh_x=inputs_mesh , mesh_data=mesh_data
        # )


        y_pred = formatter.process_output(y_pred, metadata)[..., : y_ref.shape[-1]]  # Cut off constant channels

        y_ref = y_ref[:, :1]  # If we set a maximum number of rollout steps, just cut it off now to save memory

        mean = (
            normalization_stats.delta_mean
        )
        std = (
            normalization_stats.delta_std
        )
        y_ref = normalize_target(y_ref, mean, std, formatter, metadata, device)
        # If we have masked fields, just move them back to zeros
        if mask is not None:
            mask_pred = expand_mask_to_match(mask, y_pred)
            y_pred.masked_fill_(mask_pred, 0)

        y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)

        vrmse_val = vrmse_metric(y_pred, y_ref, metadata, eps=model_epsilon).mean()
        loss = loss_multiplier * criterion(y_pred, y_ref, metadata, eps=model_epsilon).mean()
        # loss = criterion1(y_pred, y_ref,metadata)
        loss.backward()
        optimizer.step()

        print(
            f"Batch {batch_id}/{len(train_dataloader.sampler)}, loss: {loss.item()**0.5:.4f}, vrmse: {vrmse_val.item():.6f}",
            flush=True,
        )
        if use_wandb:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_vrmse": vrmse_val.item(),
                "batch_id": batch_id,
                "epoch": epoch,
            })
        batch_id+=1

        batch_size = inputs[0].size(1) # Recall batch size is at location 2: T B C H [W D]
        running += loss.item() * batch_size
        running_vrmse += vrmse_val.item() * batch_size
        sample_count += batch_size


    train_loss = running / len(train_dataloader)
    train_vrmse = running_vrmse / max(sample_count, 1)

    msg = f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | train_vrmse={train_vrmse:.6f}"
    print(msg, flush=True)
    if use_wandb:
        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": train_loss,
            "epoch_train_vrmse": train_vrmse,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

    # Always overwrite the same file locally
    torch.save(model.state_dict(), checkpoint_path)

    # Log occasionally so something is safely stored in W&B
    if epoch % artifact_interval == 0 or epoch == epochs:
        if use_wandb:
            artifact = wandb.Artifact(
                "tiny_walrus_payel",
                type="model",
                metadata={
                    "epoch": epoch,
                    "n_states": total_input_fields,
                    "epoch_train_loss": train_loss,
                },
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=["latest"])

print(f"Checkpoint saved to: {checkpoint_path}", flush=True)

if use_wandb:
    wandb.finish()
