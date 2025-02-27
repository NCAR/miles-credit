import gc
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.distributed as dist
import torch.fft
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset
import optuna
from glob import glob
from credit.transforms import load_transforms
from credit.data import ERA5_and_Forcing_Dataset
from credit.trainers.base_trainer import BaseTrainer
from overrides import overrides
import random


def cleanup():
    dist.destroy_process_group()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


class TOADataLoader:
    # This should get moved to solar.py at some point
    def __init__(self, conf):
        self.TOA = xr.open_dataset(conf["data"]["TOA_forcing_path"])
        self.times_b = pd.to_datetime(self.TOA.time.values)

    def __call__(self, datetime_input):
        doy = datetime_input.dayofyear
        hod = datetime_input.hour
        mask_toa = [doy == time.dayofyear and hod == time.hour for time in self.times_b]
        return (
            torch.tensor(((self.TOA["tsi"].sel(time=mask_toa)) / 2540585.74).to_numpy())
            .unsqueeze(0)
            .float()
        )


class WeightedRMSE(torch.nn.Module):
    def __init__(self, conf):
        super(WeightedRMSE, self).__init__()
        self.lat_weights = None
        if conf["loss"]["use_latitude_weights"]:
            lat = xr.open_dataset(conf["loss"]["latitude_weights"])["latitude"].values
            w_lat = np.cos(np.deg2rad(lat))
            w_lat = w_lat / w_lat.mean()
            self.lat_weights = (
                torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1)
            )  # Shape: (1, lat_dim, 1)

    def forward(self, predictions, targets):
        if self.lat_weights is not None:
            # Ensure lat_weights is on the same device as predictions and targets
            lat_weights = self.lat_weights.to(predictions.device)

            # Apply latitude weights
            weighted_diff = lat_weights * (predictions - targets) ** 2
            weighted_mse = weighted_diff.mean()
            weighted_rmse = torch.sqrt(weighted_mse)
            return weighted_rmse
        else:
            return torch.sqrt(torch.nn.functional.mse_loss(predictions, targets))


class ReplayBuffer:
    def __init__(self, conf, buffer_size=32, device="cpu", dtype=np.float32, rank=0):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.dtype = dtype
        self.device = device
        self.rank = rank

        # Extract relevant parameters from conf
        data_conf = conf["data"]
        filenames = data_conf.get("save_loc")
        history_len = data_conf.get("history_len", 2)
        forecast_len = data_conf.get("forecast_len", 0)
        transform = data_conf.get("transform", None)

        model_conf = conf["model"]
        input_shape = (
            model_conf["levels"] * model_conf["channels"]
            + model_conf["surface_channels"]
            + model_conf["static_channels"],
            model_conf["frames"],
            model_conf["image_height"],
            model_conf["image_width"],
        )

        self.input_shape = input_shape

        # Initialize forecast hour and index buffers
        self.forecast_hour = np.zeros((buffer_size,), dtype=np.int32)
        self.index = np.zeros((buffer_size,), dtype=np.int32)
        self.q_values = np.zeros((buffer_size,), dtype=np.float32)
        self.rmse_scores = np.zeros((buffer_size,), dtype=np.float32)

        # File names
        filenames = sorted(glob(filenames))

        # Preprocessing transformations
        transform = load_transforms(conf)

        # Initialize dataset
        self.dataset = ERA5_and_Forcing_Dataset(
            model_conf["data"]["variables"],
            model_conf["data"]["surface_variables"],
            model_conf["data"]["dynamic_forcing_variables"],
            model_conf["data"]["forcing_variables"],
            model_conf["data"]["static_variables"],
            model_conf["data"]["diagnostic_variables"],
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            transform=transform,
        )

        # Create a directory to store numpy files
        self.numpy_dir = os.path.join(conf["save_loc"], "buffer")
        os.makedirs(self.numpy_dir, exist_ok=True)

        # Reload if the dataset exists
        self.reload()

        # Rewards metric
        self.metric_fn = WeightedRMSE(conf)

    def add(self, x, lookup_key):
        """Add new experience to the buffer."""
        # Check if lookup_key equals or exceeds the length of the dataloader
        # This should be a very rare occurance
        if lookup_key.item() >= len(self.dataset):
            # If so, replace a random entry
            random_idx = random.randint(0, self.size - 1)
            file_path = os.path.join(
                self.numpy_dir, f"buffer_{self.rank}_{random_idx}.npy"
            )
            np.save(file_path, x.cpu().numpy())
            self.index[random_idx] = lookup_key.item()
            self.forecast_hour[random_idx] = 1  # Reset forecast hour
            self.q_values[random_idx] = 0.0  # Reset Q-value
            self.rmse_scores[random_idx] = 0.0  # Reset RMSE score
        elif self.size < self.buffer_size:
            file_path = os.path.join(
                self.numpy_dir, f"buffer_{self.rank}_{self.ptr}.npy"
            )
            np.save(file_path, x.cpu().numpy())
            self.index[self.ptr] = lookup_key.item()
            self.forecast_hour[self.ptr] = 1  # Initialize forecast_hour to 1
            self.q_values[self.ptr] = 0.0  # Initialize Q-value to 0
            self.rmse_scores[self.ptr] = 0.0  # Initialize RMSE score to 0
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size += 1
        else:
            # Replace a random entry if buffer is full
            random_idx = random.randint(0, self.size - 1)
            file_path = os.path.join(
                self.numpy_dir, f"buffer_{self.rank}_{random_idx}.npy"
            )
            np.save(file_path, x.cpu().numpy())
            self.index[random_idx] = lookup_key.item()
            self.forecast_hour[random_idx] = 1  # Reset forecast hour
            self.q_values[random_idx] = 0.0  # Reset Q-value
            self.rmse_scores[random_idx] = 0.0  # Reset RMSE score

    # def add(self, x, lookup_key):
    #     """Add new experience to the buffer."""
    #     if self.size < self.buffer_size:
    #         file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{self.ptr}.npy")
    #         np.save(file_path, x.cpu().numpy())
    #         self.index[self.ptr] = lookup_key.item()
    #         self.forecast_hour[self.ptr] = 1  # Initialize forecast_hour to 1
    #         self.q_values[self.ptr] = 0.0  # Initialize Q-value to 0
    #         self.rmse_scores[self.ptr] = 0.0  # Initialize RMSE score to 0
    #         self.ptr = (self.ptr + 1) % self.buffer_size
    #         self.size += 1
    #     else:
    #         # Replace the entry with the smallest Q-value if buffer is full
    #         random_idx = random.randint(0, self.size - 1)
    #         file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{random_idx}.npy")
    #         np.save(file_path, x.cpu().numpy())
    #         self.index[random_idx] = lookup_key.item()
    #         self.forecast_hour[random_idx] = 1  # Reset forecast hour
    #         self.q_values[random_idx] = 0.0  # Reset Q-value
    #         self.rmse_scores[random_idx] = 0.0  # Reset RMSE score

    def sample(self, batch_size, epsilon=0.2):
        """Sample a batch of experiences from the buffer, increment forecast_hour, and update x with new predictions."""
        epsilon_prob = np.random.rand()

        # If filling the buffer and predicting, need to catch the first added sample (q=0) which leads to a NaN
        if all(self.q_values == 0):
            indices = np.argsort(self.q_values[: self.size])[:batch_size]
        elif epsilon_prob < epsilon:
            # Exploration: select random experiences
            indices = np.random.choice(self.size, batch_size, replace=False)
        else:
            # Exploitation: select experiences with probability proportional to Q-values
            q_values_safe = np.nan_to_num(
                self.q_values[: self.size], nan=0.0, posinf=0.0, neginf=0.0
            )
            weights = q_values_safe  # Use Q-values directly
            weights -= np.min(weights)  # Shift weights to be non-negative
            weights /= np.sum(
                weights
            )  # Normalize to create a valid probability distribution
            # Sample indices with lowest RMSEs
            indices = np.random.choice(self.size, batch_size, replace=False, p=weights)
        # else:
        #     # Exploitation: select experiences with the highest Q-values
        #     indices = np.argsort(-self.q_values[:self.size])[:batch_size]

        # Increment forecast_hour for sampled indices
        self.forecast_hour[indices] += 1

        x_batch = np.zeros((batch_size, *self.input_shape), dtype=self.dtype)

        for i, idx in enumerate(indices):
            file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{idx}.npy")
            x_batch[i] = np.load(file_path, mmap_mode="r")

        x_batch = torch.FloatTensor(x_batch).to(self.device)
        return indices, x_batch

    def update_q_values(self, indices, y_predict, y_truth):
        for i, idx in enumerate(indices):
            # Calculate the RMSE for the predicted and true values
            rmse = self.metric_fn(
                y_predict[i].detach().cpu().squeeze(1), y_truth[i].cpu().squeeze(1)
            )

            # Use RMSE directly as the reward
            reward = -rmse

            # Number of updates (forecast hours) for this index
            n = self.forecast_hour[idx] - 1

            # Update Q-value using the specified formula
            self.q_values[idx] = self.q_values[idx] + (1 / n) * (
                reward - self.q_values[idx]
            )

            # Update RMSE score
            self.rmse_scores[idx] = rmse

    def update(self, indices, new_x, new_lookup_key):
        """Update existing data in the buffer."""
        for i, idx in enumerate(indices):
            file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{idx}.npy")
            np.save(file_path, new_x[i].cpu().numpy())
            self.index[idx] = new_lookup_key[i]

    def update_with_predictions(self, model, sample_size, epsilon=0.2):
        """Use stored predictions as inputs for future predictions."""

        indices, x_sample = self.sample(sample_size, epsilon=epsilon)
        ave_forecast_len = np.mean([t - 1 for t in self.forecast_hour[indices]])

        # Predict using the model
        y_predict = model(x_sample)

        y_truth = []
        x_update = []

        for i, idx in enumerate(indices):
            lookup_key = self.index[idx]

            # Load the next set of inputs using the lookup_key
            y, _ = self.load_inputs(lookup_key + 1)

            static = y[:, 67:]
            y_non_static = y[:, :67, 1:]

            y_truth.append(y_non_static)

            y_pred = y_predict[i].unsqueeze(0).cpu().detach()
            y_pred = torch.cat((y_pred, static[:, :, 1:2, :, :].cpu()), dim=1)
            y_pred = torch.cat([x_sample[i : i + 1, :, 1:2, :, :].cpu(), y_pred], dim=2)
            x_update.append(y_pred)

        x_update = torch.cat(x_update, dim=0)
        y_truth = torch.cat(y_truth, dim=0)

        new_lookup_keys = self.index[indices] + 1
        self.update(indices, x_update, new_lookup_keys)
        self.update_q_values(indices, y_predict, y_truth)  # Update Q-values
        self.save()

        return y_predict, y_truth, ave_forecast_len

    def concat_and_reshape(self, x1, x2):
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def load_inputs(self, idx):
        sample = self.dataset.__getitem__(idx)

        x = self.concat_and_reshape(
            sample["x"].unsqueeze(0), sample["x_surf"].unsqueeze(0)
        )

        if "static" in sample:
            static = (
                torch.FloatTensor(sample["static"]).unsqueeze(0).expand(2, -1, -1, -1)
            )
            x = torch.cat([x, static.unsqueeze(0)], dim=1)

        if "TOA" in sample:
            toa = torch.FloatTensor(sample["TOA"]).unsqueeze(0)
            x = torch.cat([x, toa.unsqueeze(1)], dim=1)

        y = self.concat_and_reshape(
            sample["y"].unsqueeze(0), sample["y_surf"].unsqueeze(0)
        )

        return x, y

    def populate(self):
        """Populate the buffer with random data points from the dataset."""
        dataset_size = len(self.dataset)
        random_indices = np.random.choice(dataset_size, self.buffer_size, replace=False)

        for i, idx in tqdm.tqdm(enumerate(random_indices), total=len(random_indices)):
            x, _ = self.load_inputs(idx)
            self.add(x, idx)

        self.size = self.buffer_size

    def save(self):
        """Save the forecast hours, index arrays, pointer, size, Q-values, and RMSE scores to disk."""
        np.save(
            os.path.join(self.numpy_dir, f"forecast_hours_{self.rank}.npy"),
            self.forecast_hour,
        )
        np.save(os.path.join(self.numpy_dir, f"index_{self.rank}.npy"), self.index)
        np.save(
            os.path.join(self.numpy_dir, f"ptr_{self.rank}.npy"), np.array([self.ptr])
        )
        np.save(
            os.path.join(self.numpy_dir, f"size_{self.rank}.npy"), np.array([self.size])
        )
        np.save(
            os.path.join(self.numpy_dir, f"q_values_{self.rank}.npy"), self.q_values
        )
        np.save(
            os.path.join(self.numpy_dir, f"rmse_scores_{self.rank}.npy"),
            self.rmse_scores,
        )

    def reload(self):
        """Reload the buffer from saved numpy files."""
        forecast_hour_path = os.path.join(
            self.numpy_dir, f"forecast_hours_{self.rank}.npy"
        )
        index_path = os.path.join(self.numpy_dir, f"index_{self.rank}.npy")
        ptr_path = os.path.join(self.numpy_dir, f"ptr_{self.rank}.npy")
        size_path = os.path.join(self.numpy_dir, f"size_{self.rank}.npy")
        q_values_path = os.path.join(self.numpy_dir, f"q_values_{self.rank}.npy")
        rmse_scores_path = os.path.join(self.numpy_dir, f"rmse_scores_{self.rank}.npy")

        if os.path.exists(forecast_hour_path):
            self.forecast_hour = np.load(forecast_hour_path)
        else:
            self.forecast_hour = np.zeros((self.buffer_size,), dtype=np.int32)

        if os.path.exists(index_path):
            self.index = np.load(index_path)
        else:
            self.index = np.zeros((self.buffer_size,), dtype=np.int32)

        if os.path.exists(ptr_path):
            self.ptr = np.load(ptr_path)[0]
        else:
            self.ptr = 0

        if os.path.exists(size_path):
            self.size = np.load(size_path)[0]
        else:
            self.size = 0

        if os.path.exists(q_values_path):
            self.q_values = np.load(q_values_path)
        else:
            self.q_values = np.zeros((self.buffer_size,), dtype=np.float32)

        if os.path.exists(rmse_scores_path):
            self.rmse_scores = np.load(rmse_scores_path)
        else:
            self.rmse_scores = np.zeros((self.buffer_size,), dtype=np.float32)


class Trainer(BaseTrainer):
    # Training function.
    @overrides
    def train_one_epoch(
        self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics
    ):
        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_accum_every = conf["trainer"]["grad_accum_every"]
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        rollout_p = (
            1.0
            if "stop_rollout" not in conf["trainer"]
            else conf["trainer"]["stop_rollout"]
        )
        batch_size = conf["trainer"]["train_batch_size"]

        if (
            "static_variables" in conf["data"]
            and "tsi" in conf["data"]["static_variables"]
        ):
            self.toa = TOADataLoader(conf)

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if (
            conf["trainer"]["use_scheduler"]
            and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"
        ):
            scheduler.step()

        # set up a custom tqdm
        if isinstance(trainloader.dataset, IterableDataset):
            # we sample forecast termination with probability p during training
            trainloader.dataset.set_rollout_prob(rollout_p)
        else:
            batches_per_epoch = (
                batches_per_epoch
                if 0 < batches_per_epoch < len(trainloader)
                else len(trainloader)
            )

        batch_group_generator = tqdm.tqdm(
            enumerate(trainloader),
            total=batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        static = None
        results_dict = defaultdict(list)

        replay_buffer = ReplayBuffer(
            conf, device=self.device, rank=self.rank, buffer_size=100
        )

        self.model.train()

        for i, batch in batch_group_generator:
            logs = {}

            commit_loss = 0.0

            with autocast(enabled=amp):
                x = self.model.concat_and_reshape(batch["x"], batch["x_surf"]).to(
                    self.device
                )

                if "static" in batch:
                    if static is None:
                        static = (
                            batch["static"]
                            .to(self.device)
                            .unsqueeze(2)
                            .expand(-1, -1, x.shape[2], -1, -1)
                            .float()
                        )
                    x = torch.cat((x, static.clone()), dim=1)

                if "TOA" in batch:
                    toa = batch["TOA"].to(self.device)
                    x = torch.cat([x, toa.unsqueeze(1)], dim=1)

                # Add to buffer only if it's not yet full
                if replay_buffer.size < replay_buffer.buffer_size:
                    replay_buffer.add(x, batch["index"])
                    epsilon = 0.0
                else:
                    epsilon = 0.2

                y_pred, y, ave_forecast_hour = replay_buffer.update_with_predictions(
                    self.model, batch_size, epsilon
                )

                # sample from the buffer
                y = y.to(self.device)
                y_pred = y_pred.to(self.device)
                loss = criterion(y, y_pred)

                # Metrics
                metrics_dict = metrics(y_pred.float(), y.float())
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

                loss = loss.mean() + commit_loss

                scaler.scale(loss / grad_accum_every).backward()

            accum_log(logs, {"loss": loss.item() / grad_accum_every})

            if distributed:
                torch.distributed.barrier()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(ave_forecast_hour)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                try:
                    raise optuna.TrialPruned()
                except Exception as E:
                    raise E

            # agg the results
            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len {:.6}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                np.mean(results_dict["train_forecast_len"]),
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.set_description(to_print)

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] == "cosine-annealing"
            ):
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
