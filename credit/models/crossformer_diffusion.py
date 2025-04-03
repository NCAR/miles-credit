import torch
import torch.nn as nn
from credit.models.crossformer import CrossFormer
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import *
import torch.nn.functional as F
import random
from einops import rearrange, reduce
from tqdm.auto import tqdm
from functools import partial
from collections import namedtuple


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t, *args, **kwargs):
    return t


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class CrossFormerDiffusion(CrossFormer):
    def __init__(self, self_condition, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = kwargs.get(
            "dim", (64, 128, 256, 512)
        )  # Default value as in CrossFormer
        self.self_condition = self_condition

        # Adding timestep embedding layer for diffusion
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.dim[0]), nn.SiLU(), nn.Linear(self.dim[0], self.dim[-1])
        )

    def forward(self, x, timestep, x_self_cond=False):
        x_copy = None
        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        elif self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:
            x = x.squeeze(2)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        # Add timestep embedding to the feature maps
        t_embed = self.time_mlp(timestep.view(-1, 1).float())  # (B, dim[0])
        t_embed = t_embed[:, :, None, None]  # Reshape to (B, dim[0], 1, 1)
        t_embed = t_embed.expand(
            -1, -1, x.shape[2], x.shape[3]
        )  # Expand to (B, dim[0], H, W)
        x = x + t_embed

        x = self.up_block1(x)
        x = torch.cat([x, encodings[2]], dim=1)
        x = self.up_block2(x)
        x = torch.cat([x, encodings[1]], dim=1)
        x = self.up_block3(x)
        x = torch.cat([x, encodings[0]], dim=1)
        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(
                x, size=(self.image_height, self.image_width), mode="bilinear"
            )

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {"y_pred": x, "x": x_copy}
            x = self.postblock(x)

        return x


class ModifiedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = self.model.input_channels
        self.history_len = self.model.frames

    def forward(self, img, *args, **kwargs):
        # Unpack the tensor shape
        if img.dim() == 4:  # b, c, h, w
            b, c, h, w = img.shape
            device = img.device
        elif img.dim() == 5:  # b, c, f, h, w (e.g., video or multi-frame)
            b, c, f, h, w = img.shape
            device = img.device
        else:
            raise ValueError(f"Unsupported tensor shape {img.shape}")

        # Ensure the height and width match the expected image size
        assert (
            h == self.image_size[0] and w == self.image_size[1]
        ), f"height and width of image must be {self.image_size}"

        # Randomly sample timesteps for diffusion
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Normalize the image before passing it through the model
        img = self.normalize(img)

        # Call the model's loss function (or whatever other method you want to use)
        return self.p_losses(img, t, *args, **kwargs)

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None):
        # Check the dimensions of the input tensor (x_start)
        if x_start.dim() == 4:  # For single frame (batch_size, channels, height, width)
            b, c, h, w = x_start.shape
        elif (
            x_start.dim() == 5
        ):  # For multi-frame input (batch_size, channels, frames, height, width)
            b, c, f, h, w = x_start.shape
        else:
            raise ValueError(f"Unsupported tensor shape {x_start.shape}")

        # Default to random noise if not provided
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(
            offset_noise_strength, self.offset_noise_strength
        )

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # Noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Self-conditioning logic
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # Predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        # Determine target based on the objective
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"Unknown objective {self.objective}")

        _, C, _, _, _ = (
            model_out.shape
        )  # Get the correct channel size from model output
        target = target[:, :C]  # Slice target to match model output

        # Calculate loss
        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def model_predictions(
        self,
        x,
        t,
        x_self_cond=None,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        model_output = self.model(x, t, x_self_cond)

        # Here we have the mismatch between input and output sizes
        # Concat on the statics to the model output
        C1 = model_output.shape[1]
        C2 = x.shape[1]

        # Slice the missing channels from x_input
        missing_part = x[:, C1:, :, :, :]  # Slice the extra channels from x_input

        # Concatenate along the channel dimension (dim=1)
        model_output = torch.cat((model_output, missing_part), dim=1)

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            (batch_size, channels, self.history_len, h, w),
            return_all_timesteps=return_all_timesteps,
        )


def create_model(config):
    """Initialize and return the CrossFormer model using a config dictionary."""
    return CrossFormerDiffusion(**config).to("cuda")


def create_diffusion(model, config):
    """Initialize and return the Gaussian Diffusion process."""
    return ModifiedGaussianDiffusion(model, **config)


if __name__ == "__main__":
    crossformer_config = {
        "type": "crossformer",
        "frames": 1,  # Number of input states
        "image_height": 192,  # Number of latitude grids
        "image_width": 288,  # Number of longitude grids
        "levels": 16,  # Number of upper-air variable levels
        "channels": 4,  # Upper-air variable channels
        "surface_channels": 7,  # Surface variable channels
        "input_only_channels": 3,  # Dynamic forcing, forcing, static channels
        "output_only_channels": 0,  # Diagnostic variable channels
        "patch_width": 1,  # Number of latitude grids in each 3D patch
        "patch_height": 1,  # Number of longitude grids in each 3D patch
        "frame_patch_size": 1,  # Number of input states in each 3D patch
        "dim": [32, 64, 128, 256],  # Dimensionality of each layer
        "depth": [2, 2, 2, 2],  # Depth of each layer
        "global_window_size": [8, 4, 2, 1],  # Global window size for each layer
        "local_window_size": 8,  # Local window size
        "cross_embed_kernel_sizes": [  # Kernel sizes for cross-embedding
            [4, 8, 16, 32],
            [2, 4],
            [2, 4],
            [2, 4],
        ],
        "cross_embed_strides": [2, 2, 2, 2],  # Strides for cross-embedding
        "attn_dropout": 0.0,  # Dropout probability for attention layers
        "ff_dropout": 0.0,  # Dropout probability for feed-forward layers
        "use_spectral_norm": True,  # Whether to use spectral normalization
        "padding_conf": {
            "activate": True,
            "mode": "earth",
            "pad_lat": [32, 32],
            "pad_lon": [48, 48],
        },
        "interp": False,
        "self_condition": False,
        "pretrained_weights": "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/model_levels/single_step/checkpoint.pt",
    }

    diffusion_config = {
        "image_size": (192, 288),
        "timesteps": 100,
        "sampling_timesteps": None,
        "objective": "pred_v",
        "beta_schedule": "sigmoid",
        "schedule_fn_kwargs": dict(),
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
        "offset_noise_strength": 0.0,
        "min_snr_loss_weight": False,
        "min_snr_gamma": 5,
        "immiscible": False,
    }

    model = create_model(crossformer_config).to("cuda")
    diffusion = create_diffusion(model, diffusion_config).to("cuda")

    input_tensor = torch.randn(
        1,
        crossformer_config["channels"] * crossformer_config["levels"]
        + crossformer_config["surface_channels"]
        + crossformer_config["input_only_channels"],
        crossformer_config["frames"],
        crossformer_config["image_height"],
        crossformer_config["image_width"],
    ).to("cuda")

    print(input_tensor.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    loss = diffusion(input_tensor).item()  # lazy test

    print("Loss:", loss)

    sampled_images = diffusion.sample(batch_size=1)

    print("Predicted shape:", sampled_images.shape)

    # Extract the last color channel (index -1 for the last channel)
    last_channel = sampled_images[0, -1, 0, :, :]

    import matplotlib.pyplot as plt

    # Plot and save the image
    plt.imshow(last_channel.cpu().numpy(), cmap="gray")  # Display in grayscale
    plt.axis("off")  # Turn off the axis
    plt.savefig("last_channel.png", bbox_inches="tight", pad_inches=0)
    plt.close()
