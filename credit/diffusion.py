import math
from random import random
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast
from einops import rearrange, reduce

from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from credit.diffusion_utils import (
    default,
    identity,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def standard_normal_noise(shape=None, tensor=None, device=None):
    """
    Generate standard normal noise (mean=0, std=1), either using a shape or a reference tensor.

    Args:
        shape (tuple, optional): Shape for the output tensor.
        tensor (torch.Tensor, optional): Reference tensor to match shape and device.
        device (torch.device or str, optional): Device for output. Optional if tensor is provided.

    Returns:
        torch.Tensor: Noise sampled from N(0, 1).
    """
    if tensor is not None:
        return torch.randn_like(tensor)
    elif shape is not None:
        return torch.randn(shape, device=device or "cpu")
    else:
        raise ValueError("Either `shape` or `tensor` must be provided.")


def log_uniform_noise(shape=None, tensor=None, sigma_min=0.02, sigma_max=200.0, device=None):
    """
    Sample noise from a log-uniform distribution over standard deviations.

    Args:
        shape (tuple, optional): Shape of the output noise tensor. Required if `tensor` is not provided.
        tensor (torch.Tensor, optional): Reference tensor to match shape. Used if `shape` is not provided.
        sigma_min (float): Minimum standard deviation.
        sigma_max (float): Maximum standard deviation.
        device (torch.device or str, optional): Device for the output. If None, inferred from `tensor` or defaults to 'cpu'.

    Returns:
        torch.Tensor: Noise tensor sampled from N(0, σ^2), with σ ~ log-uniform.
    """
    if tensor is not None:
        shape = tensor.shape
        device = tensor.device if device is None else device
    elif shape is not None:
        device = device or "cpu"
    else:
        raise ValueError("Either `shape` or `tensor` must be provided.")

    # Sample log σ uniformly
    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    log_sigma = torch.empty(shape, device=device).uniform_(log_sigma_min, log_sigma_max)
    sigma = log_sigma.exp()

    # Sample noise: N(0, σ^2)
    noise = torch.randn(shape, device=device) * sigma
    return noise


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_v",
        beta_schedule="sigmoid",
        noise_type="normal",
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        offset_noise_strength=0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5,
        immiscible=False,
    ):
        super().__init__()
        assert not (isinstance(GaussianDiffusion) and model.channels != model.out_dim)
        assert not hasattr(model, "random_or_learned_sinusoidal_cond") or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.condition = self.model.condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, (
            "image size must be a integer or a tuple/list of two integers"
        )
        self.image_size = image_size

        self.objective = objective

        assert objective in {"pred_noise", "pred_x0", "pred_v"}, (
            "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"
        )

        if noise_type == "normal":
            self.randn_like_fn = standard_normal_noise
        elif noise_type == "log-uniform":
            self.randn_like_fn = log_uniform_noise
        else:
            raise ValueError(f"unknown noise type {noise_type}")

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x,
        t,
        x_self_cond=None,
        x_cond=None,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        model_output = self.model(x, t, x_self_cond, x_cond)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

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

    def p_mean_variance(self, x, t, x_self_cond=None, x_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond, x_cond)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None, x_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            x_self_cond=x_self_cond,
            x_cond=x_cond,
            clip_denoised=True,
        )
        noise = self.randn_like_fn(tensor=x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, x_cond, return_all_timesteps=False):
        _, device = shape[0], self.device
        img = self.randn_like_fn(shape=shape, device=device)
        imgs = [img]

        x_start = None
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, x_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        # ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, x_cond, return_all_timesteps=False, disable_tqdm=True):
        batch, device, total_timesteps, sampling_timesteps, eta, _ = (
            shape[0],
            self.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = self.randn_like_fn(shape=shape, device=device)
        imgs = [img]

        x_start = None
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step", disable=disable_tqdm):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                self_cond,
                x_cond,
                clip_x_start=True,
                rederive_pred_noise=True,
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = self.randn_like_fn(tensor=img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        if return_all_timesteps:
            print("not all timesteps have been returned, only every 10th")
            timesteps = list(range(0, ret.shape[1], 10))  # Timesteps with step of 10

            # Include the last 10 timesteps explicitly
            last_10_timesteps = list(range(max(ret.shape[1] - 10, 0), ret.shape[1]))

            # Merge the two lists, ensuring uniqueness and order
            timesteps = sorted(set(timesteps + last_10_timesteps))

            # Slice the array with the selected timesteps
            ret = ret[:, timesteps, :, :]

        # ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, x_cond, batch_size=16, return_all_timesteps=False):
        (h, w), channels, f = (
            self.image_size,
            self.model.output_channels,
            self.model.frames,
        )
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            (batch_size, channels, f, h, w),
            x_cond,
            return_all_timesteps=return_all_timesteps,
        )

    @torch.inference_mode()
    def interpolate(self, x1, x2, x_cond=None, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            self_cond = x_start if self.self_condition else None
            x_cond = x_cond if self.condition else None
            img, x_start = self.p_sample(img, i, self_cond, x_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, "b ... -> b (...)") for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast("cuda", enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: self.randn_like_fn(tensor=x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, x_cond, noise=None, offset_noise_strength=None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: self.randn_like_fn(tensor=x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        ### maybe do the same for the condition: WEC KLUDGE
        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond, x_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, x_cond, *args, **kwargs):
        (
            b,
            _,
            h,
            w,
            device,
            img_size,
        ) = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = self.normalize(img)
        return self.p_losses(img, t, x_cond, *args, **kwargs)


class ModifiedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = self.model.input_channels
        self.history_len = self.model.frames
        self.criterion = None

    def load_loss(self, criterion):
        self.criterion = criterion

    def forward(self, img, x_cond=None, *args, **kwargs):
        # Unpack the tensor shape
        if img.dim() == 4:  # b, c, h, w
            b, _, h, w = img.shape
            device = img.device
        elif img.dim() == 5:  # b, c, f, h, w (e.g., video or multi-frame)
            b, _, _, h, w = img.shape
            device = img.device
        else:
            raise ValueError(f"Unsupported tensor shape {img.shape}")

        # Ensure the height and width match the expected image size
        assert h == self.image_size[0] and w == self.image_size[1], (
            f"height and width of image must be {self.image_size}"
        )

        # Randomly sample timesteps for diffusion
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Normalize the image before passing it through the model
        # img = self.normalize(img)

        # Call the model's loss function (or whatever other method you want to use)
        return self.p_losses(img, t, x_cond, *args, **kwargs)

    def p_losses(self, x_start, t, x_cond, noise=None, offset_noise_strength=None):
        # Check the dimensions of the input tensor (x_start)
        if x_start.dim() == 4:  # For single frame (batch_size, channels, height, width)
            b, c, h, w = x_start.shape
        elif x_start.dim() == 5:  # For multi-frame input (batch_size, channels, frames, height, width)
            b, c, f, h, w = x_start.shape
        else:
            raise ValueError(f"Unsupported tensor shape {x_start.shape}")

        # Default to random noise if not provided
        noise = default(noise, lambda: self.randn_like_fn(tensor=x_start))

        # Offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # Noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Self-conditioning logic
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, x_cond=x_cond).pred_x_start
                x_self_cond.detach_()

        # Predict and take gradient step
        model_out = self.model(x, t, x_self_cond, x_cond)

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

        # Calculate loss
        if self.criterion:
            # 1. Compute user-defined loss (per variable, per lat point)
            raw_loss = self.criterion.loss_fn(target, model_out)  # shape: [B, V, L]
            _, _, _, H, W = raw_loss.shape

            # 2. Apply latitude weights
            if self.criterion.lat_weights is not None:
                lat_weights = self.criterion.lat_weights.to(raw_loss.device).view(1, 1, 1, H, 1)
                raw_loss = raw_loss * lat_weights

            # 3. Apply variable weights
            if self.criterion.var_weights is not None:
                var_weights = self.criterion.var_weights.to(raw_loss.device).view(1, 1, 1, H, 1)
                raw_loss = raw_loss * var_weights

            # 4. Reduce per-sample: mean over V and L
            loss_per_sample = raw_loss.mean(dim=[1, 2])  # shape: [B]

            # 5. Apply DDPM timestep weighting
            loss_weight = extract(self.loss_weight, t, loss_per_sample.shape)
            weighted_loss = loss_per_sample * loss_weight  # shape: [B]

            # 6. Final loss
            loss = weighted_loss.mean()

        else:
            loss = F.mse_loss(model_out, target, reduction="none")
            loss = reduce(loss, "b ... -> b", "mean")
            loss = loss * extract(self.loss_weight, t, loss.shape)
            loss = loss.mean()

        return model_out, target, loss

    def model_predictions(
        self,
        x,
        t,
        x_self_cond=None,
        x_cond=None,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        model_output = self.model(x, t, x_self_cond, x_cond)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

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
