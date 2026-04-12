"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Aurora: A Foundation Model of the Atmosphere (Price et al., 2024)
https://arxiv.org/abs/2405.13063

Stripped of HuggingFace dependencies for use as a standalone trainable module.
"""

import dataclasses
import warnings
from datetime import timedelta
from typing import Optional

import torch

from .batch import Batch
from .decoder import Perceiver3DDecoder
from .encoder import Perceiver3DEncoder
from .lora import LoRAMode
from .swin3d import Swin3DTransformerBackbone

__all__ = ["Aurora", "AuroraSmall"]


class Aurora(torch.nn.Module):
    """The Aurora model (1.3 B parameter configuration).

    Perceiver3DEncoder → Swin3DTransformerBackbone → Perceiver3DDecoder.

    Input/output: :class:`.Batch` dataclass with ``surf_vars``, ``atmos_vars``,
    ``static_vars`` and ``metadata``.
    """

    def __init__(
        self,
        *,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        max_history_size: int = 2,
        timestep: timedelta = timedelta(hours=6),
        stabilise_level_agg: bool = False,
        use_lora: bool = True,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        level_condition: Optional[tuple[int | float, ...]] = None,
        dynamic_vars: bool = False,
        atmos_static_vars: bool = False,
        separate_perceiver: tuple[str, ...] = (),
        modulation_heads: tuple[str, ...] = (),
        positive_surf_vars: tuple[str, ...] = (),
        positive_atmos_vars: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or {}
        self.max_history_size = max_history_size
        self.timestep = timestep
        self.use_lora = use_lora
        self.positive_surf_vars = positive_surf_vars
        self.positive_atmos_vars = positive_atmos_vars

        if self.surf_stats:
            warnings.warn(
                f"Normalisation stats manually adjusted for: {', '.join(sorted(self.surf_stats.keys()))}.",
                stacklevel=2,
            )

        self.encoder = Perceiver3DEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=embed_dim // num_heads,
            depth=enc_depth,
            latent_levels=latent_levels,
            max_history_size=max_history_size,
            perceiver_ln_eps=perceiver_ln_eps,
            stabilise_level_agg=stabilise_level_agg,
            level_condition=level_condition,
            dynamic_vars=dynamic_vars,
            atmos_static_vars=atmos_static_vars,
        )

        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            attn_drop_rate=drop_rate,
            drop_rate=drop_rate,
            use_lora=use_lora,
            lora_steps=lora_steps,
            lora_mode=lora_mode,
        )

        self.decoder = Perceiver3DDecoder(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim * 2,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            depth=dec_depth,
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
            level_condition=level_condition,
            separate_perceiver=separate_perceiver,
            modulation_heads=modulation_heads,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass.

        Args:
            batch: :class:`.Batch` containing surf/atmos/static variables and metadata.

        Returns:
            :class:`.Batch`: Predicted next state (unnormalised).
        """
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        transformed_batch = batch
        if self.positive_surf_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                surf_vars={
                    k: v.clamp(min=0) if k in self.positive_surf_vars else v for k, v in batch.surf_vars.items()
                },
            )
        if self.positive_atmos_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                atmos_vars={
                    k: v.clamp(min=0) if k in self.positive_atmos_vars else v for k, v in batch.atmos_vars.items()
                },
            )

        x = self.encoder(transformed_batch, lead_time=self.timestep)
        x = self.backbone(
            x,
            lead_time=self.timestep,
            patch_res=patch_res,
            rollout_step=batch.metadata.rollout_step,
        )
        pred = self.decoder(x, batch, lead_time=self.timestep, patch_res=patch_res)

        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        pred = pred.unnormalise(surf_stats=self.surf_stats)
        return pred


class AuroraSmall(Aurora):
    """Small Aurora variant for debugging / fast iteration (~200 M params at full resolution)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("encoder_depths", (2, 6, 2))
        kwargs.setdefault("encoder_num_heads", (4, 8, 16))
        kwargs.setdefault("decoder_depths", (2, 6, 2))
        kwargs.setdefault("decoder_num_heads", (16, 8, 4))
        kwargs.setdefault("embed_dim", 256)
        kwargs.setdefault("num_heads", 8)
        kwargs.setdefault("use_lora", False)
        super().__init__(**kwargs)
