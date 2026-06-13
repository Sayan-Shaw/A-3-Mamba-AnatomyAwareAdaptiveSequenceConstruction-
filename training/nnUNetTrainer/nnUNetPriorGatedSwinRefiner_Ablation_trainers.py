"""
Ablation study trainers for PriorGatedSwinResidualRefiner.

Three ablation conditions, each paired with a matching network variant:

  1. nnUNetTrainerAblationNoUncertainty
     Prior channels: [coarse, SDF]  (uncertainty dropped, ch=2)

  2. nnUNetTrainerAblationNoSDF
     Prior channels: [coarse, uncertainty]  (SDF dropped, ch=2)

  3. nnUNetTrainerAblationCoarseOnly
     Prior channels: [coarse]  (both uncertainty and SDF dropped, ch=1)

Channel layout in the input tensor x (same as full model):
  x[:, 0] = MRI
  x[:, 1] = coarse probability/mask
  x[:, 2] = uncertainty map
  x[:, 3] = coarse SDF

The ablation networks slice only the channels they need from this tensor,
so the data pipeline does NOT need to change — all 4 channels are still
loaded/preprocessed as usual; unused channels are simply ignored at
the network's forward() entry point.

Each trainer overrides only build_network_architecture() to swap in the
appropriate ablation network.  Everything else (loss, augmentation, deep
supervision, boundary/SDF aux losses, femur mirroring fix, etc.) is
inherited unchanged from nnUNetTrainerPriorGatedSwinRefiner.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Local imports — adjust paths to match your project layout
# ---------------------------------------------------------------------------
from nnunetv2.nets.prior_aware_refiner import (
    ConvNormAct,
    DecoderStage,
    EncoderStage,
    SwinBottleneck3D,
    _as_tuple,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerPriorGatedSwinRefiner import (
    nnUNetTrainerPriorGatedSwinRefiner,
)


# ===========================================================================
# Shared ablation PriorGateBlock — accepts n_prior_channels instead of 3
# ===========================================================================

class AblationPriorGateBlock(nn.Module):
    """
    Identical logic to PriorGateBlock but parameterised on prior channel count
    so it works for 1-channel (coarse only) or 2-channel ablations.
    """

    def __init__(self, channels: int, n_prior_channels: int):
        super().__init__()
        assert n_prior_channels >= 1, "Need at least one prior channel."
        hidden = max(8, min(32, channels // 2))
        self.gate = nn.Sequential(
            ConvNormAct(n_prior_channels, hidden, 3),
            nn.Conv3d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.adapter = nn.Sequential(
            ConvNormAct(n_prior_channels, hidden, 3),
            nn.Conv3d(hidden, channels, 1, bias=True),
        )

    @staticmethod
    def _resize_prior(prior: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
        if prior.shape[2:] == tuple(spatial_shape):
            return prior
        return F.interpolate(prior, size=spatial_shape, mode="trilinear", align_corners=False)

    def forward(self, features: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        prior = self._resize_prior(prior, features.shape[2:])
        gate = self.gate(prior)
        adapted_prior = self.adapter(prior)
        return features * (0.5 + gate) + 0.1 * adapted_prior


# ===========================================================================
# Base ablation network — only differs from PriorGatedSwinResidualRefiner in:
#   (a) n_prior_channels passed to AblationPriorGateBlock
#   (b) which input channels are sliced in forward()
# ===========================================================================

class _AblationRefinerBase(nn.Module):
    """
    Internal base class shared by all three ablation network variants.
    Subclasses set `_N_PRIOR` and override `_extract_prior()`.
    """

    _N_PRIOR: int  # must be set by subclass

    def __init__(
        self,
        num_classes: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        n_conv_per_stage_decoder: Sequence[int],
        deep_supervision: bool = True,
        coarse_logit_scale: float = 1.0,
        window_size: Sequence[int] = (2, 4, 4),
    ):
        super().__init__()
        if int(num_classes) > 2:
            raise NotImplementedError("Ablation refiners are intended for binary organ datasets.")
        if n_stages < 2:
            raise ValueError("Need at least two stages.")

        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.coarse_logit_scale = float(coarse_logit_scale)
        features_per_stage = [int(i) for i in features_per_stage]
        kernel_sizes = [tuple(int(j) for j in i) for i in kernel_sizes]
        strides = [tuple(int(j) for j in i) for i in strides]
        n_blocks_per_stage = [int(i) for i in n_blocks_per_stage]
        n_conv_per_stage_decoder = [int(i) for i in n_conv_per_stage_decoder]

        self.encoder = nn.ModuleList()
        self.prior_gates = nn.ModuleList()
        in_ch = 1  # MRI only, same as full model
        for stage in range(n_stages):
            out_ch = features_per_stage[stage]
            self.encoder.append(
                EncoderStage(in_ch, out_ch, n_blocks_per_stage[stage], kernel_sizes[stage], strides[stage])
            )
            # KEY CHANGE: n_prior_channels reflects ablated prior
            self.prior_gates.append(AblationPriorGateBlock(out_ch, self._N_PRIOR))
            in_ch = out_ch

        self.context = SwinBottleneck3D(features_per_stage[-1], window_size=window_size)

        self.decoder = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        self.boundary_heads = nn.ModuleList()
        self.sdf_heads = nn.ModuleList()

        in_ch = features_per_stage[-1]
        for skip_stage in range(n_stages - 2, -1, -1):
            out_ch = features_per_stage[skip_stage]
            decoder_idx = n_stages - 2 - skip_stage
            n_decoder_blocks = n_conv_per_stage_decoder[
                min(decoder_idx, len(n_conv_per_stage_decoder) - 1)
            ]
            self.decoder.append(
                DecoderStage(
                    in_ch,
                    features_per_stage[skip_stage],
                    out_ch,
                    strides[skip_stage + 1],
                    n_decoder_blocks,
                    kernel_sizes[skip_stage],
                )
            )
            self.seg_heads.append(nn.Conv3d(out_ch, 1, 1, bias=True))
            self.boundary_heads.append(nn.Conv3d(out_ch, 1, 1, bias=True))
            self.sdf_heads.append(nn.Conv3d(out_ch, 1, 1, bias=True))
            in_ch = out_ch

        self.last_boundary_logits: Union[None, List[torch.Tensor]] = None
        self.last_sdf_logits: Union[None, List[torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Subclasses override this to select which prior channels to use
    # ------------------------------------------------------------------

    def _extract_prior(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers (identical to PriorGatedSwinResidualRefiner)
    # ------------------------------------------------------------------

    @staticmethod
    def initialize(module: nn.Module):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.InstanceNorm3d, nn.LayerNorm)):
            if getattr(module, "weight", None) is not None:
                nn.init.constant_(module.weight, 1)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

    def _coarse_logit(self, coarse: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
        if coarse.shape[2:] != tuple(spatial_shape):
            coarse = F.interpolate(coarse.float(), size=spatial_shape, mode="trilinear", align_corners=False)
        coarse = coarse.clamp(0.01, 0.99)
        return torch.logit(coarse).clamp(-5, 5) * self.coarse_logit_scale

    def _compose_logits(self, delta: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        foreground = self._coarse_logit(coarse, delta.shape[2:]) + delta
        if self.num_classes == 1:
            return foreground
        return torch.cat([torch.zeros_like(foreground), foreground], dim=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] < 4:
            raise ValueError(
                "Input must still carry all 4 channels [MRI, coarse, uncertainty, coarse_sdf]; "
                "unused channels are ignored by this ablation model."
            )

        mri = x[:, 0:1].float()
        coarse = x[:, 1:2].float()
        prior = self._extract_prior(x)  # shape: [B, _N_PRIOR, D, H, W]

        skips: List[torch.Tensor] = []
        features = mri
        for stage, gate in zip(self.encoder, self.prior_gates):
            features = stage(features)
            features = gate(features, prior)
            skips.append(features)

        features = self.context(skips[-1])
        seg_outputs: List[torch.Tensor] = []
        boundary_outputs: List[torch.Tensor] = []
        sdf_outputs: List[torch.Tensor] = []

        for decoder_stage, seg_head, boundary_head, sdf_head, skip in zip(
            self.decoder,
            self.seg_heads,
            self.boundary_heads,
            self.sdf_heads,
            reversed(skips[:-1]),
        ):
            features = decoder_stage(features, skip)
            seg_outputs.append(self._compose_logits(seg_head(features), coarse))
            boundary_outputs.append(boundary_head(features))
            sdf_outputs.append(sdf_head(features))

        seg_outputs = list(reversed(seg_outputs))
        self.last_boundary_logits = list(reversed(boundary_outputs))
        self.last_sdf_logits = list(reversed(sdf_outputs))

        if self.deep_supervision:
            return seg_outputs
        return seg_outputs[0]


# ===========================================================================
# Ablation Network 1: No Uncertainty  — prior = [coarse, SDF]  (ch=2)
# ===========================================================================

class PriorGatedSwinRefiner_NoUncertainty(_AblationRefinerBase):
    """
    Ablation: uncertainty map dropped from prior.
    Prior gate receives: [coarse (ch1), SDF (ch3)]  →  2 channels.
    """

    _N_PRIOR = 2

    def _extract_prior(self, x: torch.Tensor) -> torch.Tensor:
        # coarse=ch1, SDF=ch3 — skip ch2 (uncertainty)
        return torch.cat([x[:, 1:2], x[:, 3:4]], dim=1).float()


# ===========================================================================
# Ablation Network 2: No SDF  — prior = [coarse, uncertainty]  (ch=2)
# ===========================================================================

class PriorGatedSwinRefiner_NoSDF(_AblationRefinerBase):
    """
    Ablation: SDF channel dropped from prior.
    Prior gate receives: [coarse (ch1), uncertainty (ch2)]  →  2 channels.
    """

    _N_PRIOR = 2

    def _extract_prior(self, x: torch.Tensor) -> torch.Tensor:
        # coarse=ch1, uncertainty=ch2 — skip ch3 (SDF)
        return x[:, 1:3].float()


# ===========================================================================
# Ablation Network 3: Coarse Only  — prior = [coarse]  (ch=1)
# ===========================================================================

class PriorGatedSwinRefiner_CoarseOnly(_AblationRefinerBase):
    """
    Ablation: both uncertainty and SDF dropped from prior.
    Prior gate receives: [coarse (ch1)]  →  1 channel.
    """

    _N_PRIOR = 1

    def _extract_prior(self, x: torch.Tensor) -> torch.Tensor:
        # coarse only
        return x[:, 1:2].float()


# ===========================================================================
# Trainer 1: No Uncertainty
# ===========================================================================

class nnUNetTrainerAblationNoUncertainty(nnUNetTrainerPriorGatedSwinRefiner):
    """
    Ablation trainer: full model minus the uncertainty map prior channel.

    Prior channels seen by PriorGateBlock: [coarse, SDF]
    Comparison: nnUNetTrainerPriorGatedSwinRefiner uses [coarse, uncertainty, SDF]

    Data pipeline is unchanged — the uncertainty channel (ch2) is loaded as
    normal but silently ignored inside PriorGatedSwinRefiner_NoUncertainty.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        n_blocks_per_stage = arch_init_kwargs.get(
            "n_conv_per_stage", arch_init_kwargs.get("n_blocks_per_stage")
        )
        if n_blocks_per_stage is None:
            raise KeyError("Expected n_conv_per_stage or n_blocks_per_stage in arch kwargs.")

        network = PriorGatedSwinRefiner_NoUncertainty(
            num_classes=num_output_channels,
            n_stages=arch_init_kwargs["n_stages"],
            features_per_stage=arch_init_kwargs["features_per_stage"],
            kernel_sizes=arch_init_kwargs["kernel_sizes"],
            strides=arch_init_kwargs["strides"],
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=arch_init_kwargs["n_conv_per_stage_decoder"],
            deep_supervision=enable_deep_supervision,
            coarse_logit_scale=1.0,
            window_size=(2, 4, 4),
        )
        network.apply(network.initialize)
        return network


# ===========================================================================
# Trainer 2: No SDF
# ===========================================================================

class nnUNetTrainerAblationNoSDF(nnUNetTrainerPriorGatedSwinRefiner):
    """
    Ablation trainer: full model minus the coarse SDF prior channel.

    Prior channels seen by PriorGateBlock: [coarse, uncertainty]
    Comparison: nnUNetTrainerPriorGatedSwinRefiner uses [coarse, uncertainty, SDF]

    Data pipeline is unchanged — the SDF channel (ch3) is loaded as normal
    but silently ignored inside PriorGatedSwinRefiner_NoSDF.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        n_blocks_per_stage = arch_init_kwargs.get(
            "n_conv_per_stage", arch_init_kwargs.get("n_blocks_per_stage")
        )
        if n_blocks_per_stage is None:
            raise KeyError("Expected n_conv_per_stage or n_blocks_per_stage in arch kwargs.")

        network = PriorGatedSwinRefiner_NoSDF(
            num_classes=num_output_channels,
            n_stages=arch_init_kwargs["n_stages"],
            features_per_stage=arch_init_kwargs["features_per_stage"],
            kernel_sizes=arch_init_kwargs["kernel_sizes"],
            strides=arch_init_kwargs["strides"],
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=arch_init_kwargs["n_conv_per_stage_decoder"],
            deep_supervision=enable_deep_supervision,
            coarse_logit_scale=1.0,
            window_size=(2, 4, 4),
        )
        network.apply(network.initialize)
        return network


# ===========================================================================
# Trainer 3: Coarse Only (no uncertainty AND no SDF)
# ===========================================================================

class nnUNetTrainerAblationCoarseOnly(nnUNetTrainerPriorGatedSwinRefiner):
    """
    Ablation trainer: only the coarse map is used as prior (no uncertainty, no SDF).

    Prior channels seen by PriorGateBlock: [coarse]
    Comparison: nnUNetTrainerPriorGatedSwinRefiner uses [coarse, uncertainty, SDF]

    This is the minimal prior ablation — equivalent to a coarse-guided
    residual refiner with no geometric or probabilistic uncertainty signal.

    Data pipeline is unchanged — ch2 and ch3 are loaded but ignored inside
    PriorGatedSwinRefiner_CoarseOnly.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        n_blocks_per_stage = arch_init_kwargs.get(
            "n_conv_per_stage", arch_init_kwargs.get("n_blocks_per_stage")
        )
        if n_blocks_per_stage is None:
            raise KeyError("Expected n_conv_per_stage or n_blocks_per_stage in arch kwargs.")

        network = PriorGatedSwinRefiner_CoarseOnly(
            num_classes=num_output_channels,
            n_stages=arch_init_kwargs["n_stages"],
            features_per_stage=arch_init_kwargs["features_per_stage"],
            kernel_sizes=arch_init_kwargs["kernel_sizes"],
            strides=arch_init_kwargs["strides"],
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=arch_init_kwargs["n_conv_per_stage_decoder"],
            deep_supervision=enable_deep_supervision,
            coarse_logit_scale=1.0,
            window_size=(2, 4, 4),
        )
        network.apply(network.initialize)
        return network
