from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import nn

from nnunetv2.nets.prior_aware_refiner import EncoderStage, PriorGatedSwinResidualRefiner
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerPriorGatedSwinRefinerES150 import (
    nnUNetTrainerPriorGatedSwinRefinerES150,
)


class PriorConcatSwinResidualRefiner(PriorGatedSwinResidualRefiner):
    """
    Ablation network without multi-level prior gates.

    It keeps residual correction, Swin bottleneck, decoder, and auxiliary heads.
    The four input channels are concatenated only at the encoder input, so this
    tests prior gating against simple prior concatenation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        features_per_stage = [int(i) for i in kwargs["features_per_stage"]]
        kernel_sizes = [tuple(int(j) for j in i) for i in kwargs["kernel_sizes"]]
        strides = [tuple(int(j) for j in i) for i in kwargs["strides"]]
        n_blocks_per_stage = [int(i) for i in kwargs["n_blocks_per_stage"]]

        self.encoder[0] = EncoderStage(
            4,
            features_per_stage[0],
            n_blocks_per_stage[0],
            kernel_sizes[0],
            strides[0],
        )
        self.prior_gates = nn.ModuleList()

    def forward(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] < 4:
            raise ValueError("Expected channels [MRI, coarse, uncertainty, coarse_sdf].")

        features = x[:, 0:4].float()
        coarse = x[:, 1:2].float()

        skips: List[torch.Tensor] = []
        for stage in self.encoder:
            features = stage(features)
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


class nnUNetTrainerPriorGatedSwinNoGateRefinerES150(nnUNetTrainerPriorGatedSwinRefinerES150):
    """No-prior-gates ablation of the ES150 prior-gated Swin refiner."""

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        n_blocks_per_stage = arch_init_kwargs.get("n_conv_per_stage", arch_init_kwargs.get("n_blocks_per_stage"))
        if n_blocks_per_stage is None:
            raise KeyError("Expected n_conv_per_stage or n_blocks_per_stage in nnU-Net architecture kwargs.")

        network = PriorConcatSwinResidualRefiner(
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


PriorGatedSwinNoGateRefinerES150Trainer = nnUNetTrainerPriorGatedSwinNoGateRefinerES150
