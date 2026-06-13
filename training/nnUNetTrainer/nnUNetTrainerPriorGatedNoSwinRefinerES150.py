from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
from torch import nn

from nnunetv2.nets.prior_aware_refiner import PriorGatedSwinResidualRefiner, ResidualBlock3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerPriorGatedSwinRefinerES150 import (
    nnUNetTrainerPriorGatedSwinRefinerES150,
)


class ConvBottleneck3D(nn.Module):
    """Convolution-only bottleneck used for the no-Swin architecture ablation."""

    def __init__(self, channels: int, kernel_size: Sequence[int], n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualBlock3D(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                )
                for _ in range(int(n_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PriorGatedConvBottleneckResidualRefiner(PriorGatedSwinResidualRefiner):
    """
    Ablation network without the Swin bottleneck.

    It keeps the same MRI/coarse/uncertainty/SDF inputs, prior gates, residual
    correction around logit(coarse), decoder, and auxiliary heads. Only the
    low-resolution Swin shifted-window context block is replaced with standard
    residual 3D convolution blocks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        features_per_stage = [int(i) for i in kwargs["features_per_stage"]]
        kernel_sizes = [tuple(int(j) for j in i) for i in kwargs["kernel_sizes"]]
        self.context = ConvBottleneck3D(
            channels=features_per_stage[-1],
            kernel_size=kernel_sizes[-1],
            n_blocks=2,
        )


class nnUNetTrainerPriorGatedNoSwinRefinerES150(nnUNetTrainerPriorGatedSwinRefinerES150):
    """No-Swin-bottleneck ablation of the ES150 prior-gated residual refiner."""

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

        network = PriorGatedConvBottleneckResidualRefiner(
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


PriorGatedNoSwinRefinerES150Trainer = nnUNetTrainerPriorGatedNoSwinRefinerES150
