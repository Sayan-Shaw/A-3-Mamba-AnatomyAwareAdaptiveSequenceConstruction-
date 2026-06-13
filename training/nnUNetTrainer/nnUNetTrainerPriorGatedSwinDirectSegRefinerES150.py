from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import nn

from nnunetv2.nets.prior_aware_refiner import PriorGatedSwinResidualRefiner
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerPriorGatedSwinRefinerES150 import (
    nnUNetTrainerPriorGatedSwinRefinerES150,
)


class PriorGatedSwinDirectSegRefiner(PriorGatedSwinResidualRefiner):
    """
    Ablation network for direct segmentation.

    It keeps the same encoder, prior gates, Swin bottleneck, decoder, and
    auxiliary heads as the residual refiner, but removes the residual addition
    around logit(coarse). The segmentation head output is used directly as the
    foreground logit.
    """

    def _compose_logits(self, foreground_logit: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return foreground_logit
        return torch.cat([torch.zeros_like(foreground_logit), foreground_logit], dim=1)


class nnUNetTrainerPriorGatedSwinDirectSegRefinerES150(nnUNetTrainerPriorGatedSwinRefinerES150):
    """Direct-segmentation ablation of the ES150 prior-gated Swin refiner."""

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

        network = PriorGatedSwinDirectSegRefiner(
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


PriorGatedSwinDirectSegRefinerES150Trainer = nnUNetTrainerPriorGatedSwinDirectSegRefinerES150
