from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch._dynamo import OptimizedModule

from nnunetv2.nets.prior_aware_refiner import PriorGatedSwinResidualRefiner
from nnunetv2.preprocessing.preprocessors.prior_aware_preprocessor import compute_signed_distance
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerPriorAwareRefiner import nnUNetTrainerPriorResidualRefiner


class nnUNetTrainerPriorGatedSwinRefiner(nnUNetTrainerPriorResidualRefiner):
    """
    Paper-method trainer: prior-gated residual nnU-Net with a deterministic Swin bottleneck.

    The model receives MRI, coarse, uncertainty and coarse SDF channels. At deployment,
    the SDF channel is derived from the coarse sidecar by PriorAwareRefinerPreprocessor.
    """

    boundary_aux_loss_weight = 0.05
    sdf_aux_loss_weight = 0.02

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

        network = PriorGatedSwinResidualRefiner(
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

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        if self._is_femur_dataset():
            mirror_axes = ()
            self.inference_allowed_mirroring_axes = mirror_axes
            self.print_to_log_file("Femur dataset detected: disabling mirroring to reduce left/right confusion.")
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._unwrap_network()
        if hasattr(mod, "deep_supervision"):
            mod.deep_supervision = bool(enabled)

    def _is_femur_dataset(self) -> bool:
        plans_manager = getattr(self, "plans_manager", None)
        names = [
            str(getattr(plans_manager, "dataset_name", "")),
            str(getattr(plans_manager, "plans", {}).get("dataset_name", "") if plans_manager is not None else ""),
            str(self.dataset_json.get("name", "")),
        ]
        return any("femur" in name.lower() for name in names)

    def _unwrap_network(self) -> nn.Module:
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    @staticmethod
    def _signed_distance_target(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.float32)
        sdf = np.stack([compute_signed_distance(mask_np[i, 0]) for i in range(mask_np.shape[0])], axis=0)
        sdf = torch.from_numpy(sdf[:, None]).to(device=like.device, dtype=like.dtype)
        if sdf.shape[2:] != like.shape[2:]:
            sdf = F.interpolate(sdf, size=like.shape[2:], mode="trilinear", align_corners=False)
        return sdf.clamp(-1, 1)

    def _boundary_band_loss(
        self,
        output: Union[torch.Tensor, List[torch.Tensor]],
        target: Union[torch.Tensor, List[torch.Tensor]],
        data: torch.Tensor,
    ) -> torch.Tensor:
        first_output = output[0] if isinstance(output, list) else output
        if self.label_manager.has_regions:
            return first_output.new_tensor(0.0)

        mod = self._unwrap_network()
        boundary_logits = getattr(mod, "last_boundary_logits", None)
        sdf_logits = getattr(mod, "last_sdf_logits", None)
        if boundary_logits is None or sdf_logits is None:
            return first_output.new_tensor(0.0)

        boundary_logits = boundary_logits[0] if isinstance(boundary_logits, list) else boundary_logits
        sdf_logits = sdf_logits[0] if isinstance(sdf_logits, list) else sdf_logits
        target_full = target[0] if isinstance(target, list) else target
        target_fg = (target_full[:, 0:1] > 0).float()

        if boundary_logits.shape[2:] != target_fg.shape[2:]:
            boundary_logits = F.interpolate(
                boundary_logits,
                size=target_fg.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        if sdf_logits.shape[2:] != target_fg.shape[2:]:
            sdf_logits = F.interpolate(
                sdf_logits,
                size=target_fg.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        boundary_target = self._boundary_band(target_fg, radius=2)
        boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
        sdf_target = self._signed_distance_target(target_fg, sdf_logits)
        sdf_loss = F.smooth_l1_loss(torch.tanh(sdf_logits), sdf_target)
        return self.boundary_aux_loss_weight * boundary_loss + self.sdf_aux_loss_weight * sdf_loss


PriorGatedSwinRefinerTrainer = nnUNetTrainerPriorGatedSwinRefiner
