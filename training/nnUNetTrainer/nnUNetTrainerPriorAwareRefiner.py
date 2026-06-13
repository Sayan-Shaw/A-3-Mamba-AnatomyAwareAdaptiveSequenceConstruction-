from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch.nn import functional as F
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.preprocessing.preprocessors.prior_aware_preprocessor import (
    PriorAwareRefinerPreprocessor,
    compute_signed_distance,
)
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.prior_aware_dataloader import PriorAwareRefinerDataLoader3D
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context


class ResidualCoarseLogitWrapper(nn.Module):
    """
    Wrap a standard nnU-Net backbone as a coarse-prior residual refiner.

    The backbone predicts only a foreground correction delta from the four input
    channels. For binary softmax training we compose standard two-class logits
    whose log-odds match logit(coarse) + delta.
    """

    def __init__(self, backbone: nn.Module, num_output_channels: int):
        super().__init__()
        if num_output_channels > 2:
            raise NotImplementedError("Prior residual refiner is intended for binary organ datasets.")
        self.backbone = backbone
        self.num_output_channels = int(num_output_channels)

    @staticmethod
    def _coarse_logit(coarse: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
        if coarse.shape[2:] != tuple(spatial_shape):
            coarse = F.interpolate(coarse.float(), size=spatial_shape, mode="trilinear", align_corners=False)
        coarse = coarse.clamp(0.01, 0.99)
        return torch.logit(coarse).clamp(-5, 5)

    def _compose(self, delta: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        foreground = self._coarse_logit(coarse, delta.shape[2:]) + delta
        if self.num_output_channels == 1:
            return foreground
        return torch.cat([torch.zeros_like(foreground), foreground], dim=1)

    def forward(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] < 4:
            raise ValueError("Expected four channels: MRI, coarse mask, uncertainty, coarse SDF.")
        coarse = x[:, 1:2].float()
        delta = self.backbone(x.float())
        if isinstance(delta, (list, tuple)):
            return [self._compose(d, coarse) for d in delta]
        return self._compose(delta, coarse)

    @property
    def decoder(self):
        return getattr(self.backbone, "decoder", None)


class ImageChannel0OnlyTransform(BasicTransform):
    """Apply image-only augmentation to MRI while leaving prior channels untouched."""

    def __init__(self, transform: BasicTransform):
        super().__init__()
        self.transform = transform

    def apply(self, data_dict, **params):
        image = data_dict.get("image")
        if image is None:
            return data_dict
        mri_only = self.transform(image=image[0:1].clone())["image"]
        image = image.clone()
        image[0:1] = mri_only
        data_dict["image"] = image
        return data_dict

    def __repr__(self):
        return f"{type(self).__name__}({self.transform!r})"


class CoarseMaskCorruptionTransform(BasicTransform):
    """Mildly perturb the prior so the model learns correction, not copying."""

    def __init__(self, max_shift: Sequence[int] = (1, 2, 2), p_shift: float = 0.35, p_morph: float = 0.35):
        super().__init__()
        self.max_shift = tuple(int(i) for i in max_shift)
        self.p_shift = float(p_shift)
        self.p_morph = float(p_morph)

    @staticmethod
    def _shift_3d(x: torch.Tensor, shifts: Tuple[int, int, int]) -> torch.Tensor:
        out = torch.zeros_like(x)
        src = []
        dst = []
        for axis, shift in enumerate(shifts, start=1):
            size = x.shape[axis]
            if shift >= 0:
                src.append(slice(0, max(size - shift, 0)))
                dst.append(slice(shift, size))
            else:
                src.append(slice(-shift, size))
                dst.append(slice(0, max(size + shift, 0)))
        out[(slice(None), *dst)] = x[(slice(None), *src)]
        return out

    @staticmethod
    def _boundary(mask: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool3d(mask, 3, stride=1, padding=1)
        eroded = 1 - F.max_pool3d(1 - mask, 3, stride=1, padding=1)
        return (dilated - eroded).clamp(0, 1)

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if img.shape[0] < 4:
            return img

        original = img[1:2].clone()
        coarse = original.clone()
        if torch.rand(1).item() < self.p_shift:
            shifts = tuple(
                int(torch.randint(-m, m + 1, (1,)).item()) if m > 0 else 0
                for m in self.max_shift
            )
            coarse = self._shift_3d(coarse, shifts)

        if torch.rand(1).item() < self.p_morph:
            binary = (coarse > 0.5).float()
            if torch.rand(1).item() < 0.5:
                coarse = F.max_pool3d(binary, 3, stride=1, padding=1)
            else:
                coarse = 1 - F.max_pool3d(1 - binary, 3, stride=1, padding=1)

        coarse = coarse.clamp(0, 1)
        changed = (coarse - original).abs().clamp(0, 1)
        uncertainty = torch.maximum(img[2:3], self._boundary(coarse) * 0.5)
        uncertainty = torch.maximum(uncertainty, F.max_pool3d(changed, 3, stride=1, padding=1))
        sdf_np = compute_signed_distance((coarse[0].cpu().numpy() > 0.5).astype(np.float32))
        sdf = torch.from_numpy(sdf_np).to(device=img.device, dtype=img.dtype)[None]

        img = img.clone()
        img[1:2] = coarse
        img[2:3] = uncertainty.clamp(0, 1)
        img[3:4] = sdf
        return img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation


class nnUNetTrainerPriorResidualRefiner(nnUNetTrainer):
    boundary_band_loss_weight = 0.05
    use_mixed_precision = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4
        self.weight_decay = 1e-4
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.5
        self.save_every = 10

    def initialize(self):
        super().initialize()
        self.num_input_channels = 4

    @staticmethod
    def _match_aux_shape(arr: np.ndarray, target_shape: Tuple[int, ...], order: int) -> np.ndarray:
        if arr.shape[1:] == tuple(target_shape):
            return arr
        from scipy.ndimage import zoom

        zoom_factors = tuple(t / s for t, s in zip(target_shape, arr.shape[1:]))
        return np.stack(
            [zoom(arr[c], zoom_factors, order=order, prefilter=order > 1) for c in range(arr.shape[0])],
            axis=0,
        )

    def _assemble_validation_case_input(self, dataset_val, identifier: str, data: np.ndarray) -> np.ndarray:
        """
        Final validation must use the same prior channels as the training dataloader.

        Some older preprocessed folders have stale auxiliary channels embedded in
        ``data`` while the authoritative sidecars live in the explicit npz keys.
        Building the tensor from those keys keeps final validation consistent with
        train-time pseudo-Dice.
        """
        npz_path = Path(dataset_val.source_folder) / f"{identifier}.npz"
        with np.load(npz_path) as npz:
            missing = [k for k in ("coarse", "uncertainty") if k not in npz.files]
            if missing:
                raise RuntimeError(
                    f"Validation case '{identifier}' is missing required prior keys {missing} in '{npz_path}'."
                )
            image = np.asarray(data[:1], dtype=np.float32)
            target_shape = image.shape[1:]
            coarse = npz["coarse"].astype(np.float32)
            uncertainty = npz["uncertainty"].astype(np.float32)
            coarse_sdf = (
                npz["coarse_sdf"].astype(np.float32)
                if "coarse_sdf" in npz.files
                else compute_signed_distance(coarse[0])[None]
            )

        coarse = self._match_aux_shape(coarse, target_shape, order=0)
        uncertainty = self._match_aux_shape(uncertainty, target_shape, order=1)
        coarse_sdf = self._match_aux_shape(coarse_sdf, target_shape, order=1)
        assembled = np.concatenate([image, coarse, uncertainty, coarse_sdf], axis=0).astype(np.float32, copy=False)
        if assembled.shape[0] != 4:
            raise RuntimeError(
                f"Validation case '{identifier}' assembled {assembled.shape[0]} channels; expected 4."
            )
        if not np.isfinite(assembled).all():
            raise RuntimeError(f"Validation case '{identifier}' contains non-finite input values.")
        return assembled

    def _do_i_compile(self):
        return False

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            input_channels=4,
            output_channels=1,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return ResidualCoarseLogitWrapper(backbone, num_output_channels)

    @staticmethod
    def get_preprocessor_class():
        return PriorAwareRefinerPreprocessor

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 3:
            raise NotImplementedError("Prior residual refiner supports 3D fullres training only.")
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
        rotation_for_DA = (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi)
        mirror_axes = (0, 1, 2)
        initial_patch_size = get_patch_size(
            patch_size,
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.85, 1.25),
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]
        self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms: List[BasicTransform] = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            ignore_axes = None
            patch_size_spatial = patch_size

        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.85, 1.25),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(ImageChannel0OnlyTransform(
            GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True)
        ), apply_probability=0.1))
        transforms.append(RandomTransform(ImageChannel0OnlyTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=1,
                benchmark=True,
            )
        ), apply_probability=0.2))
        transforms.append(RandomTransform(ImageChannel0OnlyTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1,
            )
        ), apply_probability=0.15))
        transforms.append(RandomTransform(ImageChannel0OnlyTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            )
        ), apply_probability=0.15))
        transforms.append(RandomTransform(ImageChannel0OnlyTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=1,
            )
        ), apply_probability=0.25))
        transforms.append(RandomTransform(ImageChannel0OnlyTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            )
        ), apply_probability=0.1))
        transforms.append(RandomTransform(ImageChannel0OnlyTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            )
        ), apply_probability=0.3))
        transforms.append(RandomTransform(CoarseMaskCorruptionTransform(), apply_probability=0.25))

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(RemoveLabelTansform(-1, 0))
        if regions is not None:
            transforms.append(ConvertSegmentationToRegionsTransform(
                regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                channel_in_seg=0,
            ))
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = PriorAwareRefinerDataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = PriorAwareRefinerDataLoader3D(
            dataset_val,
            self.batch_size,
            patch_size,
            patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        if hasattr(mod.backbone, "decoder"):
            mod.backbone.decoder.deep_supervision = enabled

    @staticmethod
    def _boundary_band(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
        mask = mask.float()
        kernel = 2 * radius + 1
        dilated = F.max_pool3d(mask, kernel, stride=1, padding=radius)
        eroded = 1 - F.max_pool3d(1 - mask, kernel, stride=1, padding=radius)
        return (dilated - eroded).clamp(0, 1)

    def _boundary_band_loss(
        self,
        output: Union[torch.Tensor, List[torch.Tensor]],
        target: Union[torch.Tensor, List[torch.Tensor]],
        data: torch.Tensor,
    ) -> torch.Tensor:
        if self.label_manager.has_regions or self.boundary_band_loss_weight <= 0:
            first = output[0] if isinstance(output, list) else output
            return first.new_tensor(0.0)

        outputs = output if isinstance(output, list) else [output]
        targets = target if isinstance(target, list) else [target]
        weights = [1 / (2 ** i) for i in range(len(outputs))]
        weights = [w / sum(weights) for w in weights]
        total = outputs[0].new_tensor(0.0)

        for weight, logits, tgt in zip(weights, outputs, targets):
            tgt_fg = (tgt > 0).float()
            coarse = F.interpolate(data[:, 1:2].float(), size=tgt.shape[2:], mode="trilinear", align_corners=False)
            band = torch.maximum(self._boundary_band(tgt_fg), self._boundary_band((coarse > 0.5).float()))
            ce = F.cross_entropy(logits, tgt[:, 0].long().clamp(min=0), reduction="none")
            denom = band[:, 0].sum().clamp_min(1.0)
            total = total + weight * (ce * band[:, 0]).sum() / denom
        return total * self.boundary_band_loss_weight

    def _autocast_context(self):
        if self.device.type != "cuda":
            return dummy_context()
        return autocast(self.device.type, enabled=self.use_mixed_precision)

    @staticmethod
    def _all_finite(x) -> bool:
        if torch.is_tensor(x):
            return bool(torch.isfinite(x).all())
        if isinstance(x, (list, tuple)):
            return all(nnUNetTrainerPriorResidualRefiner._all_finite(i) for i in x)
        return True

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            output = self.network(data)
            loss = self.loss(output, target) + self._boundary_band_loss(output, target, data)

        if not self._all_finite(output):
            raise FloatingPointError(
                f"Non-finite network output at epoch {self.current_epoch}. "
                "Aborting before optimizer step to preserve the last good checkpoint."
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite training loss at epoch {self.current_epoch}. "
                "Aborting before optimizer step to preserve the last good checkpoint."
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if not bool(torch.isfinite(grad_norm)):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {self.current_epoch}. "
                    "Aborting before optimizer step to preserve the last good checkpoint."
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if not bool(torch.isfinite(grad_norm)):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {self.current_epoch}. "
                    "Aborting before optimizer step to preserve the last good checkpoint."
                )
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with self._autocast_context():
            output = self.network(data)
            loss = self.loss(output, target) + self._boundary_band_loss(output, target, data)

        if not self._all_finite(output):
            raise FloatingPointError(
                f"Non-finite validation output at epoch {self.current_epoch}. "
                "Aborting to avoid logging misleading metrics."
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite validation loss at epoch {self.current_epoch}. "
                "Aborting to avoid logging misleading metrics."
            )

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}


class nnUNetTrainerPriorAwareRefiner(nnUNetTrainerPriorResidualRefiner):
    """Compatibility name. The default prior-aware path is now the lean residual refiner."""


class nnUNetTrainerPriorAwareRefinerNoContext(nnUNetTrainerPriorResidualRefiner):
    """Compatibility name; no global context is used in the lean refiner."""


nnUNetTrainerPriorAwareResidualRefiner = nnUNetTrainerPriorResidualRefiner
