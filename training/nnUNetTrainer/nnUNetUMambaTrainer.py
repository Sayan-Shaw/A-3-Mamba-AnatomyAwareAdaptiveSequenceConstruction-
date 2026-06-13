# pyright: reportMissingImports=false
# /storage/ss_sayan/Mamba3D-MedSeg-main/nnunetv2/training/nnUNetTrainer/nnUNetUMambaTrainer.py

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.preprocessing.preprocessors.umamba_preprocessor import UMambaPreprocessor
from nnunetv2.training.dataloading.umamba_dataloader import UMambaDataLoader3D
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import (
    LimitedLenWrapper,
)
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import (
    DownsampleSegForDSTransform2,
)
import numpy as np
import torch
from typing import List, Tuple, Union
from pathlib import Path
import sys

# ── U-Mamba local import ──────────────────────────────────────────────────────
_UMAMBA_DIR = Path(__file__).resolve().parents[2] / "nets_bladder_Onion_3d_uncertainty_both_intraCluster"
if str(_UMAMBA_DIR) not in sys.path:
    sys.path.insert(0, str(_UMAMBA_DIR))

from Vmamba_stageC_MS_v6_3d_tri import Mamba_segnet  # noqa: E402

# ── Channel indices ───────────────────────────────────────────────────────────
CH_MRI         = 0   # MRI image
CH_COARSE      = 1   # coarse / pseudo mask
CH_UNCERTAINTY = 2   # uncertainty map


# ─────────────────────────────────────────────────────────────────────────────
# Transform : intensity augmentation on MRI channel ONLY
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAwareIntensityTransform(AbstractTransform):
    """
    Applies intensity augmentation exclusively to channel 0 (MRI).
    Channels 1 (coarse) and 2 (uncertainty) are passed through unchanged.
    """

    def __init__(self, p_per_sample: float = 0.15):
        self._pipeline = Compose([
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_sample=p_per_sample,
                p_per_channel=0.5,
            ),
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_channel=0.5,
                p_per_sample=p_per_sample,
            ),
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25),
                p_per_sample=p_per_sample,
            ),
            ContrastAugmentationTransform(
                contrast_range=(0.75, 1.25),
                preserve_range=True,
                p_per_sample=p_per_sample,
            ),
            GammaTransform(
                gamma_range=(0.7, 1.5),
                invert_image=False,
                retain_stats=True,
                p_per_sample=p_per_sample,
            ),
            GammaTransform(
                gamma_range=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
                p_per_sample=p_per_sample,
            ),
        ])

    def __call__(self, **data_dict):
        data     = data_dict["data"]                              # (B, C, D, H, W)
        mri_dict = {"data": data[:, CH_MRI : CH_MRI + 1].copy()}
        mri_dict = self._pipeline(**mri_dict)
        data[:, CH_MRI : CH_MRI + 1] = mri_dict["data"]
        data_dict["data"] = data
        return data_dict


# ─────────────────────────────────────────────────────────────────────────────
# Network wrapper
# ─────────────────────────────────────────────────────────────────────────────
class UMambaSegnetWrapper(nn.Module):
    """
    nnU-Net passes a single (B, 3, D, H, W) tensor.
    Mamba_segnet expects (mri, mask, uncertainty) as separate tensors.
    """

    def __init__(self, net: Mamba_segnet):
        super().__init__()
        self.net   = net
        self.model = net.model   # owns the deep_supervision flag

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     if x.ndim != 5:
    #         raise ValueError(f"Expected [B, C, D, H, W], got ndim={x.ndim}")
    #     if x.shape[1] < 3:
    #         raise ValueError(
    #             f"Expected ≥3 channels [MRI, coarse, uncertainty], got {x.shape[1]}"
    #         )
    #     mri         = x[:, 0:1].float()
    #     mask        = x[:, 1:2].float()
    #     uncertainty = x[:, 2:3].float()
    #     return self.net(mri, mask, uncertainty)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"DEBUG input shape: {x.shape}")   # ← add this temporarily
        mri         = x[:, 0:1].float()
        mask        = x[:, 1:2].float()
        uncertainty = x[:, 2:3].float()
        # print(f"DEBUG mri: {mri.shape}, mask: {mask.shape}, uncertainty: {uncertainty.shape}")
        return self.net(mri, mask, uncertainty)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class nnUNetTrainerUMamba(nnUNetTrainer):

    # ── 1. network ────────────────────────────────────────────────────────
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)

        if len(configuration_manager.patch_size) != 3:
            raise NotImplementedError("Only 3-D models are supported.")

        mamba_segnet = Mamba_segnet(
            norm_cfg         = "IN",
            activation_cfg   = "LeakyReLU",
            img_size         = (64,128,128),
            num_classes      = label_manager.num_segmentation_heads,
            weight_std       = False,
            deep_supervision = enable_deep_supervision,
            bimamba          = False,
            rand             = False,
            debi             = False,
            derand           = False,
            sc_en            = ["M", "M", "M", "M"],
            sc               = ["M", "M", "M"],
            depths           = [3, 4, 3],
            depths_en        = [2, 3, 4, 3],
        )
        return UMambaSegnetWrapper(mamba_segnet)

    # ── 2. augmentation pipeline ──────────────────────────────────────────
    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int, ...]],
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg:  int = 1,
        border_val_seg:        int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded:       bool = False,
        foreground_labels        = None,
        regions                  = None,
        ignore_label             = None,
    ) -> AbstractTransform:

        transforms = []

        # ── mirror / flip only (never changes size) ──────────────────────
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(mirror_axes))

        # ── intensity aug: MRI only ──────────────────────────────────────
        transforms.append(ChannelAwareIntensityTransform(p_per_sample=0.15))

        # ── mask-based normalisation ─────────────────────────────────────
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskTransform(
                    apply_to_channels=[i for i, m in enumerate(use_mask_for_norm) if m],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        # ── deep supervision target downsampling ─────────────────────────
        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales,
                    order=0,
                    input_key="seg",
                    output_key="target",
                )
            )

        # ── numpy → tensor ───────────────────────────────────────────────
        transforms.append(NumpyToTensor(["data", "seg"], "float"))

        return Compose(transforms)

    # ── 3. train step ─────────────────────────────────────────────────────
    def train_step(self, batch: dict) -> dict:
        data   = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [
                torch.from_numpy(t).to(self.device, non_blocking=True)
                if isinstance(t, np.ndarray)
                else t.to(self.device, non_blocking=True)
                for t in target
            ]
            target = [t.clamp(min=0) for t in target]   # ← added
        else:
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target).to(self.device, non_blocking=True)
            else:
                target = target.to(self.device, non_blocking=True)
            target = target.clamp(min=0)                 # ← added

        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ── 4. validation step ────────────────────────────────────────────────
    def validation_step(self, batch: dict) -> dict:
        data   = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [
                torch.from_numpy(t).to(self.device, non_blocking=True)
                if isinstance(t, np.ndarray)
                else t.to(self.device, non_blocking=True)
                for t in target
            ]
            target = [t.clamp(min=0) for t in target]   # ← added
        else:
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target).to(self.device, non_blocking=True)
            else:
                target = target.to(self.device, non_blocking=True)
            target = target.clamp(min=0)                 # ← added

        output = self.network(data)
        del data
        l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask   = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss":    l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    # ── 5. optimiser ──────────────────────────────────────────────────────
    def configure_optimizers(self):
        self.initial_lr = 1e-4
        optimizer    = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        print(optimizer)
        return optimizer, lr_scheduler

    # ── 6. deep supervision toggle ────────────────────────────────────────
    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            self.network.module.model.deep_supervision = enabled
        else:
            self.network.model.deep_supervision = enabled

    # ── 7. preprocessor ───────────────────────────────────────────────────
    @staticmethod
    def get_preprocessor_class():
        return UMambaPreprocessor

    # ── 8. dataloaders ────────────────────────────────────────────────────
    # def get_plain_dataloaders(self, initial_patch_size, dim):
    #     if dim != 3:
    #         raise NotImplementedError(f"UMambaDataLoader3D only supports 3D. Got dim={dim}.")

    #     dataset_tr, dataset_val = self.get_tr_and_val_datasets()

    #     dl_tr = UMambaDataLoader3D(
    #         dataset_tr,
    #         self.batch_size,
    #         self.configuration_manager.patch_size,  # ← use final patch size directly
    #         self.configuration_manager.patch_size,
    #         self.label_manager,
    #         oversample_foreground_percent=self.oversample_foreground_percent,
    #         sampling_probabilities=None,
    #         pad_sides=None,
    #     )
    #     dl_val = UMambaDataLoader3D(
    #         dataset_val,
    #         self.batch_size,
    #         self.configuration_manager.patch_size,
    #         self.configuration_manager.patch_size,
    #         self.label_manager,
    #         oversample_foreground_percent=self.oversample_foreground_percent,
    #         sampling_probabilities=None,
    #         pad_sides=None,
    #     )
    #     return dl_tr, dl_val

    def get_plain_dataloaders(self, initial_patch_size, dim):
        if dim != 3:
            raise NotImplementedError(f"UMambaDataLoader3D only supports 3D. Got dim={dim}.")

        # ── hardcode final patch size to match U-Mamba img_size ──────────
        UMAMBA_PATCH_SIZE = [64, 128, 128]

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = UMambaDataLoader3D(
            dataset_tr,
            self.batch_size,
            UMAMBA_PATCH_SIZE,
            UMAMBA_PATCH_SIZE,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = UMambaDataLoader3D(
            dataset_val,
            self.batch_size,
            UMAMBA_PATCH_SIZE,
            UMAMBA_PATCH_SIZE,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim        = len(patch_size)

        downsampe_scales = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        deep_supervision_scales = None
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1, 1, 1]] + list(
                list(i)
                for i in 1 / np.cumprod(np.vstack(downsampe_scales), axis=0)
            )[:-1]

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val   = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )

        return mt_gen_train, mt_gen_val


# alias
nnUNetUMambaTrainer = nnUNetTrainerUMamba