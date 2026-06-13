from __future__ import annotations

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerMRIOnlyFromPriorPreprocessed(nnUNetTrainer):
    """
    Standard nnU-Net baseline for prior-aware preprocessed folders.

    Dataset711/Dataset715 .npz files store four channels under "data"
    (MRI, coarse, uncertainty, coarse SDF). A plain nnU-Net trainer would
    therefore receive the prior channels too. This wrapper keeps the standard
    nnU-Net architecture/loss/splits, but slices the batch to MRI only.
    """

    @staticmethod
    def _mri_only(data):
        if torch.is_tensor(data):
            if data.ndim == 5:
                return data[:, :1]
            if data.ndim == 4:
                return data[:1]
            return data
        arr = np.asarray(data)
        return arr[:1].astype(np.float32, copy=False)

    def _do_i_compile(self):
        return False

    def train_step(self, batch: dict) -> dict:
        batch = dict(batch)
        batch["data"] = self._mri_only(batch["data"])
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        batch = dict(batch)
        batch["data"] = self._mri_only(batch["data"])
        return super().validation_step(batch)

    def _assemble_validation_case_input(self, dataset_val, identifier: str, data: np.ndarray) -> np.ndarray:
        return self._mri_only(data)


class nnUNetTrainerMRIOnlyFromPriorPreprocessedES150(nnUNetTrainerMRIOnlyFromPriorPreprocessed):
    """MRI-only standard nnU-Net baseline with a fixed 150-epoch budget."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150


class nnUNetTrainerMRIOnlyFromPriorPreprocessedES75(nnUNetTrainerMRIOnlyFromPriorPreprocessed):
    """MRI-only standard nnU-Net baseline with a fixed 75-epoch budget."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 75


class nnUNetTrainerMRIOnlyFromPriorPreprocessedES200(nnUNetTrainerMRIOnlyFromPriorPreprocessed):
    """MRI-only standard nnU-Net baseline with a fixed 200-epoch budget."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
