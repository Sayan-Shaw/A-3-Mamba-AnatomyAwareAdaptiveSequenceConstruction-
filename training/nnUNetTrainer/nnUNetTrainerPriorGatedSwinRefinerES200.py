from __future__ import annotations

import warnings
from typing import Union

import torch
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerPriorGatedSwinRefiner import (
    nnUNetTrainerPriorGatedSwinRefiner,
)


class nnUNetTrainerPriorGatedSwinRefinerES200(nnUNetTrainerPriorGatedSwinRefiner):
    """Prior-gated Swin refiner with a fixed 200-epoch budget."""

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
        self.planned_num_epochs = self.num_epochs
        self.runtime_num_epochs_override = None
        self.best_ema_pseudo_dice = None

    def _load_checkpoint_dict(self, filename_or_checkpoint: Union[dict, str]) -> dict:
        if isinstance(filename_or_checkpoint, str):
            return torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        return filename_or_checkpoint

    def on_validation_epoch_end(self, val_outputs: list):
        super().on_validation_epoch_end(val_outputs)
        current_dice = self.logger.get_value("ema_fg_dice", step=-1)
        if self.best_ema_pseudo_dice is None or current_dice > self.best_ema_pseudo_dice:
            self.best_ema_pseudo_dice = current_dice

    def run_training(self):
        return super().run_training()

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)

        if self.local_rank != 0 or self.disable_checkpointing:
            return

        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        checkpoint["planned_num_epochs"] = int(self.planned_num_epochs)
        checkpoint["best_ema_pseudo_dice"] = self.best_ema_pseudo_dice
        checkpoint["runtime_num_epochs_override"] = self.runtime_num_epochs_override
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        checkpoint = self._load_checkpoint_dict(filename_or_checkpoint)
        if isinstance(filename_or_checkpoint, str):
            super().load_checkpoint(filename_or_checkpoint)
        else:
            if not self.was_initialized:
                self.initialize()

            new_state_dict = {}
            for k, value in checkpoint["network_weights"].items():
                key = k
                if key not in self.network.state_dict().keys() and key.startswith("module."):
                    key = key[7:]
                new_state_dict[key] = value

            self.my_init_kwargs = checkpoint["init_args"]
            self.current_epoch = checkpoint["current_epoch"]
            self.logger.load_checkpoint(checkpoint["logging"])
            self._best_ema = checkpoint["_best_ema"]
            self.inference_allowed_mirroring_axes = checkpoint.get(
                "inference_allowed_mirroring_axes",
                self.inference_allowed_mirroring_axes,
            )

            if self.is_ddp:
                if isinstance(self.network.module, OptimizedModule):
                    self.network.module._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network.module.load_state_dict(new_state_dict)
            else:
                if isinstance(self.network, OptimizedModule):
                    self.network._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if self.grad_scaler is not None and checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

        planned_num_epochs = checkpoint.get("planned_num_epochs")
        if planned_num_epochs is None:
            warnings.warn(
                "Checkpoint does not include planned_num_epochs. Reusing the current trainer budget.",
                stacklevel=2,
            )
            self.planned_num_epochs = int(self.num_epochs)
        else:
            self.planned_num_epochs = int(planned_num_epochs)
            self.num_epochs = int(planned_num_epochs)

        self.best_ema_pseudo_dice = checkpoint.get(
            "best_ema_pseudo_dice",
            checkpoint.get("_best_ema", self._best_ema),
        )
        self.runtime_num_epochs_override = checkpoint.get("runtime_num_epochs_override", None)


PriorGatedSwinRefinerES200Trainer = nnUNetTrainerPriorGatedSwinRefinerES200
