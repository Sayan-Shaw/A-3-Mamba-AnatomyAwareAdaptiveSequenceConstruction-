from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import join
from threadpoolctl import threadpool_limits

from nnunetv2.preprocessing.preprocessors.prior_aware_preprocessor import compute_signed_distance
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class PriorAwareRefinerDataLoader3D(nnUNetDataLoader):
    """
    nnU-Net dataloader that exposes four input channels:
      MRI, coarse mask/probability, uncertainty, coarse signed distance.

    It reads the sidecar arrays from the same .npz files created by the
    UMamba/PriorAware preprocessors. If coarse_sdf is absent in older
    preprocessed folders, it is computed on the fly from the coarse mask.
    """

    def __init__(
        self,
        data: nnUNetBaseDataset,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        label_manager: LabelManager,
        oversample_foreground_percent: float = 0.0,
        sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
        pad_sides: Union[List[int], Tuple[int, ...]] = None,
        probabilistic_oversampling: bool = False,
        transforms=None,
    ):
        super().__init__(
            data,
            batch_size,
            patch_size,
            final_patch_size,
            label_manager,
            oversample_foreground_percent,
            sampling_probabilities,
            pad_sides,
            probabilistic_oversampling,
            transforms,
        )

    def determine_shapes(self):
        if self.patch_size_was_2d:
            raise NotImplementedError("PriorAwareRefinerDataLoader3D only supports 3D training.")
        spatial_shape = self.final_patch_size if self.transforms is not None else self.patch_size
        data_shape = (self.batch_size, 4, *spatial_shape)
        _, seg, _, _ = self._data.load_case(self._data.identifiers[0])
        return data_shape, (self.batch_size, seg.shape[0], *spatial_shape)

    def _load_prior_case(self, identifier: str):
        data, seg, _, properties = self._data.load_case(identifier)
        npz_path = join(self._data.source_folder, identifier + ".npz")
        with np.load(npz_path) as npz:
            if "coarse" not in npz.files or "uncertainty" not in npz.files:
                raise RuntimeError(
                    f"{npz_path} does not contain coarse/uncertainty. "
                    "Run preprocessing with PriorAwareRefinerPreprocessor or UMambaPreprocessor."
                )
            coarse = npz["coarse"].astype(np.float32)
            uncertainty = npz["uncertainty"].astype(np.float32)
            if "coarse_sdf" in npz.files:
                coarse_sdf = npz["coarse_sdf"].astype(np.float32)
            else:
                coarse_sdf = compute_signed_distance(coarse[0])[None]

        # Some prior-aware preprocessors store the sidecars both as dedicated
        # npz keys and inside "data" for inference compatibility. Training must
        # still expose exactly four channels, so keep only the MRI image here
        # and take priors from their explicit keys.
        image = np.asarray(data[:1], dtype=np.float32)
        target_shape = image.shape[1:]
        coarse = self._match_shape(coarse, target_shape, order=0)
        uncertainty = self._match_shape(uncertainty, target_shape, order=1)
        coarse_sdf = self._match_shape(coarse_sdf, target_shape, order=1)
        stacked = np.concatenate(
            [
                image,
                coarse.astype(np.float32, copy=False),
                uncertainty.astype(np.float32, copy=False),
                coarse_sdf.astype(np.float32, copy=False),
            ],
            axis=0,
        )
        return stacked, np.asarray(seg), properties

    @staticmethod
    def _match_shape(arr: np.ndarray, target_shape: Tuple[int, ...], order: int) -> np.ndarray:
        if arr.shape[1:] == tuple(target_shape):
            return arr
        from scipy.ndimage import zoom

        zoom_factors = tuple(t / s for t, s in zip(target_shape, arr.shape[1:]))
        return np.stack(
            [zoom(arr[c], zoom_factors, order=order, prefilter=order > 1) for c in range(arr.shape[0])],
            axis=0,
        )

    @staticmethod
    def _normalize_class_locations(class_locations: dict, dim: int) -> dict:
        normalized = {}
        for key, value in class_locations.items():
            arr = np.asarray(value)
            if arr.size == 0:
                normalized[key] = value
            elif arr.ndim == 2 and arr.shape[1] == dim:
                zeros = np.zeros((arr.shape[0], 1), dtype=arr.dtype)
                normalized[key] = np.concatenate([zeros, arr], axis=1)
            else:
                normalized[key] = value
        return normalized

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = torch.empty(self.data_shape, dtype=torch.float32)
        seg_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, identifier in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, properties = self._load_prior_case(identifier)
                    shape = data.shape[1:]
                    class_locations = self._normalize_class_locations(properties["class_locations"], len(shape))
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)
                    bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

                    data_cropped_np = np.ascontiguousarray(crop_and_pad_nd(data, bbox, 0)).copy()
                    seg_cropped_np = np.ascontiguousarray(crop_and_pad_nd(seg, bbox, -1)).copy()
                    data_cropped = torch.from_numpy(data_cropped_np).float()
                    seg_cropped = torch.from_numpy(seg_cropped_np).to(torch.int16)

                    if self.transforms is not None:
                        transformed = self.transforms(image=data_cropped, segmentation=seg_cropped)
                        data_sample = transformed["image"]
                        seg_sample = transformed["segmentation"]
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    data_all[j] = data_sample
                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample

        return {"data": data_all, "target": seg_all, "keys": selected_keys}
