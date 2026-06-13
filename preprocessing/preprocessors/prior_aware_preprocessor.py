from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import write_pickle
from scipy.ndimage import distance_transform_edt

from nnunetv2.preprocessing.preprocessors.default_preprocessor2 import INTERMEDIATE_SHAPE
from nnunetv2.preprocessing.preprocessors.umamba_preprocessor import UMambaPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def compute_signed_distance(mask: np.ndarray, max_distance: float = 20.0) -> np.ndarray:
    mask = np.asarray(mask > 0.5, dtype=bool)
    if mask.size == 0:
        return mask.astype(np.float32)
    if mask.any():
        inside = distance_transform_edt(mask)
    else:
        inside = np.zeros(mask.shape, dtype=np.float32)
    if (~mask).any():
        outside = distance_transform_edt(~mask)
    else:
        outside = np.zeros(mask.shape, dtype=np.float32)
    sdf = inside - outside
    sdf = np.clip(sdf, -max_distance, max_distance) / max_distance
    return sdf.astype(np.float32)


class PriorAwareRefinerPreprocessor(UMambaPreprocessor):
    """
    UMamba-compatible preprocessor plus the fourth prior channel.

    Training .npz files keep the original nnU-Net image under ``data`` and add:
      coarse, uncertainty, coarse_sdf

    Raw inference returns a four-channel tensor directly:
      MRI, coarse, uncertainty, coarse_sdf
    """

    SDF_KEY = "coarse_sdf"

    @classmethod
    def _final_shape(cls) -> Tuple[int, int, int]:
        return tuple(int(INTERMEDIATE_SHAPE[axis]) for axis in cls.FINAL_TRANSPOSE)

    def _preprocess_image_to_final_shape(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        """
        Match the tensor contract used by the saved prior-aware training cases.

        Existing models were trained and validated on tensors in final
        (D, H, W) = (64, 128, 128) space. Running DefaultPreprocessor2 first to
        (128, 64, 128) and then transposing at raw inference swaps the MRI axis
        semantics relative to the saved .npz tensors. We therefore normalize and
        resample image/seg directly into the final tensor shape, while keeping
        the normal nnU-Net properties so export still returns original geometry.
        """
        if isinstance(dataset_json, str):
            from batchgenerators.utilities.file_and_folder_operations import load_json

            dataset_json = load_json(dataset_json)

        data = np.copy(data)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], (
                "Shape mismatch between image and segmentation. "
                "Use --verify_dataset_integrity to debug."
            )
            seg = np.copy(seg)

        has_seg = seg is not None
        original_spacing = list(properties["spacing"])
        properties["shape_before_cropping"] = data.shape[1:]

        # Keep MRI, segmentation, coarse, uncertainty, and SDF in one shared
        # image grid. Sidecars are loaded later from raw files, so this
        # preprocessor intentionally does not crop the image tensor.
        bbox = [[0, int(dim)] for dim in data.shape[1:]]
        properties["bbox_used_for_cropping"] = bbox
        properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

        # Raw inference has no segmentation. DefaultPreprocessor2._normalize
        # still passes seg[0] into the normalizer, so provide a neutral mask
        # only to satisfy that signature. This does not inject GT information:
        # use_mask_for_norm is controlled by the plans and is False here.
        norm_seg = seg
        if norm_seg is None:
            norm_seg = np.zeros((1, *data.shape[1:]), dtype=np.int16)

        data = self._normalize(
            data,
            norm_seg,
            configuration_manager,
            plans_manager.foreground_intensity_properties_per_channel,
        )

        target_shape = self._final_shape()
        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing

        data = configuration_manager.resampling_fn_data(
            data,
            target_shape,
            original_spacing,
            target_spacing,
        )
        if seg is not None:
            seg = configuration_manager.resampling_fn_seg(
                seg,
                target_shape,
                original_spacing,
                target_spacing,
            )

        if has_seg:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = (
                label_manager.foreground_regions
                if label_manager.has_regions
                else label_manager.foreground_labels
            )
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)
            properties["class_locations"] = self._sample_foreground_locations(
                seg,
                collect_for_this,
                verbose=self.verbose,
            )
            seg = self.modify_seg_fn(
                seg,
                plans_manager,
                dataset_json,
                configuration_manager,
            )
            seg = seg.astype(np.int16) if np.max(seg) > 127 else seg.astype(np.int8)

        return data, seg, properties

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: list[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict,
    ):
        data, seg, properties = self.run_case(
            image_files,
            seg_file,
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        coarse = data[1:2].astype(np.float32, copy=False)
        uncertainty = data[2:3].astype(np.float32, copy=False)
        coarse_sdf = data[3:4].astype(np.float32, copy=False)
        np.savez_compressed(
            output_filename_truncated + ".npz",
            data=data,
            seg=seg,
            coarse=coarse.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
            coarse_sdf=coarse_sdf.astype(np.float32),
        )
        write_pickle(properties, output_filename_truncated + ".pkl")

    def run_case(
        self,
        image_files: list[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        if isinstance(dataset_json, str):
            from batchgenerators.utilities.file_and_folder_operations import load_json

            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()
        data, properties = rw.read_images(image_files)
        seg = None
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)

        data, seg, properties = self._preprocess_image_to_final_shape(
            data,
            seg,
            properties,
            plans_manager,
            configuration_manager,
            dataset_json,
        )

        sidecars = self._load_sidecars_for_image(image_files[0])
        if sidecars is None:
            case_id = self._case_id_from_image_file(image_files[0])
            raise FileNotFoundError(
                f"Prior-aware inference requires coarse and uncertainty sidecars for {case_id}. "
                "Expected folders such as coarseTs/coarseTr and uncertaintyTs/uncertaintyTr next to images."
            )

        coarse_path, uncertainty_path = sidecars
        coarse = self._load_and_resample_nifti(coarse_path, INTERMEDIATE_SHAPE, order=0)
        uncertainty = self._load_and_resample_nifti(uncertainty_path, INTERMEDIATE_SHAPE, order=1)
        coarse = coarse.transpose(self.FINAL_TRANSPOSE)
        uncertainty = uncertainty.transpose(self.FINAL_TRANSPOSE)
        target_shape = data.shape[1:]
        if coarse.shape != target_shape or uncertainty.shape != target_shape:
            raise RuntimeError(
                f"Prior sidecar shape mismatch for {image_files[0]}: "
                f"image={target_shape}, coarse={coarse.shape}, uncertainty={uncertainty.shape}"
            )
        coarse_sdf = compute_signed_distance(coarse)
        data = np.concatenate(
            [
                data.astype(np.float32, copy=False),
                coarse[None].astype(np.float32),
                uncertainty[None].astype(np.float32),
                coarse_sdf[None].astype(np.float32),
            ],
            axis=0,
        )
        return data, seg, properties

    @staticmethod
    def _load_sidecars_for_image(image_file: str) -> Optional[tuple[Path, Path]]:
        image_path = Path(image_file)
        case_id = UMambaPreprocessor._case_id_from_image_file(image_file)
        dataset_root = image_path.parent.parent
        coarse_folders = ("coarseTs", "coarseTr", "coarse")
        uncertainty_folders = ("uncertaintyTs", "uncertaintyTr", "uncertainty")

        coarse_path = None
        uncertainty_path = None
        for folder in coarse_folders:
            candidate = dataset_root / folder / f"{case_id}.nii.gz"
            if candidate.exists():
                coarse_path = candidate
                break
        for folder in uncertainty_folders:
            candidate = dataset_root / folder / f"{case_id}.nii.gz"
            if candidate.exists():
                uncertainty_path = candidate
                break

        if coarse_path is None and uncertainty_path is None:
            return None
        if coarse_path is None or uncertainty_path is None:
            raise FileNotFoundError(
                f"Found only one sidecar for {case_id}. Need both coarse and uncertainty maps."
            )
        return coarse_path, uncertainty_path


class UM_no_boundingBox_Preprocessor(PriorAwareRefinerPreprocessor):
    """Compatibility alias for the CIMP plans already using this preprocessor name."""
