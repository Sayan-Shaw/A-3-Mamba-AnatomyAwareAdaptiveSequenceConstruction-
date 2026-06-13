# # nnunetv2/preprocessing/preprocessors/umamba_preprocessor.py
# from __future__ import annotations

# import numpy as np
# from pathlib import Path
# from typing import Union, Tuple

# import nibabel as nib
# from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
# from nnunetv2.utilities.plans_handling.plans_handler import (
#     ConfigurationManager,
#     PlansManager,
# )
# from batchgenerators.utilities.file_and_folder_operations import save_pickle


# class UMambaPreprocessor(DefaultPreprocessor):
#     """
#     Extends DefaultPreprocessor to additionally read coarse mask and
#     uncertainty map, resample them appropriately, and save them into
#     the output .npz alongside the standard 'data' and 'seg' arrays.

#     Output .npz keys
#     ----------------
#     data          : (1, D, H, W)  float32  — MRI, resampled + normalised by nnUNet
#     seg           : (1, D, H, W)  int16    — ground-truth label
#     coarse        : (1, D, H, W)  float32  — coarse mask, nearest-neighbour resampled
#     uncertainty   : (1, D, H, W)  float32  — uncertainty map, linear resampled
#     """

#     # Folder names (relative to dataset raw root)
#     COARSE_FOLDER      = "coarseTr"
#     UNCERTAINTY_FOLDER = "uncertaintyTr"

#     def run_case_save(
#         self,
#         output_filename_truncated: str,
#         image_files: list[str],
#         seg_file: Union[str, None],
#         plans_manager: PlansManager,
#         configuration_manager: ConfigurationManager,
#         dataset_json: dict,
#     ):
#         """
#         Override the parent method so we can inject coarse + uncertainty
#         into the saved .npz after the standard preprocessing is done.
#         """
#         # ── 1. Standard nnUNet preprocessing (writes data + seg) ─────────
#         super().run_case_save(
#             output_filename_truncated,
#             image_files,
#             seg_file,
#             plans_manager,
#             configuration_manager,
#             dataset_json,
#         )

#         # ── 2. Derive paths for coarse / uncertainty ──────────────────────
#         # image_files[0] is e.g. ".../imagesTr/case_001_0000.nii.gz"
#         # We infer the case identifier (e.g. "case_001") from it.
#         case_id = self._case_id_from_image_file(image_files[0])
#         dataset_root = Path(image_files[0]).parent.parent   # .../DatasetXXX_Foo/

#         coarse_path      = dataset_root / self.COARSE_FOLDER      / f"{case_id}.nii.gz"
#         uncertainty_path = dataset_root / self.UNCERTAINTY_FOLDER / f"{case_id}.nii.gz"

#         if not coarse_path.exists():
#             raise FileNotFoundError(
#                 f"Expected coarse mask at {coarse_path}. "
#                 "Make sure coarseTr/ exists in your dataset folder."
#             )
#         if not uncertainty_path.exists():
#             raise FileNotFoundError(
#                 f"Expected uncertainty map at {uncertainty_path}. "
#                 "Make sure uncertaintyTr/ exists in your dataset folder."
#             )

#         # ── 3. Load the already-saved .npz so we know target shape ───────
#         npz_path = output_filename_truncated + ".npz"
#         saved    = np.load(npz_path)
#         data     = saved["data"]   # (C, D, H, W)
#         seg      = saved["seg"]    # (1, D, H, W)
#         target_shape = data.shape[1:]   # (D, H, W)

#         # ── 4. Load + resample coarse (nearest-neighbour) ─────────────────
#         coarse_arr = self._load_and_resample_nifti(
#             coarse_path,
#             target_shape,
#             order=0,          # nearest-neighbour — preserves binary mask
#         )

#         # ── 5. Load + resample uncertainty (linear) ───────────────────────
#         uncertainty_arr = self._load_and_resample_nifti(
#             uncertainty_path,
#             target_shape,
#             order=1,          # linear — preserves continuous probability map
#         )

#         # ── 6. Re-save .npz with the extra arrays ────────────────────────
#         np.savez_compressed(
#             npz_path,
#             data        = data,
#             seg         = seg,
#             coarse      = coarse_arr[np.newaxis].astype(np.uint8),      # (1,D,H,W)
#             uncertainty = uncertainty_arr[np.newaxis].astype(np.float32), # (1,D,H,W)
#         )

#     # ── helpers ───────────────────────────────────────────────────────────

#     @staticmethod
#     def _case_id_from_image_file(image_file: str) -> str:
#         """
#         "…/imagesTr/case_001_0000.nii.gz"  →  "case_001"

#         Strips the _XXXX channel suffix and the file extension.
#         """
#         stem = Path(image_file).name                # "case_001_0000.nii.gz"
#         stem = stem.replace(".nii.gz", "").replace(".nii", "").replace(".nrrd", "")
#         # Remove trailing _0000 / _0001 etc.
#         parts = stem.rsplit("_", 1)
#         if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
#             return parts[0]
#         return stem   # fallback if no channel suffix

#     @staticmethod
#     def _load_and_resample_nifti(
#         path: Path,
#         target_shape: Tuple[int, int, int],
#         order: int,
#     ) -> np.ndarray:
#         """
#         Load a NIfTI volume and resample to target_shape using
#         scipy zoom with the requested interpolation order.

#         Returns a (D, H, W) float32 array.
#         """
#         from scipy.ndimage import zoom

#         img  = nib.load(str(path))
#         arr  = img.get_fdata(dtype=np.float32)  # (H, W, D) or (D, H, W) — nib is HWD

#         # nibabel loads as (X, Y, Z) = (W, H, D); we want (D, H, W) to match nnUNet
#         arr = arr.transpose(2, 1, 0)            # → (D, H, W)

#         if arr.shape != target_shape:
#             zoom_factors = tuple(t / s for t, s in zip(target_shape, arr.shape))
#             arr = zoom(arr, zoom_factors, order=order, prefilter=order > 1)

#         return arr


# nnunetv2/preprocessing/preprocessors/umamba_preprocessor.py
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union, Tuple

import nibabel as nib
# ── change 1: inherit from DefaultPreprocessor2 instead of DefaultPreprocessor
from nnunetv2.preprocessing.preprocessors.default_preprocessor2 import (
    DefaultPreprocessor2,
    INTERMEDIATE_SHAPE,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from scipy.ndimage import zoom


class UMambaPreprocessor(DefaultPreprocessor2):   # ← change 2: was DefaultPreprocessor
    """
    Extends DefaultPreprocessor2 (which forces all cases to INTERMEDIATE_SHAPE).
    Additionally:
      - loads coarse + uncertainty sidecars
      - resamples them to INTERMEDIATE_SHAPE
      - applies final transpose to get (64, 128, 128) for ALL arrays
      - saves data, seg, coarse, uncertainty into .npz
    """

    COARSE_FOLDER      = "coarseTr"
    UNCERTAINTY_FOLDER = "uncertaintyTr"
    FINAL_TRANSPOSE    = (1, 0, 2)     # (128,64,128) → (64,128,128)

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: list[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict,
    ):
        # ── 1. DefaultPreprocessor2 pipeline → fixed (128,64,128) ────────
        super().run_case_save(
            output_filename_truncated,
            image_files,
            seg_file,
            plans_manager,
            configuration_manager,
            dataset_json,
        )

        # ── 2. derive paths for coarse / uncertainty ──────────────────────
        case_id      = self._case_id_from_image_file(image_files[0])
        dataset_root = Path(image_files[0]).parent.parent

        coarse_path      = dataset_root / self.COARSE_FOLDER      / f"{case_id}.nii.gz"
        uncertainty_path = dataset_root / self.UNCERTAINTY_FOLDER / f"{case_id}.nii.gz"

        if not coarse_path.exists():
            raise FileNotFoundError(
                f"Expected coarse mask at {coarse_path}. "
                "Make sure coarseTr/ exists in your dataset folder."
            )
        if not uncertainty_path.exists():
            raise FileNotFoundError(
                f"Expected uncertainty map at {uncertainty_path}. "
                "Make sure uncertaintyTr/ exists in your dataset folder."
            )

        # ── 3. load saved .npz ────────────────────────────────────────────
        npz_path    = output_filename_truncated + ".npz"
        saved       = np.load(npz_path)
        data        = saved["data"]       # (1, 128, 64, 128)
        seg         = saved["seg"]        # (1, 128, 64, 128)
        inter_shape = data.shape[1:]      # should always be INTERMEDIATE_SHAPE

        # assert inter_shape == INTERMEDIATE_SHAPE, \
        #     f"Expected {INTERMEDIATE_SHAPE}, got {inter_shape}"
        inter_shape = data.shape[1:]
        if inter_shape != INTERMEDIATE_SHAPE:
            from scipy.ndimage import zoom as scipy_zoom
            print(f"WARNING: shape {inter_shape} != {INTERMEDIATE_SHAPE}, force resampling")
            zf   = tuple(t / s for t, s in zip(INTERMEDIATE_SHAPE, inter_shape))
            data = np.stack([scipy_zoom(data[c], zf, order=1) for c in range(data.shape[0])], axis=0)
            seg  = np.stack([scipy_zoom(seg[c],  zf, order=0) for c in range(seg.shape[0])],  axis=0)
            inter_shape = INTERMEDIATE_SHAPE

        # ── 4. load + resample sidecars to INTERMEDIATE_SHAPE ─────────────
        coarse_arr      = self._load_and_resample_nifti(
            coarse_path,
            INTERMEDIATE_SHAPE,
            order=0,    # nearest-neighbour — preserves binary mask
        )
        uncertainty_arr = self._load_and_resample_nifti(
            uncertainty_path,
            INTERMEDIATE_SHAPE,
            order=1,    # linear — preserves continuous probability map
        )
        # shapes now: (128, 64, 128) each

        # ── 5. final transpose ALL arrays: (128,64,128) → (64,128,128) ───
        t4d = (0,) + tuple(ax + 1 for ax in self.FINAL_TRANSPOSE)  # (0,2,1,3)

        data            = data.transpose(t4d)
        seg             = seg.transpose(t4d)
        coarse_arr      = coarse_arr.transpose(self.FINAL_TRANSPOSE)
        uncertainty_arr = uncertainty_arr.transpose(self.FINAL_TRANSPOSE)

        if self.verbose:
            print(
                f'{case_id} | '
                f'intermediate: {inter_shape} | '
                f'final: {data.shape[1:]} | '
                f'coarse: {coarse_arr.shape} | '
                f'uncertainty: {uncertainty_arr.shape}'
            )

        # ── 6. re-save .npz with all 4 arrays ────────────────────────────
        np.savez_compressed(
            npz_path,
            data        = data,
            seg         = seg,
            coarse      = coarse_arr[np.newaxis].astype(np.uint8),
            uncertainty = uncertainty_arr[np.newaxis].astype(np.float32),
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _case_id_from_image_file(image_file: str) -> str:
        stem  = Path(image_file).name
        stem  = stem.replace(".nii.gz", "").replace(".nii", "").replace(".nrrd", "")
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
            return parts[0]
        return stem

    @staticmethod
    def _load_and_resample_nifti(
        path: Path,
        target_shape: Tuple[int, int, int],
        order: int,
    ) -> np.ndarray:
        """
        Load NIfTI, apply same transposes as nnUNet does to MRI,
        then resample to target_shape.

        Transposes applied:
          nibabel (W,H,D) → (D,H,W)      via transpose(2,1,0)
          transpose_forward=[1,0,2]       via transpose(1,0,2)
          result: (128,64,128) = INTERMEDIATE_SHAPE
        """
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
        arr = arr.transpose(2, 1, 0)    # nibabel: (W,H,D) → (D,H,W)
        arr = arr.transpose(1, 0, 2)    # transpose_forward=[1,0,2]: → (128,64,128)

        if arr.shape != target_shape:
            zoom_factors = tuple(t / s for t, s in zip(target_shape, arr.shape))
            arr = zoom(arr, zoom_factors, order=order, prefilter=order > 1)

        return arr