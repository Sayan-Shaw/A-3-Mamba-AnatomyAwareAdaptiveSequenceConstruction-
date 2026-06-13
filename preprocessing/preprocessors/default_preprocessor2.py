# nnunetv2/preprocessing/preprocessors/default_preprocessor2.py

import multiprocessing
import shutil
from time import sleep
from typing import Union, Tuple, List

import nnunetv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from tqdm import tqdm


# ── fixed intermediate shape (after nnUNet pipeline, before our final transpose)
INTERMEDIATE_SHAPE = (128, 64, 128)   # (D, H, W) after transpose_forward=[1,0,2]


class DefaultPreprocessor2(object):
    """
    Same as DefaultPreprocessor but resampling always targets
    INTERMEDIATE_SHAPE = (128, 64, 128) regardless of crop output.

    This fixes the inconsistency where crop_to_nonzero gives different
    shapes per case (e.g. 128,52,128 vs 128,64,128).

    Pipeline:
        transpose_forward  →  crop to nonzero  →  normalize
        →  resample to FIXED (128,64,128)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        data = np.copy(data)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], (
                "Shape mismatch between image and segmentation. "
                "Use --verify_dataset_integrity to debug."
            )
            seg = np.copy(seg)

        has_seg = seg is not None

        # ── 1. transpose ─────────────────────────────────────────────────
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # ── 2. crop to nonzero ───────────────────────────────────────────
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # ── 3. spacing setup ─────────────────────────────────────────────
        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing

        # ── 4. normalize (MUST happen before resampling) ─────────────────
        data = self._normalize(
            data, seg, configuration_manager,
            plans_manager.foreground_intensity_properties_per_channel,
        )

        # ── 5. resample to FIXED shape (ignore spacing-derived shape) ─────
        old_shape = data.shape[1:]
        new_shape = INTERMEDIATE_SHAPE      # <── always (128, 64, 128)

        # spacing-derived shape kept only for logging
        spacing_derived_shape = compute_new_shape(
            data.shape[1:], original_spacing, target_spacing
        )

        data = configuration_manager.resampling_fn_data(
            data, new_shape, original_spacing, target_spacing
        )
        seg = configuration_manager.resampling_fn_seg(
            seg, new_shape, original_spacing, target_spacing
        )

        if self.verbose:
            print(
                f'shape after crop: {old_shape} | '
                f'spacing-derived shape: {spacing_derived_shape} | '
                f'forced to: {new_shape}'
            )

        # ── 6. foreground locations for oversampling ──────────────────────
        if has_seg:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = (
                label_manager.foreground_regions
                if label_manager.has_regions
                else label_manager.foreground_labels
            )
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            properties['class_locations'] = self._sample_foreground_locations(
                seg, collect_for_this, verbose=self.verbose
            )
            seg = self.modify_seg_fn(
                seg, plans_manager, dataset_json, configuration_manager
            )

        seg = seg.astype(np.int16) if np.max(seg) > 127 else seg.astype(np.int8)
        return data, seg

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()
        data, data_properties = rw.read_images(image_files)

        seg = None
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)

        data, seg = self.run_case_npy(
            data, seg, data_properties,
            plans_manager, configuration_manager, dataset_json,
        )
        return data, seg, data_properties

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: List[str],
        seg_file: str,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        data, seg, properties = self.run_case(
            image_files, seg_file,
            plans_manager, configuration_manager, dataset_json,
        )
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def _sample_foreground_locations(
        seg: np.ndarray,
        classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
        seed: int = 1234,
        verbose: bool = False,
    ):
        num_samples = 10000
        min_percent_coverage = 0.01
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(
                target_num_samples,
                int(np.ceil(len(all_locs) * min_percent_coverage)),
            )
            selected = all_locs[
                rndst.choice(len(all_locs), target_num_samples, replace=False)
            ]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs

    def _normalize(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        configuration_manager: ConfigurationManager,
        foreground_intensity_properties_per_channel: dict,
    ) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(
                join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                scheme,
                'nnunetv2.preprocessing.normalization',
            )
            if normalizer_class is None:
                raise RuntimeError(f"Unable to locate class '{scheme}' for normalization")
            normalizer = normalizer_class(
                use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                intensityproperties=foreground_intensity_properties_per_channel[str(c)],
            )
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def run(
        self,
        dataset_name_or_id: Union[int, str],
        configuration_name: str,
        plans_identifier: str,
        num_processes: int,
    ):
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), \
            "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), \
            f"Plans file not found: {plans_file}. Run nnUNet_plan_experiment first."

        plans         = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing configuration : {configuration_name}')
            print(f'All cases forced to shape   : {INTERMEDIATE_SHAPE}')

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json      = load_json(dataset_json_file)

        output_directory = join(
            nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier
        )
        if isdir(output_directory):
            shutil.rmtree(output_directory)
        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(
            join(nnUNet_raw, dataset_name), dataset_json
        )

        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for k in dataset.keys():
                r.append(p.starmap_async(
                    self.run_case_save,
                    ((
                        join(output_directory, k),
                        dataset[k]['images'],
                        dataset[k]['label'],
                        plans_manager,
                        configuration_manager,
                        dataset_json,
                    ),)
                ))
            remaining = list(range(len(dataset)))
            workers   = [j for j in p._pool]
            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError(
                            'A background worker died. Likely OOM — try reducing -np.'
                        )
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def modify_seg_fn(
        self,
        seg: np.ndarray,
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
    ) -> np.ndarray:
        return seg