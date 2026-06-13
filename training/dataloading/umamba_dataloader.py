# nnunetv2/training/dataloading/umamba_dataloader.py
from __future__ import annotations

import numpy as np
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D


class UMambaDataLoader3D(nnUNetDataLoader3D):
    """
    Extends nnUNetDataLoader3D to load coarse + uncertainty from .npz
    and stack them as channels 1 and 2 alongside MRI (channel 0).

    Invalid cases (missing coarse/uncertainty or wrong shape) are
    removed at init so they are never sampled.

    Output batch["data"] shape : (B, 3, patch_D, patch_H, patch_W)
        channel 0 : MRI
        channel 1 : coarse mask
        channel 2 : uncertainty map
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ── remove invalid cases at init so they never get sampled ────
        invalid = set()
        for key in list(self._data.keys()):
            npz_path = self._data[key]['data_file']
            try:
                npz = np.load(npz_path)
                if 'coarse' not in npz.files or 'uncertainty' not in npz.files:
                    invalid.add(key)
                elif npz['data'].shape != (1, 64, 128, 128):
                    invalid.add(key)
            except Exception:
                invalid.add(key)

        if invalid:
            print(f"UMambaDataLoader3D: removing {len(invalid)} invalid cases: {sorted(invalid)}")
            for key in invalid:
                del self._data.dataset[key]
            self.indices = list(self._data.keys())

        print(f"UMambaDataLoader3D: {len(self.indices)} valid cases remaining")

    def determine_shapes(self):
        npz_path = self._data[self.indices[0]]['data_file']
        npz      = np.load(npz_path)
        seg      = npz['seg']

        data_shape = (self.batch_size, 3, *self.final_patch_size)
        seg_shape  = (self.batch_size, seg.shape[0], *self.final_patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys   = self.get_indices()
        case_properties = []

        data_all = np.zeros(
            (self.batch_size, 3, *self.final_patch_size), dtype=np.float32
        )
        seg_all = np.zeros(
            (self.batch_size, 1, *self.final_patch_size), dtype=np.int16
        )

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            # ── always load directly from .npz ────────────────────────
            npz_path   = self._data[i]['data_file']
            npz        = np.load(npz_path)
            data       = npz["data"]          # (1, D, H, W)
            seg        = npz["seg"]           # (1, D, H, W)
            coarse     = npz["coarse"]        # (1, D, H, W)
            uncert     = npz["uncertainty"]   # (1, D, H, W)
            properties = self._data[i]['properties']

            case_properties.append(properties)

            # ── stack all 3 channels before cropping ──────────────────
            data = np.concatenate([data, coarse, uncert], axis=0)  # (3,D,H,W)

            # ── crop/pad logic (same as parent) ───────────────────────
            shape = data.shape[1:]
            dim   = len(shape)

            bbox_lbs, bbox_ubs = self.get_bbox(
                shape, force_fg, properties['class_locations']
            )

            valid_bbox_lbs = [max(0, bbox_lbs[d]) for d in range(dim)]
            valid_bbox_ubs = [min(shape[d], bbox_ubs[d]) for d in range(dim)]

            this_slice = tuple(
                [slice(0, data.shape[0])] +
                [slice(lb, ub) for lb, ub in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[this_slice]

            this_slice = tuple(
                [slice(0, seg.shape[0])] +
                [slice(lb, ub) for lb, ub in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            seg = seg[this_slice]

            padding = [
                (-min(0, bbox_lbs[d]), max(bbox_ubs[d] - shape[d], 0))
                for d in range(dim)
            ]

            # ── pad ───────────────────────────────────────────────────
            padded = np.pad(
                data, ((0, 0), *padding), 'constant', constant_values=0
            )
            if padded.shape[1:] != tuple(self.final_patch_size):
                slices = tuple(
                    [slice(0, padded.shape[0])] +
                    [slice(0, s) for s in self.final_patch_size]
                )
                padded = padded[slices]
            data_all[j] = padded

            seg_padded = np.pad(
                seg, ((0, 0), *padding), 'constant', constant_values=-1
            )
            if seg_padded.shape[1:] != tuple(self.final_patch_size):
                slices = tuple(
                    [slice(0, seg_padded.shape[0])] +
                    [slice(0, s) for s in self.final_patch_size]
                )
                seg_padded = seg_padded[slices]
            seg_all[j] = seg_padded

        return {
            'data':       data_all,
            'seg':        seg_all,
            'properties': case_properties,
            'keys':       selected_keys,
        }