# Copyright 2026 Imaging Data Commons
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
IDC-specific transforms for MONAI pipelines.

Provides transforms commonly needed when working with IDC DICOM data.

Note: MONAI's LoadImaged with ITKReader handles regular DICOM series directly.
However, DICOM Segmentation (DICOM-SEG) files require special handling - use
LoadDicomSegd which uses itkwasm-dicom for robust DICOM-SEG reading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Hashable, Mapping

import numpy as np

from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.data import MetaTensor
from monai.utils import optional_import

itkwasm_dicom, has_itkwasm = optional_import("itkwasm_dicom")


class CTWindowd(MapTransform):
    """
    Apply CT windowing (window/level) to images.

    CT images are stored in Hounsfield Units (HU). This transform applies
    windowing to focus on specific tissue types.

    Common CT windows:
    - Lung: center=-600, width=1500
    - Soft tissue/Abdomen: center=40, width=400
    - Bone: center=400, width=1800
    - Brain: center=40, width=80
    - Liver: center=60, width=150

    Args:
        keys: Keys of the images to transform.
        window_center: Center of the window in HU.
        window_width: Width of the window in HU.
        output_min: Minimum output value (default: 0.0).
        output_max: Maximum output value (default: 1.0).
        allow_missing_keys: Don't raise exception if key is missing.

    Example:
        >>> from idc_monai.transforms import CTWindowd
        >>> # Apply lung window
        >>> transform = CTWindowd(
        ...     keys=["image"],
        ...     window_center=-600,
        ...     window_width=1500,
        ... )
    """

    def __init__(
        self,
        keys: KeysCollection,
        window_center: float,
        window_width: float,
        output_min: float = 0.0,
        output_max: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.window_center = window_center
        self.window_width = window_width
        self.output_min = output_min
        self.output_max = output_max

    def __call__(self, data: Mapping[Hashable, any]) -> dict[Hashable, any]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            # Calculate window bounds
            lower = self.window_center - self.window_width / 2
            upper = self.window_center + self.window_width / 2

            # Apply windowing
            img = img.clip(lower, upper)
            # Normalize to output range
            img = (img - lower) / (upper - lower)
            img = img * (self.output_max - self.output_min) + self.output_min

            d[key] = img
        return d


# Predefined CT windows for convenience
CT_WINDOWS = {
    "lung": {"window_center": -600, "window_width": 1500},
    "soft_tissue": {"window_center": 40, "window_width": 400},
    "abdomen": {"window_center": 40, "window_width": 400},
    "bone": {"window_center": 400, "window_width": 1800},
    "brain": {"window_center": 40, "window_width": 80},
    "liver": {"window_center": 60, "window_width": 150},
    "mediastinum": {"window_center": 50, "window_width": 350},
}


def get_ct_window_transform(
    keys: KeysCollection,
    window_name: str,
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> CTWindowd:
    """
    Get a CT windowing transform for a named window preset.

    Args:
        keys: Keys of the images to transform.
        window_name: Name of the window preset (e.g., 'lung', 'bone', 'brain').
        output_min: Minimum output value.
        output_max: Maximum output value.

    Returns:
        CTWindowd transform configured for the specified window.

    Raises:
        ValueError: If window_name is not recognized.

    Example:
        >>> transform = get_ct_window_transform(["image"], "lung")
    """
    if window_name not in CT_WINDOWS:
        available = ", ".join(CT_WINDOWS.keys())
        raise ValueError(
            f"Unknown CT window '{window_name}'. Available: {available}"
        )

    window = CT_WINDOWS[window_name]
    return CTWindowd(
        keys=keys,
        window_center=window["window_center"],
        window_width=window["window_width"],
        output_min=output_min,
        output_max=output_max,
    )


class LoadDicomSegd(MapTransform):
    """
    Load DICOM Segmentation (DICOM-SEG) files using ITKWasm.

    DICOM-SEG is an enhanced multiframe DICOM format that stores segmentation
    masks. Unlike regular DICOM series (one file per slice), DICOM-SEG stores
    all slices in a single file with segment metadata.

    This transform uses itkwasm-dicom which wraps dcmqi for robust DICOM-SEG
    reading with proper spatial metadata extraction.

    Args:
        keys: Keys of the DICOM-SEG paths to load.
        allow_missing_keys: Don't raise exception if key is missing.

    Note:
        - Input: path to directory containing a single SEG .dcm file,
          or direct path to the .dcm file
        - Output: MetaTensor with shape matching the segmentation volume
        - Spatial metadata (affine) is extracted from the DICOM-SEG

    Example:
        >>> from idc_monai.transforms import LoadDicomSegd
        >>> transform = LoadDicomSegd(keys=["label"])
        >>> data = {"label": "/path/to/seg_series_dir"}
        >>> result = transform(data)
        >>> print(result["label"].shape)
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not has_itkwasm:
            raise ImportError(
                "itkwasm-dicom is required for LoadDicomSegd. "
                "Install it with: pip install itkwasm-dicom"
            )

    def _find_dcm_file(self, path: Path) -> Path:
        """Find .dcm file in directory or return path if already a file."""
        if path.is_file():
            return path
        dcm_files = list(path.glob("*.dcm"))
        if not dcm_files:
            raise FileNotFoundError(f"No .dcm files found in {path}")
        return dcm_files[0]

    def _build_affine(self, img) -> np.ndarray:
        """Build 4x4 affine matrix from itkwasm Image metadata.

        Note: Assumes array has been transposed from (Z, Y, X) to (X, Y, Z),
        so spacing/origin/direction are also reordered accordingly.
        """
        # Reverse to match transposed (X, Y, Z) array order
        spacing = np.array(img.spacing)[::-1]
        origin = np.array(img.origin)[::-1]
        direction = np.array(img.direction).reshape(3, 3)
        # Reverse both rows and columns for transposed coordinates
        direction = direction[::-1, ::-1]

        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
        return affine

    def __call__(self, data: Mapping[Hashable, any]) -> dict[Hashable, any]:
        d = dict(data)
        for key in self.key_iterator(d):
            path = Path(d[key])
            dcm_file = self._find_dcm_file(path)

            # Read using ITKWasm
            seg_image, overlay_info = itkwasm_dicom.read_segmentation(dcm_file)

            # Convert itkwasm.Image to numpy array
            # ITKWasm returns (Z, Y, X) but ITKReader returns (X, Y, Z)
            # Transpose to match ITKReader/MONAI convention
            seg_array = np.asarray(seg_image.data)
            seg_array = np.transpose(seg_array, (2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)

            # Extract spatial metadata from itkwasm Image
            affine = self._build_affine(seg_image)

            # Create MONAI MetaTensor with metadata
            meta_tensor = MetaTensor(seg_array)
            meta_tensor.affine = affine
            meta_tensor.meta["filename_or_obj"] = str(dcm_file)
            meta_tensor.meta["overlay_info"] = overlay_info
            meta_tensor.meta["spacing"] = tuple(seg_image.spacing)
            meta_tensor.meta["origin"] = tuple(seg_image.origin)

            d[key] = meta_tensor
            d[f"{key}_meta_dict"] = dict(meta_tensor.meta)

        return d
