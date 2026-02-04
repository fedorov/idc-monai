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

Note: MONAI's LoadImaged with ITKReader handles DICOM series directly -
no format conversion is needed. These transforms provide additional
preprocessing utilities specific to IDC data workflows.
"""

from __future__ import annotations

from typing import Hashable, Mapping

from monai.config import KeysCollection
from monai.transforms import MapTransform


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
