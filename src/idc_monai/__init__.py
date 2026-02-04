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
IDC-MONAI: Tools for using Imaging Data Commons with MONAI.
"""

from idc_monai.dataset import IDCDataset, IDCCacheDataset
from idc_monai.utils import (
    get_client,
    query_collections,
    query_series_with_segmentations,
    query_analysis_results,
    get_collection_info,
    download_series,
    get_series_path,
    check_commercial_license,
)
from idc_monai.transforms import (
    CTWindowd,
    CT_WINDOWS,
    get_ct_window_transform,
)

__version__ = "0.1.0"

__all__ = [
    # Dataset classes
    "IDCDataset",
    "IDCCacheDataset",
    # Query utilities
    "get_client",
    "query_collections",
    "query_series_with_segmentations",
    "query_analysis_results",
    "get_collection_info",
    # Download utilities
    "download_series",
    "get_series_path",
    "check_commercial_license",
    # Transforms
    "CTWindowd",
    "CT_WINDOWS",
    "get_ct_window_transform",
]
