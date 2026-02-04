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
IDC Dataset classes for MONAI integration.

Provides MONAI-compatible dataset classes that handle automatic download
and loading of DICOM data from the NCI Imaging Data Commons.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

from monai.data import Dataset, CacheDataset
from monai.config import PathLike
from monai.utils import optional_import

idc_index, has_idc = optional_import("idc_index")
pd, has_pandas = optional_import("pandas")

logger = logging.getLogger(__name__)


class IDCDataset(Dataset):
    """
    A MONAI-compatible dataset for loading data from the NCI Imaging Data Commons.

    This dataset handles automatic download of DICOM series from IDC and prepares
    them for use with MONAI transforms. It supports both image-only and
    image-with-segmentation workflows.

    Args:
        series_uids: List of SeriesInstanceUIDs to include in the dataset.
            Can also be a pandas DataFrame with 'SeriesInstanceUID' column.
        download_dir: Directory to download and cache DICOM files.
        transform: MONAI transforms to apply to each sample.
        seg_series_uids: Optional list of segmentation SeriesInstanceUIDs
            corresponding to each image series. Must be same length as series_uids.
        image_key: Key name for image data in the output dictionary.
        seg_key: Key name for segmentation data in the output dictionary.
        download: Whether to download data if not present locally.
        progress: Whether to show download progress.
        client: Optional pre-configured IDCClient instance. If not provided,
            a new client will be created.

    Example:
        >>> from idc_index import IDCClient
        >>> from idc_monai import IDCDataset
        >>> from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
        >>>
        >>> # Query for CT series
        >>> client = IDCClient()
        >>> series = client.sql_query('''
        ...     SELECT SeriesInstanceUID FROM index
        ...     WHERE collection_id = 'nlst' AND Modality = 'CT'
        ...     LIMIT 10
        ... ''')
        >>>
        >>> # Create dataset (reusing the same client)
        >>> transforms = Compose([
        ...     LoadImaged(keys=["image"]),
        ...     EnsureChannelFirstd(keys=["image"]),
        ... ])
        >>> dataset = IDCDataset(
        ...     series_uids=list(series['SeriesInstanceUID']),
        ...     download_dir="./data",
        ...     transform=transforms,
        ...     client=client,
        ... )
        >>> print(len(dataset))
        10
    """

    def __init__(
        self,
        series_uids: Sequence[str] | pd.DataFrame,
        download_dir: PathLike,
        transform: Callable | None = None,
        seg_series_uids: Sequence[str] | None = None,
        image_key: str = "image",
        seg_key: str = "label",
        download: bool = True,
        progress: bool = True,
        client: idc_index.IDCClient | None = None,
    ) -> None:
        if not has_idc:
            raise ImportError(
                "idc-index is required for IDCDataset. "
                "Install it with: pip install idc-index"
            )

        # Initialize or use provided IDCClient
        self.client = client if client is not None else idc_index.IDCClient()

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.image_key = image_key
        self.seg_key = seg_key
        self.progress = progress

        # Handle DataFrame input
        if has_pandas and isinstance(series_uids, pd.DataFrame):
            if "SeriesInstanceUID" not in series_uids.columns:
                raise ValueError(
                    "DataFrame must contain 'SeriesInstanceUID' column"
                )
            series_uids = list(series_uids["SeriesInstanceUID"])

        self.series_uids = list(series_uids)
        self.seg_series_uids = list(seg_series_uids) if seg_series_uids else None

        if self.seg_series_uids and len(self.seg_series_uids) != len(self.series_uids):
            raise ValueError(
                f"seg_series_uids length ({len(self.seg_series_uids)}) must match "
                f"series_uids length ({len(self.series_uids)})"
            )

        # Download data if requested
        if download:
            self._download_data()

        # Build data list for MONAI Dataset
        data = self._build_data_list()
        super().__init__(data=data, transform=transform)

    def _download_data(self) -> None:
        """Download DICOM series from IDC."""
        # Collect all UIDs to download
        all_uids = list(self.series_uids)
        if self.seg_series_uids:
            all_uids.extend(self.seg_series_uids)

        # Filter to only UIDs not already downloaded
        uids_to_download = []
        for uid in all_uids:
            series_path = self._get_series_path(uid)
            if not series_path.exists() or not any(series_path.iterdir()):
                uids_to_download.append(uid)

        if uids_to_download:
            logger.info(f"Downloading {len(uids_to_download)} series from IDC...")
            self.client.download_from_selection(
                seriesInstanceUID=uids_to_download,
                downloadDir=str(self.download_dir),
                dirTemplate="%SeriesInstanceUID",
            )
            logger.info("Download complete.")
        else:
            logger.info("All series already downloaded.")

    def _get_series_path(self, series_uid: str) -> Path:
        """Get local path for a series."""
        return self.download_dir / series_uid

    def _build_data_list(self) -> list[dict]:
        """Build list of data dictionaries for MONAI Dataset."""
        data = []
        for i, series_uid in enumerate(self.series_uids):
            item = {
                self.image_key: str(self._get_series_path(series_uid)),
                "series_uid": series_uid,
            }
            if self.seg_series_uids:
                seg_uid = self.seg_series_uids[i]
                item[self.seg_key] = str(self._get_series_path(seg_uid))
                item["seg_series_uid"] = seg_uid
            data.append(item)
        return data


class IDCCacheDataset(CacheDataset):
    """
    A cached version of IDCDataset for faster training.

    This dataset caches transformed data in memory after the first epoch,
    significantly speeding up subsequent epochs. Use this when your transforms
    include expensive operations like resampling.

    Args:
        series_uids: List of SeriesInstanceUIDs to include in the dataset.
        download_dir: Directory to download and cache DICOM files.
        transform: MONAI transforms to apply to each sample.
        seg_series_uids: Optional list of segmentation SeriesInstanceUIDs.
        image_key: Key name for image data in the output dictionary.
        seg_key: Key name for segmentation data in the output dictionary.
        download: Whether to download data if not present locally.
        cache_rate: Fraction of data to cache (0.0 to 1.0).
        num_workers: Number of workers for caching.
        progress: Whether to show progress bars.
        client: Optional pre-configured IDCClient instance.

    Example:
        >>> from idc_monai import IDCCacheDataset
        >>> from monai.transforms import Compose, LoadImaged, Spacingd
        >>>
        >>> transforms = Compose([
        ...     LoadImaged(keys=["image"]),
        ...     Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0)),
        ... ])
        >>> dataset = IDCCacheDataset(
        ...     series_uids=series_list,
        ...     download_dir="./data",
        ...     transform=transforms,
        ...     cache_rate=1.0,  # Cache all data
        ... )
    """

    def __init__(
        self,
        series_uids: Sequence[str] | pd.DataFrame,
        download_dir: PathLike,
        transform: Callable | None = None,
        seg_series_uids: Sequence[str] | None = None,
        image_key: str = "image",
        seg_key: str = "label",
        download: bool = True,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        progress: bool = True,
        client: idc_index.IDCClient | None = None,
    ) -> None:
        if not has_idc:
            raise ImportError(
                "idc-index is required for IDCCacheDataset. "
                "Install it with: pip install idc-index"
            )

        # Initialize or use provided IDCClient
        self.client = client if client is not None else idc_index.IDCClient()

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.image_key = image_key
        self.seg_key = seg_key

        # Handle DataFrame input
        if has_pandas and isinstance(series_uids, pd.DataFrame):
            if "SeriesInstanceUID" not in series_uids.columns:
                raise ValueError(
                    "DataFrame must contain 'SeriesInstanceUID' column"
                )
            series_uids = list(series_uids["SeriesInstanceUID"])

        self.series_uids = list(series_uids)
        self.seg_series_uids = list(seg_series_uids) if seg_series_uids else None

        if self.seg_series_uids and len(self.seg_series_uids) != len(self.series_uids):
            raise ValueError(
                f"seg_series_uids length ({len(self.seg_series_uids)}) must match "
                f"series_uids length ({len(self.series_uids)})"
            )

        # Download data if requested
        if download:
            self._download_data()

        # Build data list
        data = self._build_data_list()

        super().__init__(
            data=data,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
        )

    def _download_data(self) -> None:
        """Download DICOM series from IDC."""
        all_uids = list(self.series_uids)
        if self.seg_series_uids:
            all_uids.extend(self.seg_series_uids)

        uids_to_download = []
        for uid in all_uids:
            series_path = self._get_series_path(uid)
            if not series_path.exists() or not any(series_path.iterdir()):
                uids_to_download.append(uid)

        if uids_to_download:
            logger.info(f"Downloading {len(uids_to_download)} series from IDC...")
            self.client.download_from_selection(
                seriesInstanceUID=uids_to_download,
                downloadDir=str(self.download_dir),
                dirTemplate="%SeriesInstanceUID",
            )
            logger.info("Download complete.")

    def _get_series_path(self, series_uid: str) -> Path:
        """Get local path for a series."""
        return self.download_dir / series_uid

    def _build_data_list(self) -> list[dict]:
        """Build list of data dictionaries."""
        data = []
        for i, series_uid in enumerate(self.series_uids):
            item = {
                self.image_key: str(self._get_series_path(series_uid)),
                "series_uid": series_uid,
            }
            if self.seg_series_uids:
                seg_uid = self.seg_series_uids[i]
                item[self.seg_key] = str(self._get_series_path(seg_uid))
                item["seg_series_uid"] = seg_uid
            data.append(item)
        return data
