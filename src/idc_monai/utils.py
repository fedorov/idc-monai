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
Utility functions for IDC-MONAI integration.

Provides helper functions for querying IDC metadata, discovering datasets,
and preparing data for MONAI workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from monai.utils import optional_import

idc_index, has_idc = optional_import("idc_index")
pd, has_pandas = optional_import("pandas")


def get_client() -> idc_index.IDCClient:
    """
    Get an IDCClient instance.

    Returns:
        IDCClient instance for querying and downloading IDC data.

    Raises:
        ImportError: If idc-index is not installed.
    """
    if not has_idc:
        raise ImportError(
            "idc-index is required. Install it with: pip install idc-index"
        )
    return idc_index.IDCClient()


def query_collections(
    client: idc_index.IDCClient | None = None,
    modality: str | None = None,
    cancer_type: str | None = None,
    tumor_location: str | None = None,
    species: str = "Human",
) -> pd.DataFrame:
    """
    Query IDC collections using collections_index.

    Args:
        client: IDCClient instance. If None, creates a new one.
        modality: Filter by imaging modality (e.g., 'CT', 'MR', 'PT').
            Searches in the Modalities field.
        cancer_type: Filter by cancer type (e.g., 'Lung', 'Breast').
            Searches in the CancerTypes field.
        tumor_location: Filter by tumor location (e.g., 'Lung', 'Brain').
            Searches in the TumorLocations field.
        species: Filter by species (default: 'Human').

    Returns:
        DataFrame with collection information from collections_index:
        - collection_id, Subjects, CancerTypes, TumorLocations
        - Species, Modalities, SupportingData

    Example:
        >>> client = get_client()
        >>> # Find lung cancer CT collections
        >>> collections = query_collections(
        ...     client=client,
        ...     modality='CT',
        ...     cancer_type='Lung',
        ... )
        >>> print(collections[['collection_id', 'Subjects', 'CancerTypes']])
    """
    if client is None:
        client = get_client()

    # Fetch collections_index
    client.fetch_index("collections_index")

    # Build WHERE clauses
    where_clauses = []
    if species:
        where_clauses.append(f"Species LIKE '%{species}%'")
    if modality:
        where_clauses.append(f"Modalities LIKE '%{modality}%'")
    if cancer_type:
        where_clauses.append(f"CancerTypes LIKE '%{cancer_type}%'")
    if tumor_location:
        where_clauses.append(f"TumorLocations LIKE '%{tumor_location}%'")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
    SELECT
        collection_id,
        Subjects,
        CancerTypes,
        TumorLocations,
        Species,
        Modalities,
        SupportingData
    FROM collections_index
    WHERE {where_sql}
    ORDER BY Subjects DESC
    """

    return client.sql_query(query)


def query_series_with_segmentations(
    client: idc_index.IDCClient | None = None,
    collection_id: str | None = None,
    modality: str = "CT",
    body_part: str | None = None,
    algorithm_name: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Find image series that have corresponding DICOM segmentations.

    This is useful for supervised learning tasks where you need paired
    image and label data.

    Args:
        client: IDCClient instance. If None, creates a new one.
        collection_id: Filter by specific collection.
        modality: Source image modality (default: 'CT').
        body_part: Filter by body part examined.
        algorithm_name: Filter segmentations by algorithm (e.g., 'TotalSegmentator').
        limit: Maximum number of results to return.

    Returns:
        DataFrame with columns:
        - image_series_uid: SeriesInstanceUID of the source image
        - seg_series_uid: SeriesInstanceUID of the segmentation
        - collection_id, PatientID, BodyPartExamined
        - AlgorithmName, total_segments

    Example:
        >>> client = get_client()
        >>> # Find CT series with TotalSegmentator segmentations
        >>> paired_data = query_series_with_segmentations(
        ...     client=client,
        ...     modality='CT',
        ...     algorithm_name='TotalSegmentator',
        ...     limit=100,
        ... )
        >>> print(f"Found {len(paired_data)} image-segmentation pairs")
    """
    if client is None:
        client = get_client()

    # Fetch seg_index for segmentation metadata
    client.fetch_index("seg_index")

    # Build WHERE clauses
    where_clauses = [f"src.Modality = '{modality}'"]
    if collection_id:
        where_clauses.append(f"src.collection_id = '{collection_id}'")
    if body_part:
        where_clauses.append(f"src.BodyPartExamined LIKE '%{body_part}%'")
    if algorithm_name:
        where_clauses.append(f"seg.AlgorithmName LIKE '%{algorithm_name}%'")

    where_sql = " AND ".join(where_clauses)
    limit_sql = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT
        src.SeriesInstanceUID as image_series_uid,
        seg.SeriesInstanceUID as seg_series_uid,
        src.collection_id,
        src.PatientID,
        src.BodyPartExamined,
        seg.AlgorithmName,
        seg.total_segments
    FROM seg_index seg
    JOIN index src ON seg.segmented_SeriesInstanceUID = src.SeriesInstanceUID
    WHERE {where_sql}
    {limit_sql}
    """

    return client.sql_query(query)


def get_collection_info(
    collection_id: str,
    client: idc_index.IDCClient | None = None,
) -> dict:
    """
    Get detailed information about an IDC collection using collections_index.

    Args:
        collection_id: The collection identifier (e.g., 'nlst', 'tcga_luad').
        client: IDCClient instance. If None, creates a new one.

    Returns:
        Dictionary with collection metadata from collections_index:
        - collection_id, Subjects, CancerTypes, TumorLocations
        - Species, Modalities, SupportingData, DOI, URL
        Plus computed stats from primary index:
        - series_count, total_size_mb

    Example:
        >>> info = get_collection_info('nlst')
        >>> print(f"NLST: {info['Subjects']} subjects, Cancer: {info['CancerTypes']}")
    """
    if client is None:
        client = get_client()

    # Get metadata from collections_index
    client.fetch_index("collections_index")
    coll_query = f"""
    SELECT *
    FROM collections_index
    WHERE collection_id = '{collection_id}'
    """
    coll_info = client.sql_query(coll_query)

    if len(coll_info) == 0:
        raise ValueError(f"Collection '{collection_id}' not found in IDC")

    result = coll_info.iloc[0].to_dict()

    # Add computed stats from primary index
    stats_query = f"""
    SELECT
        COUNT(DISTINCT SeriesInstanceUID) as series_count,
        SUM(series_size_MB) as total_size_mb
    FROM index
    WHERE collection_id = '{collection_id}'
    """
    stats = client.sql_query(stats_query)
    if len(stats) > 0:
        result["series_count"] = int(stats.iloc[0]["series_count"])
        result["total_size_mb"] = float(stats.iloc[0]["total_size_mb"])
        result["total_size_gb"] = result["total_size_mb"] / 1024

    return result


def download_series(
    series_uids: Sequence[str],
    download_dir: str | Path,
    client: idc_index.IDCClient | None = None,
    dir_template: str = "%SeriesInstanceUID",
) -> list[Path]:
    """
    Download DICOM series from IDC.

    Args:
        series_uids: List of SeriesInstanceUIDs to download.
        download_dir: Directory to save downloaded files.
        client: IDCClient instance. If None, creates a new one.
        dir_template: Directory structure template. Default organizes by SeriesInstanceUID.

    Returns:
        List of paths to downloaded series directories.

    Example:
        >>> series_uids = ['1.2.3.4.5', '1.2.3.4.6']
        >>> paths = download_series(series_uids, './data')
        >>> print(f"Downloaded to: {paths}")
    """
    if client is None:
        client = get_client()

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    client.download_from_selection(
        seriesInstanceUID=list(series_uids),
        downloadDir=str(download_dir),
        dirTemplate=dir_template,
    )

    # Return paths to downloaded directories
    return [download_dir / uid for uid in series_uids]


def get_series_path(
    series_uid: str,
    download_dir: str | Path,
) -> Path:
    """
    Get the local path for a downloaded series.

    Args:
        series_uid: The SeriesInstanceUID.
        download_dir: The base download directory.

    Returns:
        Path to the series directory.
    """
    return Path(download_dir) / series_uid


def check_commercial_license(
    series_uids: Sequence[str],
    client: idc_index.IDCClient | None = None,
) -> tuple[list[str], list[str]]:
    """
    Check which series are available for commercial use.

    IDC data has different licenses. CC-BY allows commercial use,
    while CC-BY-NC restricts commercial applications.

    Args:
        series_uids: List of SeriesInstanceUIDs to check.
        client: IDCClient instance. If None, creates a new one.

    Returns:
        Tuple of (commercial_ok, commercial_restricted) series UIDs.

    Example:
        >>> commercial_ok, restricted = check_commercial_license(series_uids)
        >>> print(f"{len(commercial_ok)} series available for commercial use")
        >>> print(f"{len(restricted)} series restricted to non-commercial use")
    """
    if client is None:
        client = get_client()

    # Format UIDs for SQL IN clause
    uid_list = ", ".join(f"'{uid}'" for uid in series_uids)

    query = f"""
    SELECT SeriesInstanceUID, license_short_name
    FROM index
    WHERE SeriesInstanceUID IN ({uid_list})
    """

    results = client.sql_query(query)

    commercial_ok = []
    commercial_restricted = []

    for _, row in results.iterrows():
        uid = row["SeriesInstanceUID"]
        license_name = row["license_short_name"] or ""

        if "NC" in license_name.upper():
            commercial_restricted.append(uid)
        else:
            commercial_ok.append(uid)

    return commercial_ok, commercial_restricted


def query_analysis_results(
    client: idc_index.IDCClient | None = None,
    source_collection: str | None = None,
    modality: str | None = None,
) -> pd.DataFrame:
    """
    Query available analysis results (AI segmentations, annotations) from IDC.

    Args:
        client: IDCClient instance. If None, creates a new one.
        source_collection: Filter by source collection (e.g., 'tcga_luad').
        modality: Filter by modality in the analysis results.

    Returns:
        DataFrame with analysis results from analysis_results_index:
        - analysis_result_id, analysis_result_title
        - Subjects, Collections, Modalities, DOI

    Example:
        >>> client = get_client()
        >>> # Find all analysis results for TCGA-LUAD
        >>> results = query_analysis_results(
        ...     client=client,
        ...     source_collection='tcga_luad',
        ... )
        >>> print(results[['analysis_result_id', 'analysis_result_title']])
    """
    if client is None:
        client = get_client()

    # Fetch analysis_results_index
    client.fetch_index("analysis_results_index")

    # Build WHERE clauses
    where_clauses = []
    if source_collection:
        where_clauses.append(f"Collections LIKE '%{source_collection}%'")
    if modality:
        where_clauses.append(f"Modalities LIKE '%{modality}%'")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
    SELECT
        analysis_result_id,
        analysis_result_title,
        Subjects,
        Collections,
        Modalities,
        DOI
    FROM analysis_results_index
    WHERE {where_sql}
    ORDER BY Subjects DESC
    """

    return client.sql_query(query)
