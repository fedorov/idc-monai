#!/usr/bin/env python
"""Check raw DICOM metadata to understand orientation differences between CT and SEG.

This script reads the raw DICOM tags using pydicom to examine:
- ImagePositionPatient (origin)
- ImageOrientationPatient (direction cosines)
- PixelSpacing
- Per-frame metadata in DICOM-SEG

Key finding: CT and SEG often have different ImageOrientationPatient:
- CT: [1, 0, 0, 0, 1, 0]  -> Y direction = +1
- SEG: [1, 0, 0, 0, -1, 0] -> Y direction = -1

This means they cover the same physical space but with opposite Y array ordering.

Usage:
    cd idc_monai
    source .venv/bin/activate
    python dev/check_dicom_metadata.py
"""

import os
import tempfile
import numpy as np
import pydicom
from pathlib import Path

from idc_index import IDCClient
from itkwasm_dicom import read_segmentation

# Download test data
print("="*60)
print("Downloading data")
print("="*60)

client = IDCClient()
client.fetch_index("seg_index")

paired = client.sql_query("""
    SELECT src.SeriesInstanceUID as image_uid,
           seg.SeriesInstanceUID as seg_uid,
           src.collection_id
    FROM seg_index seg
    JOIN index src ON seg.segmented_SeriesInstanceUID = src.SeriesInstanceUID
    WHERE src.Modality = 'CT'
      AND seg.AlgorithmName LIKE '%TotalSegmentator%'
    ORDER BY src.series_size_MB ASC
    LIMIT 1
""")

demo_pair = paired.iloc[0]
print(f"Using: {demo_pair['collection_id']}")

seg_dir = tempfile.mkdtemp(prefix="idc_dicom_")
print(f"Downloading to {seg_dir}...")
client.download_from_selection(
    seriesInstanceUID=[demo_pair['image_uid'], demo_pair['seg_uid']],
    downloadDir=seg_dir,
    dirTemplate="%SeriesInstanceUID"
)

image_path = Path(os.path.join(seg_dir, demo_pair['image_uid']))
seg_path = Path(os.path.join(seg_dir, demo_pair['seg_uid']))
dcm_files = list(seg_path.glob("*.dcm"))
seg_dcm = dcm_files[0]

print("\n" + "="*60)
print("CT DICOM metadata")
print("="*60)

ct_files = sorted(image_path.glob("*.dcm"))
print(f"Number of CT DICOM files: {len(ct_files)}")

# Read first and last CT slice
ct_first = pydicom.dcmread(str(ct_files[0]))
ct_last = pydicom.dcmread(str(ct_files[-1]))

print(f"\nFirst CT slice:")
print(f"  ImagePositionPatient: {ct_first.ImagePositionPatient}")
print(f"  ImageOrientationPatient: {ct_first.ImageOrientationPatient}")
print(f"  PixelSpacing: {ct_first.PixelSpacing}")
print(f"  SliceThickness: {ct_first.SliceThickness if hasattr(ct_first, 'SliceThickness') else 'N/A'}")

print(f"\nLast CT slice:")
print(f"  ImagePositionPatient: {ct_last.ImagePositionPatient}")

# Calculate CT world extent
ct_origin_first = np.array([float(x) for x in ct_first.ImagePositionPatient])
ct_origin_last = np.array([float(x) for x in ct_last.ImagePositionPatient])
print(f"\nCT world Z range: {ct_origin_first[2]} to {ct_origin_last[2]}")

print("\n" + "="*60)
print("SEG DICOM metadata")
print("="*60)

seg_ds = pydicom.dcmread(str(seg_dcm))

print(f"SEG SOP Class: {seg_ds.SOPClassUID}")
print(f"Number of frames: {seg_ds.NumberOfFrames}")

# Get SharedFunctionalGroupsSequence for common metadata
if hasattr(seg_ds, 'SharedFunctionalGroupsSequence'):
    shared = seg_ds.SharedFunctionalGroupsSequence[0]
    if hasattr(shared, 'PixelMeasuresSequence'):
        pm = shared.PixelMeasuresSequence[0]
        print(f"\nShared PixelMeasures:")
        print(f"  PixelSpacing: {pm.PixelSpacing}")
        if hasattr(pm, 'SliceThickness'):
            print(f"  SliceThickness: {pm.SliceThickness}")
        if hasattr(pm, 'SpacingBetweenSlices'):
            print(f"  SpacingBetweenSlices: {pm.SpacingBetweenSlices}")

    if hasattr(shared, 'PlaneOrientationSequence'):
        po = shared.PlaneOrientationSequence[0]
        print(f"\nShared PlaneOrientation:")
        print(f"  ImageOrientationPatient: {po.ImageOrientationPatient}")

# Get PerFrameFunctionalGroupsSequence for per-frame position
if hasattr(seg_ds, 'PerFrameFunctionalGroupsSequence'):
    pffgs = seg_ds.PerFrameFunctionalGroupsSequence

    # Get first and last frame positions
    first_frame = pffgs[0]
    last_frame = pffgs[-1]

    if hasattr(first_frame, 'PlanePositionSequence'):
        first_pos = first_frame.PlanePositionSequence[0].ImagePositionPatient
        print(f"\nFirst SEG frame ImagePositionPatient: {first_pos}")

    if hasattr(last_frame, 'PlanePositionSequence'):
        last_pos = last_frame.PlanePositionSequence[0].ImagePositionPatient
        print(f"Last SEG frame ImagePositionPatient: {last_pos}")

    # Find unique positions
    positions = []
    for frame in pffgs:
        if hasattr(frame, 'PlanePositionSequence'):
            pos = frame.PlanePositionSequence[0].ImagePositionPatient
            positions.append([float(x) for x in pos])

    positions = np.array(positions)
    unique_z = np.unique(positions[:, 2])
    print(f"\nNumber of unique Z positions: {len(unique_z)}")
    print(f"SEG Z range: {unique_z.min()} to {unique_z.max()}")

    unique_y = np.unique(positions[:, 1])
    print(f"SEG Y values (unique): {unique_y}")

print("\n" + "="*60)
print("ITKWasm output for comparison")
print("="*60)

seg_image, overlay_info = read_segmentation(seg_dcm)
print(f"ITKWasm origin: {seg_image.origin}")
print(f"ITKWasm spacing: {seg_image.spacing}")
print(f"ITKWasm direction: {np.array(seg_image.direction).reshape(3,3)}")
print(f"ITKWasm size: {seg_image.size}")

# Calculate world extents from ITKWasm
origin = np.array(seg_image.origin)
spacing = np.array(seg_image.spacing)
direction = np.array(seg_image.direction).reshape(3, 3)
size = np.array(seg_image.size)

corner_000 = origin
corner_max = origin + direction @ (spacing * (size - 1))
print(f"\nITKWasm world corner [0,0,0]: {corner_000}")
print(f"ITKWasm world corner [max]: {corner_max}")

print(f"\n\nData directory: {seg_dir}")
