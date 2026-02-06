#!/usr/bin/env python
"""Compare CT (ITKReader) and SEG (ITKWasm) loading to understand the transformation difference."""

import os
import tempfile
import numpy as np

from idc_index import IDCClient
from itkwasm_dicom import read_segmentation
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data.image_reader import ITKReader
from monai.data import MetaTensor

# Initialize client and find a CT with segmentation
print("="*60)
print("Step 1: Download test data from IDC")
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

# Download
seg_dir = tempfile.mkdtemp(prefix="idc_ct_seg_")
print(f"Downloading to {seg_dir}...")
client.download_from_selection(
    seriesInstanceUID=[demo_pair['image_uid'], demo_pair['seg_uid']],
    downloadDir=seg_dir,
    dirTemplate="%SeriesInstanceUID"
)

from pathlib import Path
image_path = os.path.join(seg_dir, demo_pair['image_uid'])
seg_path = Path(os.path.join(seg_dir, demo_pair['seg_uid']))
dcm_files = list(seg_path.glob("*.dcm"))
seg_dcm = dcm_files[0]

print("\n" + "="*60)
print("Step 2: Load CT with ITKReader (MONAI default)")
print("="*60)

ct_load = Compose([
    LoadImaged(keys=["image"], reader=ITKReader()),
    EnsureChannelFirstd(keys=["image"]),
])
ct_data = ct_load({"image": image_path})
ct_image = ct_data["image"]
ct_affine = ct_image.affine.numpy()

print(f"CT shape: {ct_image.shape}")
print(f"CT affine:\n{ct_affine}")

# Extract components from CT affine
ct_origin = ct_affine[:3, 3]
ct_spacing_direction = ct_affine[:3, :3]
# Get diagonal (assuming axis-aligned for simplicity)
ct_spacing_diag = np.diag(ct_spacing_direction)
print(f"CT origin: {ct_origin}")
print(f"CT affine diagonal: {ct_spacing_diag}")

print("\n" + "="*60)
print("Step 3: Load SEG with ITKWasm (raw output)")
print("="*60)

seg_image, overlay_info = read_segmentation(seg_dcm)

# Raw ITKWasm output
itkwasm_array = np.asarray(seg_image.data).copy()
itkwasm_spacing = np.array(seg_image.spacing)
itkwasm_origin = np.array(seg_image.origin)
itkwasm_direction = np.array(seg_image.direction).reshape(3, 3)
itkwasm_size = np.array(seg_image.size)

print(f"ITKWasm array shape (Z,Y,X): {itkwasm_array.shape}")
print(f"ITKWasm size (X,Y,Z): {itkwasm_size}")
print(f"ITKWasm spacing: {itkwasm_spacing}")
print(f"ITKWasm origin: {itkwasm_origin}")
print(f"ITKWasm direction:\n{itkwasm_direction}")

# Build naive affine (just using metadata directly)
itkwasm_affine = np.eye(4)
itkwasm_affine[:3, :3] = itkwasm_direction @ np.diag(itkwasm_spacing)
itkwasm_affine[:3, 3] = itkwasm_origin
print(f"\nITKWasm affine (naive):\n{itkwasm_affine}")

print("\n" + "="*60)
print("Step 4: Compare affine components")
print("="*60)

print("\nAffine diagonals (spacing with direction signs):")
print(f"  CT:      {ct_spacing_diag}")
print(f"  ITKWasm: {np.diag(itkwasm_affine[:3, :3])}")

print("\nOrigins:")
print(f"  CT:      {ct_origin}")
print(f"  ITKWasm: {itkwasm_origin}")

print("\n" + "="*60)
print("Step 5: Analyze ITKReader transformation")
print("="*60)

# Let's use ITK directly to see what it does with the CT
import itk

# Load first CT DICOM file to get metadata
ct_files = sorted(Path(image_path).glob("*.dcm"))
print(f"Number of CT DICOM files: {len(ct_files)}")

# Read the CT series with ITK
PixelType = itk.ctype('signed short')
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageSeriesReader[ImageType].New()
dicom_names = itk.GDCMSeriesFileNames.New()
dicom_names.SetDirectory(str(image_path))
series_ids = list(dicom_names.GetSeriesUIDs())
file_names = list(dicom_names.GetFileNames(series_ids[0]))
reader.SetFileNames(file_names)
reader.Update()
ct_itk_image = reader.GetOutput()

itk_spacing = np.array(ct_itk_image.GetSpacing())
itk_origin = np.array(ct_itk_image.GetOrigin())
itk_direction = np.array(ct_itk_image.GetDirection())

print(f"\nITK (via itk library directly) CT metadata:")
print(f"  Spacing: {itk_spacing}")
print(f"  Origin: {itk_origin}")
print(f"  Direction:\n{itk_direction}")

# Build affine from ITK directly
itk_affine = np.eye(4)
itk_affine[:3, :3] = itk_direction @ np.diag(itk_spacing)
itk_affine[:3, 3] = itk_origin
print(f"\nITK affine (from itk library):\n{itk_affine}")

print("\n" + "="*60)
print("Step 6: Compare ITK (direct) vs MONAI ITKReader")
print("="*60)

print(f"\nITK affine diagonal: {np.diag(itk_affine[:3, :3])}")
print(f"MONAI CT affine diagonal: {ct_spacing_diag}")
print(f"\nITK origin: {itk_origin}")
print(f"MONAI CT origin: {ct_origin}")

# Check if MONAI applies any transformation
print(f"\nDifference in affines:")
print(f"ITK vs MONAI CT:\n{itk_affine - ct_affine}")

print("\n" + "="*60)
print("Step 7: Test world coordinate alignment")
print("="*60)

# Transpose SEG to (X, Y, Z) to match array layout convention
seg_array_xyz = np.transpose(itkwasm_array, (2, 1, 0))
print(f"SEG array shape after transpose (X,Y,Z): {seg_array_xyz.shape}")

# Find a labeled voxel
labeled_indices = np.argwhere(seg_array_xyz > 0)
if len(labeled_indices) > 0:
    mid_idx = len(labeled_indices) // 2
    seg_voxel = labeled_indices[mid_idx]
    print(f"\nTest SEG voxel (X,Y,Z): {seg_voxel}")

    # World coordinate from SEG (using ITKWasm metadata directly)
    seg_world = itkwasm_affine @ np.append(seg_voxel, 1)
    print(f"SEG world (ITKWasm affine): {seg_world[:3]}")

    # Convert to CT voxel using CT affine
    ct_voxel = np.linalg.inv(ct_affine) @ seg_world
    print(f"CT voxel (from MONAI CT affine): {ct_voxel[:3]}")

    # Also try with ITK affine
    ct_voxel_itk = np.linalg.inv(itk_affine) @ seg_world
    print(f"CT voxel (from ITK affine): {ct_voxel_itk[:3]}")

    # CT shape without channel
    ct_shape = ct_image.shape[1:]
    print(f"\nCT shape: {ct_shape}")

    ct_voxel_int = np.round(ct_voxel[:3]).astype(int)
    in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))
    print(f"CT voxel (rounded): {ct_voxel_int}, in bounds: {in_bounds}")

print(f"\n\nData directory: {seg_dir}")
