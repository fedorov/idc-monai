#!/usr/bin/env python
"""Compare dcmqi CLI output with ITKWasm to verify they produce identical results.

This script was used to verify that ITKWasm (which wraps dcmqi via WebAssembly)
produces the same output as the dcmqi CLI tool.

Key finding: ITKWasm and dcmqi produce IDENTICAL output:
- Same array data
- Same origin, spacing, direction
- Same affine matrix

This confirms that any differences between DICOM-SEG and CT loading are due
to MONAI's ITKReader applying additional transformations, not ITKWasm issues.

Requirements:
    - dcmqi CLI binaries (download from https://github.com/QIICR/dcmqi/releases)
    - Set path to dcmqi in the script

Usage:
    cd idc_monai
    source .venv/bin/activate
    python dev/compare_dcmqi_itkwasm.py
"""

import os
import subprocess
import tempfile
import numpy as np

from idc_index import IDCClient
from itkwasm_dicom import read_segmentation

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
seg_dir = tempfile.mkdtemp(prefix="idc_compare_")
print(f"Downloading to {seg_dir}...")
client.download_from_selection(
    seriesInstanceUID=[demo_pair['seg_uid']],
    downloadDir=seg_dir,
    dirTemplate="%SeriesInstanceUID"
)

# Find the SEG file
from pathlib import Path
seg_path = Path(os.path.join(seg_dir, demo_pair['seg_uid']))
dcm_files = list(seg_path.glob("*.dcm"))
seg_dcm = dcm_files[0]
print(f"SEG file: {seg_dcm}")

print("\n" + "="*60)
print("Step 2: Read with ITKWasm")
print("="*60)

seg_image, overlay_info = read_segmentation(seg_dcm)

itkwasm_array = np.asarray(seg_image.data).copy()
itkwasm_spacing = np.array(seg_image.spacing)
itkwasm_origin = np.array(seg_image.origin)
itkwasm_direction = np.array(seg_image.direction).reshape(3, 3)
itkwasm_size = np.array(seg_image.size)

print(f"ITKWasm array shape (raw): {itkwasm_array.shape}")
print(f"ITKWasm size (X,Y,Z): {itkwasm_size}")
print(f"ITKWasm spacing: {itkwasm_spacing}")
print(f"ITKWasm origin: {itkwasm_origin}")
print(f"ITKWasm direction:\n{itkwasm_direction}")

print("\n" + "="*60)
print("Step 3: Convert with dcmqi to NRRD")
print("="*60)

dcmqi_out = os.path.join(seg_dir, "dcmqi_nrrd")
os.makedirs(dcmqi_out, exist_ok=True)

dcmqi_cmd = [
    "/Users/af61/Downloads/dcmqi-1.5.0-mac/bin/segimage2itkimage",
    "--inputDICOM", str(seg_dcm),
    "--outputDirectory", dcmqi_out,
    "-t", "nrrd",
    "--mergeSegments",
    "--verbose"
]
print(f"Running: {' '.join(dcmqi_cmd)}")
result = subprocess.run(dcmqi_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error: {result.stderr}")
else:
    print(result.stdout[:500] if result.stdout else "Success")

# Find the nrrd file
nrrd_files = list(Path(dcmqi_out).glob("*.nrrd"))
if nrrd_files:
    nrrd_file = nrrd_files[0]
    print(f"NRRD file: {nrrd_file}")
else:
    print("No NRRD file found!")
    exit(1)

print("\n" + "="*60)
print("Step 4: Read NRRD with ITK")
print("="*60)

import itk

# Read the NRRD file
itk_image = itk.imread(str(nrrd_file))

dcmqi_array = itk.GetArrayFromImage(itk_image)
dcmqi_spacing = np.array(itk_image.GetSpacing())
dcmqi_origin = np.array(itk_image.GetOrigin())
dcmqi_direction = np.array(itk_image.GetDirection())

print(f"dcmqi array shape (raw): {dcmqi_array.shape}")
print(f"dcmqi spacing: {dcmqi_spacing}")
print(f"dcmqi origin: {dcmqi_origin}")
print(f"dcmqi direction:\n{dcmqi_direction}")

print("\n" + "="*60)
print("Step 5: Compare metadata")
print("="*60)

print("\nSpacing comparison:")
print(f"  ITKWasm: {itkwasm_spacing}")
print(f"  dcmqi:   {dcmqi_spacing}")
print(f"  Match:   {np.allclose(itkwasm_spacing, dcmqi_spacing)}")

print("\nOrigin comparison:")
print(f"  ITKWasm: {itkwasm_origin}")
print(f"  dcmqi:   {dcmqi_origin}")
print(f"  Ratio:   {dcmqi_origin / itkwasm_origin}")

print("\nDirection comparison:")
print(f"  ITKWasm:\n{itkwasm_direction}")
print(f"  dcmqi:\n{dcmqi_direction}")

# Check if there's a simple transformation
lps_to_ras = np.diag([-1, -1, 1])
print(f"\nIf we apply LPS-to-RAS (negate X and Y) to ITKWasm origin:")
transformed_origin = lps_to_ras @ itkwasm_origin
print(f"  Transformed: {transformed_origin}")
print(f"  dcmqi:       {dcmqi_origin}")
print(f"  Match:       {np.allclose(transformed_origin, dcmqi_origin)}")

print(f"\nIf we apply LPS-to-RAS to ITKWasm direction:")
transformed_direction = lps_to_ras @ itkwasm_direction
print(f"  Transformed:\n{transformed_direction}")
print(f"  dcmqi:\n{dcmqi_direction}")
print(f"  Match: {np.allclose(transformed_direction, dcmqi_direction)}")

print("\n" + "="*60)
print("Step 6: Compare array data")
print("="*60)

# ITKWasm returns array in (Z, Y, X), ITK also returns in (Z, Y, X)
print(f"ITKWasm array shape: {itkwasm_array.shape}")
print(f"dcmqi array shape:   {dcmqi_array.shape}")

# Check if arrays are equal
if itkwasm_array.shape == dcmqi_array.shape:
    arrays_equal = np.array_equal(itkwasm_array, dcmqi_array)
    print(f"Arrays equal: {arrays_equal}")

    if not arrays_equal:
        # Check various flip combinations
        for flip_axes in [(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]:
            flipped = itkwasm_array
            for ax in flip_axes:
                flipped = np.flip(flipped, axis=ax)
            if np.array_equal(flipped, dcmqi_array):
                print(f"Arrays equal after flipping axes {flip_axes}")
                break
        else:
            # Check non-zero values
            itkwasm_nonzero = np.sum(itkwasm_array > 0)
            dcmqi_nonzero = np.sum(dcmqi_array > 0)
            print(f"ITKWasm non-zero voxels: {itkwasm_nonzero}")
            print(f"dcmqi non-zero voxels: {dcmqi_nonzero}")

print("\n" + "="*60)
print("Step 7: Voxel-to-world coordinate comparison")
print("="*60)

# Find a labeled voxel in dcmqi array
labeled_indices = np.argwhere(dcmqi_array > 0)
if len(labeled_indices) > 0:
    mid_idx = len(labeled_indices) // 2
    test_voxel_zyx = labeled_indices[mid_idx]  # (z, y, x) order
    test_voxel_xyz = test_voxel_zyx[::-1]  # (x, y, z) order

    print(f"Test voxel (z,y,x): {test_voxel_zyx}")
    print(f"Test voxel (x,y,z): {test_voxel_xyz}")

    # Build affine from ITKWasm metadata
    itkwasm_affine = np.eye(4)
    itkwasm_affine[:3, :3] = itkwasm_direction @ np.diag(itkwasm_spacing)
    itkwasm_affine[:3, 3] = itkwasm_origin

    # Build affine from dcmqi metadata
    dcmqi_affine = np.eye(4)
    dcmqi_affine[:3, :3] = dcmqi_direction @ np.diag(dcmqi_spacing)
    dcmqi_affine[:3, 3] = dcmqi_origin

    print(f"\nITKWasm affine:\n{itkwasm_affine}")
    print(f"\ndcmqi affine:\n{dcmqi_affine}")

    # Compute world coordinates
    voxel_h = np.append(test_voxel_xyz, 1)
    itkwasm_world = itkwasm_affine @ voxel_h
    dcmqi_world = dcmqi_affine @ voxel_h

    print(f"\nWorld coords (ITKWasm): {itkwasm_world[:3]}")
    print(f"World coords (dcmqi):   {dcmqi_world[:3]}")

print(f"\n\nData directory: {seg_dir}")
