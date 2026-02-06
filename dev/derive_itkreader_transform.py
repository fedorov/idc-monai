#!/usr/bin/env python
"""Derive the transformation that MONAI ITKReader applies, and replicate it for SEG."""

import os
import tempfile
import numpy as np

from idc_index import IDCClient
from itkwasm_dicom import read_segmentation
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data.image_reader import ITKReader
from monai.data import MetaTensor
from pathlib import Path

# Download test data
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

seg_dir = tempfile.mkdtemp(prefix="idc_derive_")
print(f"Downloading to {seg_dir}...")
client.download_from_selection(
    seriesInstanceUID=[demo_pair['image_uid'], demo_pair['seg_uid']],
    downloadDir=seg_dir,
    dirTemplate="%SeriesInstanceUID"
)

image_path = os.path.join(seg_dir, demo_pair['image_uid'])
seg_path = Path(os.path.join(seg_dir, demo_pair['seg_uid']))
dcm_files = list(seg_path.glob("*.dcm"))
seg_dcm = dcm_files[0]

# Load CT with MONAI ITKReader
ct_load = Compose([
    LoadImaged(keys=["image"], reader=ITKReader()),
    EnsureChannelFirstd(keys=["image"]),
])
ct_data = ct_load({"image": image_path})
ct_image = ct_data["image"]
ct_affine = ct_image.affine.numpy()

print(f"\nCT (MONAI ITKReader):")
print(f"  Shape: {ct_image.shape}")
print(f"  Affine:\n{ct_affine}")

# Load SEG with ITKWasm
seg_image, overlay_info = read_segmentation(seg_dcm)

itkwasm_array = np.asarray(seg_image.data).copy()
itkwasm_spacing = np.array(seg_image.spacing)
itkwasm_origin = np.array(seg_image.origin)
itkwasm_direction = np.array(seg_image.direction).reshape(3, 3)
itkwasm_size = np.array(seg_image.size)

print(f"\nSEG (ITKWasm raw):")
print(f"  Array shape (Z,Y,X): {itkwasm_array.shape}")
print(f"  Size (X,Y,Z): {itkwasm_size}")
print(f"  Spacing: {itkwasm_spacing}")
print(f"  Origin: {itkwasm_origin}")
print(f"  Direction:\n{itkwasm_direction}")

print("\n" + "="*60)
print("Step 2: Understanding ITKReader's transformation")
print("="*60)

# ITKReader does the following:
# 1. Reads image with SimpleITK/ITK which normalizes direction to identity
# 2. Applies a LPS-to-"ITK internal" coordinate flip (negate X and Y in origin and affine)
#
# The key is that MONAI's ITKReader produces affines with:
# - Negative X and Y in the diagonal (when original direction was positive)
# - Origin at the "opposite corner" in X and Y

# For the CT:
# - Original DICOM has direction [[1,0,0],[0,-1,0],[0,0,-1]] typically
# - ITK normalizes this and adjusts origin
# - MONAI then flips X and Y signs

# For the SEG with direction [[1,0,0],[0,-1,0],[0,0,-1]]:
# - X direction is +1, Y direction is -1, Z direction is -1
# - After "ITKReader-style" processing:
#   - X: direction becomes -1, origin moves to far X corner
#   - Y: direction is already negative, stays negative, origin unchanged
#   - Z: direction -1, need to handle this

# Let me analyze what CT array looks like vs SEG array
# to understand the array layout

print("\nCT affine breakdown:")
print(f"  Diagonal: {np.diag(ct_affine[:3,:3])}")  # [-0.7, -0.7, +5]
print(f"  Origin: {ct_affine[:3, 3]}")  # [179.6, 340.6, -328]

print("\nSEG raw metadata breakdown:")
print(f"  Direction diagonal: {np.diag(itkwasm_direction)}")  # [1, -1, -1]
print(f"  Origin: {itkwasm_origin}")  # [-179.6, 18.6, -83]

print("\n" + "="*60)
print("Step 3: Build ITKReader-compatible affine for SEG")
print("="*60)

# The pattern is:
# For each axis with positive direction: negate direction, move origin to far corner
# For each axis with negative direction: negate direction, keep origin same
#
# Actually, simpler: ITKReader always produces negative X and Y in the diagonal,
# and adjusts the origin to the far corner for those axes.

# Let's compute what the affine should be:
# Final diagonal should be: [-spacing[0], -spacing[1], +spacing[2]*sign(dir[2,2])]
# But we also need to consider the Z direction which is -1

# Looking at the CT pattern: [-0.7, -0.7, +5]
# The Z has original direction -1, but final diagonal is +5 (positive)
# This means: if original direction is -1, we flip the array and make diagonal positive

# So the algorithm for ITKReader-compatible transform:
# 1. For X: Always negate. Move origin to far corner.
# 2. For Y: Always negate. Move origin to far corner.
# 3. For Z: If direction is -1, flip array and use positive spacing. If +1, keep as is.

# Transpose array to (X, Y, Z)
seg_array = np.transpose(itkwasm_array, (2, 1, 0))
print(f"Array after transpose to (X,Y,Z): {seg_array.shape}")

# Apply flips where needed and build affine
final_array = seg_array.copy()
final_origin = itkwasm_origin.copy()
final_diagonal = np.zeros(3)

print("\nProcessing each axis:")

# X axis: direction is +1, ITKReader negates it
# Need to flip array and move origin
print(f"  X: direction={itkwasm_direction[0,0]}")
final_array = np.flip(final_array, axis=0)
final_origin[0] = itkwasm_origin[0] + (itkwasm_size[0] - 1) * itkwasm_spacing[0] * itkwasm_direction[0,0]
final_diagonal[0] = -itkwasm_spacing[0]
print(f"     Flipped array, new origin X = {final_origin[0]}")

# Y axis: direction is -1, ITKReader expects negative
# Array stays as is (direction already negative), origin stays
print(f"  Y: direction={itkwasm_direction[1,1]}")
# Actually wait - if direction is -1, ITKReader still negates.
# For Y with dir=-1, final diagonal = -spacing (which matches raw)
# But origin handling... let me check CT more carefully

# Let's look at the math more carefully
# CT had original direction [[1,0,0],[0,1,0],[0,0,1]] probably (axis-aligned)
# After ITKReader: origin [179.6, 340.6, -328], diagonal [-0.7, -0.7, +5]
# This suggests X and Y were flipped (dir becomes negative, origin moved)
# Z was kept positive

# For SEG with direction [[1,0,0],[0,-1,0],[0,0,-1]]:
# X: dir=+1 -> flip array, negate spacing, move origin to far corner
# Y: dir=-1 -> this is already "pointing negative", no flip needed? Let's see...
# Z: dir=-1 -> flip array, use positive spacing, move origin

# The CT Z has dir=-1 from DICOM but ITKReader shows +5 in diagonal
# So for negative Z direction, ITKReader flips Z and uses positive spacing

final_diagonal[1] = -itkwasm_spacing[1]
# Y origin: if dir=-1, we still want to end up with negative diagonal
# Origin should be at the "max Y" position
final_origin[1] = itkwasm_origin[1] + (itkwasm_size[1] - 1) * itkwasm_spacing[1] * abs(itkwasm_direction[1,1])
print(f"     Origin Y adjusted to {final_origin[1]}")

# Z axis: direction is -1, ITKReader would flip to positive
print(f"  Z: direction={itkwasm_direction[2,2]}")
final_array = np.flip(final_array, axis=2)
final_origin[2] = itkwasm_origin[2] + (itkwasm_size[2] - 1) * itkwasm_spacing[2] * itkwasm_direction[2,2]
final_diagonal[2] = itkwasm_spacing[2]  # Positive after flip
print(f"     Flipped array, new origin Z = {final_origin[2]}")

# Build final affine
seg_affine = np.eye(4)
seg_affine[:3, :3] = np.diag(final_diagonal)
seg_affine[:3, 3] = final_origin

print(f"\nFinal SEG array shape: {final_array.shape}")
print(f"Final SEG affine:\n{seg_affine}")
print(f"Final SEG diagonal: {final_diagonal}")
print(f"Final SEG origin: {final_origin}")

print(f"\nCT affine for comparison:\n{ct_affine}")
print(f"CT diagonal: {np.diag(ct_affine[:3,:3])}")
print(f"CT origin: {ct_affine[:3, 3]}")

print("\n" + "="*60)
print("Step 4: Test alignment")
print("="*60)

# Find a labeled voxel
labeled_indices = np.argwhere(final_array > 0)
if len(labeled_indices) > 0:
    mid_idx = len(labeled_indices) // 2
    seg_voxel = labeled_indices[mid_idx]
    print(f"\nTest SEG voxel: {seg_voxel}")

    # World coordinate from SEG
    seg_world = seg_affine @ np.append(seg_voxel, 1)
    print(f"SEG world: {seg_world[:3]}")

    # Convert to CT voxel
    ct_voxel = np.linalg.inv(ct_affine) @ seg_world
    print(f"CT voxel: {ct_voxel[:3]}")

    ct_shape = ct_image.shape[1:]
    ct_voxel_int = np.round(ct_voxel[:3]).astype(int)
    in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))
    print(f"CT voxel (rounded): {ct_voxel_int}, in bounds: {in_bounds}")

    if in_bounds:
        ct_value = ct_image[0, ct_voxel_int[0], ct_voxel_int[1], ct_voxel_int[2]].item()
        print(f"CT value at this location: {ct_value:.1f} HU")

print(f"\n\nData directory: {seg_dir}")
