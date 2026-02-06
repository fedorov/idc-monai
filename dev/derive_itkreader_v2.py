#!/usr/bin/env python
"""Derive the transformation that MONAI ITKReader applies - v2 with corrected Y handling."""

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
print("Step 1: Download and load data")
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

seg_dir = tempfile.mkdtemp(prefix="idc_derive2_")
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

# Load SEG with ITKWasm
seg_image, overlay_info = read_segmentation(seg_dcm)

itkwasm_array = np.asarray(seg_image.data).copy()
itkwasm_spacing = np.array(seg_image.spacing)
itkwasm_origin = np.array(seg_image.origin)
itkwasm_direction = np.array(seg_image.direction).reshape(3, 3)
itkwasm_size = np.array(seg_image.size)

print(f"\nCT affine:\n{ct_affine}")
print(f"\nSEG direction:\n{itkwasm_direction}")
print(f"SEG origin: {itkwasm_origin}")
print(f"SEG size: {itkwasm_size}")

print("\n" + "="*60)
print("Step 2: Analyze the pattern")
print("="*60)

# The key insight: ITKReader applies what's called LPS-to-RAS in MONAI
# But more specifically, it flips the array to have positive direction
# and adjusts the affine accordingly.

# For the CT with assumed direction [[1,0,0],[0,1,0],[0,0,1]]:
# - Final diagonal is [-0.7, -0.7, +5]
# - This means X and Y are "reversed" (negative spacing), Z is normal
# - Origin is moved to match the flipped array

# For SEG with direction [[1,0,0],[0,-1,0],[0,0,-1]]:
# - X: dir=+1, needs to match CT's -1 -> flip X, negate spacing
# - Y: dir=-1, CT also has -0.7 -> should match without flip?
#      But wait, CT direction is +1 and gets negated to -0.7
#      SEG direction is -1 and should also become -0.7 (keep negative)
# - Z: dir=-1, CT has +5 -> SEG should also have +5 -> flip Z, use positive spacing

# The question is: when does the array get flipped?
# ITKReader flips the array when the FINAL affine would have positive spacing
# to ensure consistent behavior.

# Actually, let me think about this differently.
# The goal of ITKReader is:
# - Always have negative X and Y in the affine diagonal
# - Always have positive Z in the affine diagonal
# - Adjust array and origin to match

# For SEG:
# X: dir=+1, spacing=0.7 -> product = +0.7
#    Target: -0.7 -> flip array X, origin moves to far X
# Y: dir=-1, spacing=0.7 -> product = -0.7
#    Target: -0.7 -> already matches! No flip needed, origin stays
# Z: dir=-1, spacing=5 -> product = -5
#    Target: +5 -> flip array Z, origin moves to far Z

# So for Y, there should be NO flip and NO origin adjustment

print("\nProcessing:")
print(f"X: dir={itkwasm_direction[0,0]}, product={itkwasm_direction[0,0]*itkwasm_spacing[0]}")
print(f"   Target: -0.703125 -> {'flip' if itkwasm_direction[0,0] > 0 else 'no flip'}")

print(f"Y: dir={itkwasm_direction[1,1]}, product={itkwasm_direction[1,1]*itkwasm_spacing[1]}")
print(f"   Target: -0.703125 -> {'flip' if itkwasm_direction[1,1] > 0 else 'no flip'}")

print(f"Z: dir={itkwasm_direction[2,2]}, product={itkwasm_direction[2,2]*itkwasm_spacing[2]}")
print(f"   Target: +5 -> {'flip' if itkwasm_direction[2,2] < 0 else 'no flip'}")

print("\n" + "="*60)
print("Step 3: Build correct affine and flip array")
print("="*60)

# Transpose array to (X, Y, Z)
seg_array = np.transpose(itkwasm_array, (2, 1, 0))
print(f"Array after transpose to (X,Y,Z): {seg_array.shape}")

final_array = seg_array.copy()
final_origin = itkwasm_origin.copy()
final_diagonal = np.zeros(3)

# X axis: direction=+1, target=-1
# Flip array and move origin to far corner
final_array = np.flip(final_array, axis=0)
final_origin[0] = itkwasm_origin[0] + (itkwasm_size[0] - 1) * itkwasm_spacing[0] * itkwasm_direction[0,0]
final_diagonal[0] = -itkwasm_spacing[0]
print(f"X: flipped, new origin = {final_origin[0]}")

# Y axis: direction=-1, target=-1
# NO flip needed, origin STAYS THE SAME (already at correct corner)
final_diagonal[1] = -itkwasm_spacing[1]
# Origin Y stays as is!
print(f"Y: NOT flipped, origin stays = {final_origin[1]}")

# Z axis: direction=-1, target=+1
# Flip array and move origin to far corner
final_array = np.flip(final_array, axis=2)
final_origin[2] = itkwasm_origin[2] + (itkwasm_size[2] - 1) * itkwasm_spacing[2] * itkwasm_direction[2,2]
final_diagonal[2] = itkwasm_spacing[2]
print(f"Z: flipped, new origin = {final_origin[2]}")

# Build final affine
seg_affine = np.eye(4)
seg_affine[:3, :3] = np.diag(final_diagonal)
seg_affine[:3, 3] = final_origin

print(f"\nFinal SEG affine:\n{seg_affine}")
print(f"\nCT affine:\n{ct_affine}")

print("\n" + "="*60)
print("Step 4: Compare origins")
print("="*60)

print(f"SEG origin: {final_origin}")
print(f"CT origin:  {ct_affine[:3, 3]}")
print(f"Difference: {final_origin - ct_affine[:3, 3]}")

# The origins are different - this could be because SEG and CT
# don't have the exact same extent. Let me check the world extents.

print("\n" + "="*60)
print("Step 5: Check world coordinate extents")
print("="*60)

# CT extent
ct_shape = ct_image.shape[1:]  # (X, Y, Z) = (512, 512, 50)
ct_corner_0 = ct_affine @ np.array([0, 0, 0, 1])
ct_corner_max = ct_affine @ np.array([ct_shape[0]-1, ct_shape[1]-1, ct_shape[2]-1, 1])
print(f"CT voxel [0,0,0] -> world: {ct_corner_0[:3]}")
print(f"CT voxel [{ct_shape[0]-1},{ct_shape[1]-1},{ct_shape[2]-1}] -> world: {ct_corner_max[:3]}")

# SEG extent
seg_shape = final_array.shape
seg_corner_0 = seg_affine @ np.array([0, 0, 0, 1])
seg_corner_max = seg_affine @ np.array([seg_shape[0]-1, seg_shape[1]-1, seg_shape[2]-1, 1])
print(f"\nSEG voxel [0,0,0] -> world: {seg_corner_0[:3]}")
print(f"SEG voxel [{seg_shape[0]-1},{seg_shape[1]-1},{seg_shape[2]-1}] -> world: {seg_corner_max[:3]}")

print("\n" + "="*60)
print("Step 6: Test alignment")
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

    ct_voxel_int = np.round(ct_voxel[:3]).astype(int)
    in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))
    print(f"CT voxel (rounded): {ct_voxel_int}, in bounds: {in_bounds}")

    if in_bounds:
        ct_value = ct_image[0, ct_voxel_int[0], ct_voxel_int[1], ct_voxel_int[2]].item()
        print(f"CT value at this location: {ct_value:.1f} HU")

    # Let me also check a corner voxel to verify alignment
    print("\n--- Corner test ---")
    seg_voxel_0 = np.array([0, 0, 0])
    seg_world_0 = seg_affine @ np.append(seg_voxel_0, 1)
    ct_voxel_0 = np.linalg.inv(ct_affine) @ seg_world_0
    print(f"SEG voxel [0,0,0] -> world {seg_world_0[:3]} -> CT voxel {ct_voxel_0[:3]}")

print(f"\n\nData directory: {seg_dir}")
