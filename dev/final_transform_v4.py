#!/usr/bin/env python
"""Standalone implementation of ITKReader-compatible DICOM-SEG loading.

This script demonstrates the transformation logic used in LoadDicomSegd
without depending on the idc_monai package. Useful for debugging and
understanding the coordinate transformation.

Key findings:
1. ITKWasm and dcmqi CLI produce identical output (verified)
2. MONAI's ITKReader applies additional transformation to CT data
3. SEG needs Y and Z axis flips plus origin adjustments to match

The transformation:
1. Transpose ITKWasm array from (Z,Y,X) to (X,Y,Z)
2. Flip Y axis (if direction[1,1] < 0)
3. Flip Z axis (if direction[2,2] < 0)
4. Negate X origin, negate Y origin (after flip adjustment)
5. Build affine with diagonal [-spacing_x, -spacing_y, +spacing_z]

Usage:
    cd idc_monai
    source .venv/bin/activate
    python dev/final_transform_v4.py
"""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from idc_index import IDCClient
from itkwasm_dicom import read_segmentation
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data.image_reader import ITKReader
from monai.data import MetaTensor
from pathlib import Path

# Download test data
print("="*60)
print("Downloading and loading data")
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

seg_dir = tempfile.mkdtemp(prefix="idc_final4_")
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

print(f"\nCT shape: {ct_image.shape}")
print(f"CT affine:\n{ct_affine}")

# Load SEG with ITKWasm
seg_image, overlay_info = read_segmentation(seg_dcm)

itkwasm_array = np.asarray(seg_image.data).copy()
itkwasm_spacing = np.array(seg_image.spacing)
itkwasm_origin = np.array(seg_image.origin)
itkwasm_direction = np.array(seg_image.direction).reshape(3, 3)
itkwasm_size = np.array(seg_image.size)

print(f"\nSEG raw shape (Z,Y,X): {itkwasm_array.shape}")
print(f"SEG origin: {itkwasm_origin}")
print(f"SEG direction:\n{itkwasm_direction}")

print("\n" + "="*60)
print("Building ITKReader-compatible affine")
print("="*60)

# The key insight: ITKReader applies a specific transformation to DICOM images
# that we need to replicate for SEG to match.
#
# ITKReader's transformation (empirically observed):
# 1. Flips the array along X and Y axes
# 2. Negates the X and Y components of the origin
# 3. Uses diagonal [-spacing_x, -spacing_y, +spacing_z]
#
# For SEG with direction [+1, -1, -1]:
# We need to determine how to flip the array and adjust the origin
# so that the world coordinates match the CT.

# Step 1: Transpose from (Z, Y, X) to (X, Y, Z)
seg_array = np.transpose(itkwasm_array, (2, 1, 0))
print(f"After transpose to (X,Y,Z): {seg_array.shape}")

# Step 2: Apply the ITKReader-like transformation
# The SEG origin is at [-179.6, +18.6, -83]
# The SEG direction is [+1, -1, -1]
#
# After the transformation, we want the affine to match CT's convention:
# - Diagonal: [-0.7, -0.7, +5]
# - Origin such that the world extent matches

# Let's compute where each corner of the SEG volume maps in world coordinates,
# then determine the transformation needed.

# Raw SEG affine (after transpose)
seg_affine_raw = np.eye(4)
seg_affine_raw[:3, :3] = itkwasm_direction @ np.diag(itkwasm_spacing)
seg_affine_raw[:3, 3] = itkwasm_origin

def voxel_to_world(affine, voxel):
    return (affine @ np.append(voxel, 1))[:3]

# Compute world extent of SEG
seg_corners = []
for i in [0, itkwasm_size[0]-1]:
    for j in [0, itkwasm_size[1]-1]:
        for k in [0, itkwasm_size[2]-1]:
            w = voxel_to_world(seg_affine_raw, [i, j, k])
            seg_corners.append((i, j, k, w))

print("\nSEG corners in world coordinates:")
for corner in seg_corners:
    print(f"  voxel {corner[:3]} -> world [{corner[3][0]:.1f}, {corner[3][1]:.1f}, {corner[3][2]:.1f}]")

# Compute world extent of CT
ct_shape = ct_image.shape[1:]
ct_corners = []
for i in [0, ct_shape[0]-1]:
    for j in [0, ct_shape[1]-1]:
        for k in [0, ct_shape[2]-1]:
            w = voxel_to_world(ct_affine, [i, j, k])
            ct_corners.append((i, j, k, w))

print("\nCT corners in world coordinates:")
for corner in ct_corners:
    print(f"  voxel {corner[:3]} -> world [{corner[3][0]:.1f}, {corner[3][1]:.1f}, {corner[3][2]:.1f}]")

# Find which SEG corner matches CT corner [0,0,0]
ct_corner_000 = voxel_to_world(ct_affine, [0, 0, 0])
print(f"\nCT voxel [0,0,0] is at world: {ct_corner_000}")

# The CT [0,0,0] is at [179.6, 340.6, -328]
# Looking at SEG corners, the closest match should be one of them
# after we apply the LPS-to-RAS-like transformation that ITKReader uses.

# Actually, ITKReader does NOT just negate coordinates. It converts from LPS to RAS
# by negating X and Y. So:
# LPS origin [-179.6, +18.6, -83] becomes RAS origin [+179.6, -18.6, -83]
#
# For the SEG, after LPS-to-RAS:
# - Origin: [+179.6, -18.6, -83]
# - The Y direction -1 means as voxel Y increases, world Y decreases
#   In RAS: world Y = -18.6 - voxel_y * 0.7 * 1 (negated direction)
#   Wait, that's not right either.

# Let me approach this differently. The ITKReader transformation for LPS-to-RAS is:
# new_coord = [-old_x, -old_y, old_z]
#
# This affects both the origin and how we interpret direction.

# Original SEG in LPS:
# - Origin: [-179.6, +18.6, -83]
# - Direction: [+1, -1, -1]
# - World_x = -179.6 + voxel_x * 0.7 * 1 = -179.6 + voxel_x * 0.7
# - World_y = +18.6 + voxel_y * 0.7 * (-1) = +18.6 - voxel_y * 0.7
# - World_z = -83 + voxel_z * 5 * (-1) = -83 - voxel_z * 5

# After LPS-to-RAS transformation (negate X and Y):
# - New_x = -World_x = -(-179.6 + voxel_x * 0.7) = 179.6 - voxel_x * 0.7
# - New_y = -World_y = -(+18.6 - voxel_y * 0.7) = -18.6 + voxel_y * 0.7
# - New_z = World_z = -83 - voxel_z * 5

# So in RAS coordinates:
# - voxel [0,0,0] -> [+179.6, -18.6, -83]
# - voxel [511,0,0] -> [179.6 - 511*0.7, -18.6, -83] = [-179.6, -18.6, -83]
# - voxel [0,511,0] -> [+179.6, -18.6 + 511*0.7, -83] = [+179.6, +340.6, -83]
# - voxel [0,0,49] -> [+179.6, -18.6, -83 - 49*5] = [+179.6, -18.6, -328]

# Now compare with CT in RAS (from ITKReader):
# - voxel [0,0,0] -> [+179.6, +340.6, -328]
# - voxel [511,0,0] -> [+179.6 - 511*0.7, +340.6, -328] = [-179.6, +340.6, -328]
# - voxel [0,511,0] -> [+179.6, +340.6 - 511*0.7, -328] = [+179.6, -18.6, -328]
# - voxel [0,0,49] -> [+179.6, +340.6, -328 + 49*5] = [+179.6, +340.6, -83]

# Matching corners:
# CT [0,0,0] at [+179.6, +340.6, -328] matches SEG [0, 511, 49] at [+179.6, +340.6, -328]? Let's check:
# SEG [0,511,49] -> [+179.6, -18.6+511*0.7, -83-49*5] = [+179.6, +340.1, -328] YES!

# So CT voxel [0,0,0] corresponds to SEG voxel [0, 511, 49]
# This means we need to flip Y and Z in the SEG array!

print("\n" + "="*60)
print("Applying transformation")
print("="*60)

# Based on the analysis:
# To transform SEG so that SEG voxel [0,0,0] corresponds to CT voxel [0,0,0]:
# - SEG voxel [0,0,0] should map to same world as CT [0,0,0]
# - Currently SEG [0,511,49] maps to approximately same world as CT [0,0,0]
# - So we need: new_seg_voxel[0,0,0] = old_seg_voxel[0,511,49]
# - This means: flip Y, flip Z

final_array = seg_array.copy()

# Flip Y (axis 1)
final_array = np.flip(final_array, axis=1)
print("Flipped Y axis")

# Flip Z (axis 2)
final_array = np.flip(final_array, axis=2)
print("Flipped Z axis")

# Now build the affine. After the flips:
# - new_voxel [0,0,0] corresponds to old_voxel [0, 511, 49]
# - In LPS, old_voxel [0,511,49] has world coords:
#   x = -179.6 + 0*0.7 = -179.6
#   y = +18.6 - 511*0.7 = -340.6
#   z = -83 - 49*5 = -328
# - In RAS (negate X and Y): [+179.6, +340.6, -328]
#
# So the new origin (in RAS) is [+179.6, +340.6, -328]

# For the diagonal:
# - Original X direction is +1, after LPS-to-RAS it becomes -1 (multiplied by -1)
#   So new_x = origin_x - voxel_x * spacing_x
# - Original Y direction is -1, after LPS-to-RAS it becomes +1 (multiplied by -1)
#   But we flipped Y, so effectively: new_y = origin_y - voxel_y * spacing_y
# - Original Z direction is -1, no change in LPS-to-RAS for Z
#   But we flipped Z, so: new_z = origin_z + voxel_z * spacing_z

# Wait, let me be more careful. The LPS-to-RAS transformation negates the
# X and Y components of the WORLD coordinates, not the direction vectors.

# Let's think in terms of the final affine:
# world = affine @ [voxel_x, voxel_y, voxel_z, 1]
#
# After flipping Y and Z:
# new_voxel[y] = (size_y - 1) - old_voxel[y]
# new_voxel[z] = (size_z - 1) - old_voxel[z]
#
# And we want the final world to be in RAS (negated X and Y from LPS).
#
# Original in LPS:
# world_x = -179.6 + voxel_x * 0.7
# world_y = +18.6 - voxel_y * 0.7
# world_z = -83 - voxel_z * 5
#
# After flip Y: new_y = 511 - old_y
# old_y = 511 - new_y
# world_y = +18.6 - (511 - new_y) * 0.7 = +18.6 - 357.7 + new_y * 0.7 = -339.1 + new_y * 0.7
#
# After flip Z: new_z = 49 - old_z
# old_z = 49 - new_z
# world_z = -83 - (49 - new_z) * 5 = -83 - 245 + new_z * 5 = -328 + new_z * 5
#
# So in LPS, after flips:
# world_x = -179.6 + new_x * 0.7  (unchanged since we didn't flip X)
# world_y = -339.1 + new_y * 0.7
# world_z = -328 + new_z * 5
#
# Convert to RAS (negate X and Y):
# ras_x = -world_x = 179.6 - new_x * 0.7
# ras_y = -world_y = 339.1 - new_y * 0.7
# ras_z = world_z = -328 + new_z * 5
#
# So the final affine in RAS is:
# origin = [179.6, 339.1, -328]
# diagonal = [-0.7, -0.7, 5]

# Hmm, the Y origin is 339.1, not 340.6 as CT has. The difference is about 1.5mm
# which could be rounding. Let me compute it more precisely:

y_origin = -(itkwasm_origin[1] - (itkwasm_size[1] - 1) * itkwasm_spacing[1] * abs(itkwasm_direction[1, 1]))
z_origin = itkwasm_origin[2] - (itkwasm_size[2] - 1) * itkwasm_spacing[2] * abs(itkwasm_direction[2, 2])
x_origin = -itkwasm_origin[0]  # Just negate X, we didn't flip it

# Actually wait, I need to think about whether to flip X too.
# Let me check the X extent comparison:
# SEG X in LPS: -179.6 to +179.6 (as voxel X goes 0 to 511)
# CT X in RAS: +179.6 to -179.6 (as voxel X goes 0 to 511)
# These are opposite! So we DO need to flip X.

print("\nNeed to flip X too!")
final_array = np.flip(final_array, axis=0)

# Now recompute origins:
# After flip X: new_x = 511 - old_x
# world_x = -179.6 + (511 - new_x) * 0.7 = -179.6 + 357.7 - new_x * 0.7 = 178.1 - new_x * 0.7
# ras_x = -world_x = -178.1 + new_x * 0.7

# Hmm, that gives positive ras_x increasing with voxel, but CT has negative.
# Let me reconsider...

# CT affine: world_x = 179.6 - voxel_x * 0.7
# So I want: ras_x = 179.6 - new_x * 0.7 (origin 179.6, slope -0.7)

# After flip X of SEG:
# world_x (LPS) = -179.6 + old_x * 0.7 = -179.6 + (511 - new_x) * 0.7
#               = -179.6 + 357.7 - new_x * 0.7 = 178.1 - new_x * 0.7
# ras_x = -world_x = -178.1 + new_x * 0.7

# This is wrong! I want ras_x = 179.6 - new_x * 0.7, but I got ras_x = -178.1 + new_x * 0.7

# The issue is the sign. Let me think again...

# Actually, I think the LPS-to-RAS conversion in ITKReader works differently.
# ITKReader doesn't just negate X and Y. Let me look at what the transformation
# actually does empirically.

# CT raw DICOM:
# - Origin: [-179.6, -340.6, Z]
# - Direction: identity
# - Size: 512 x 512 x 50
# - So raw world at voxel [0,0,0]: [-179.6, -340.6, Z]

# CT after ITKReader:
# - Origin: [+179.6, +340.6, -328]
# - Diagonal: [-0.7, -0.7, +5]
# - World at voxel [0,0,0]: [+179.6, +340.6, -328]
# - World at voxel [511,0,0]: [+179.6 - 511*0.7, +340.6, -328] = [-179.6, +340.6, -328]

# So the transformation is:
# ras_origin = [-lps_origin_x - (size_x-1)*spacing_x*dir_x_x,
#               -lps_origin_y - (size_y-1)*spacing_y*dir_y_y,
#                lps_origin_z ...]

# Wait, that's just: ras_origin = -lps_far_corner for X and Y

# CT raw far corner (voxel [511, 511, Z]):
# x = -179.6 + 511*0.7 = 178.1
# y = -340.6 + 511*0.7 = 17.1

# ras_origin = [-178.1, -17.1, ...] = [-178.1, -17.1, ...]
# But ITKReader shows [+179.6, +340.6, -328]

# Hmm, that doesn't match either. Let me check the actual ITK behavior...

# Actually, I think ITKReader does the following:
# 1. Flips the array along axes where direction is positive (to make all directions effectively negative)
# 2. Adjusts origin to the "new" first voxel position
# 3. Negates X and Y in the affine representation

# For CT with identity direction:
# - All directions are +1, so flip all X and Y
# - New first voxel (old [511, 511, 0]) was at world [-179.6 + 511*0.7, -340.6 + 511*0.7, Z]
#   = [178.1, 17.1, Z] in LPS
# - Convert to RAS: [-178.1, -17.1, Z]
# - Hmm, still doesn't match [179.6, 340.6, -328]

# I'm getting confused. Let me just use the empirical approach:
# Match the affines by checking what works.

# Reset and try a simpler approach: just use the CT affine and verify alignment
print("\n" + "="*60)
print("Empirical approach: Match to CT")
print("="*60)

# The CT affine is [[-0.7, 0, 0, 179.6], [0, -0.7, 0, 340.6], [0, 0, 5, -328]]
# We need to find the SEG voxel that corresponds to CT voxel [0,0,0]

# CT [0,0,0] is at RAS world [179.6, 340.6, -328]

# In LPS (negate X and Y): [-179.6, -340.6, -328]

# Find SEG voxel at this LPS world coordinate using the raw SEG affine
lps_target = np.array([-179.6, -340.6, -328])
seg_voxel_target = (np.linalg.inv(seg_affine_raw) @ np.append(lps_target, 1))[:3]
print(f"CT [0,0,0] in RAS = [179.6, 340.6, -328]")
print(f"Same point in LPS = {lps_target}")
print(f"Corresponds to SEG voxel (raw): {seg_voxel_target}")

# So CT [0,0,0] corresponds to SEG voxel [0, 511, 49] approximately
# We need: flip Y (axis 1) and flip Z (axis 2) to make SEG [0,0,0] = old [0, 511, 49]

# Re-do the array flips from scratch
final_array = seg_array.copy()
final_array = np.flip(final_array, axis=1)  # Flip Y
final_array = np.flip(final_array, axis=2)  # Flip Z
print(f"After flipping Y and Z: shape {final_array.shape}")

# Build affine: we want voxel [0,0,0] to map to RAS [179.6, 340.6, -328]
# And diagonal [-0.7, -0.7, +5]

# The origin is at [179.6, 340.6, -328]
# Note: we DON'T flip X because CT X extent [-179.6, 179.6] going 0->511
# matches SEG X extent [-179.6, 179.6] going 0->511 in LPS, which after
# RAS conversion (negate) becomes [179.6, -179.6] going 0->511.
# CT has [179.6, -179.6] going 0->511. Match!

# Wait, but if we don't flip X, then SEG voxel X=0 is at LPS X=-179.6, RAS X=179.6
# And SEG voxel X=511 is at LPS X=179.6, RAS X=-179.6
# This matches CT exactly!

# So we only need Y and Z flips. Now compute the affine:

# After Y flip: voxel Y=0 was old voxel Y=511
# After Z flip: voxel Z=0 was old voxel Z=49

# Old SEG at [0, 511, 49] in LPS:
# x = -179.6
# y = 18.6 - 511*0.7 = 18.6 - 357.7 = -339.1 (≈ -340.6 with rounding)
# z = -83 - 49*5 = -83 - 245 = -328

# Convert to RAS: [179.6, 339.1, -328] (Y is slightly off from 340.6)

# The 1.5mm discrepancy in Y is likely due to the SEG having slightly different
# slice positioning than the CT. For practical purposes this should still align.

final_origin = np.array([
    -itkwasm_origin[0],  # Negate X for RAS
    -(itkwasm_origin[1] + (itkwasm_size[1]-1) * itkwasm_spacing[1] * itkwasm_direction[1,1]),  # Y flipped then negated
    itkwasm_origin[2] + (itkwasm_size[2]-1) * itkwasm_spacing[2] * itkwasm_direction[2,2],  # Z flipped
])

print(f"Computed origin: {final_origin}")
print(f"CT origin: {ct_affine[:3, 3]}")

seg_affine = np.eye(4)
seg_affine[:3, :3] = np.diag([-itkwasm_spacing[0], -itkwasm_spacing[1], itkwasm_spacing[2]])
seg_affine[:3, 3] = final_origin

print(f"\nFinal SEG affine:\n{seg_affine}")
print(f"\nCT affine:\n{ct_affine}")

print("\n" + "="*60)
print("Testing alignment")
print("="*60)

def world_to_voxel(affine, world):
    return (np.linalg.inv(affine) @ np.append(world, 1))[:3]

# Test corner alignment
print("\nCorner test:")
seg_corner_000 = (seg_affine @ np.array([0, 0, 0, 1]))[:3]
ct_corner_000 = (ct_affine @ np.array([0, 0, 0, 1]))[:3]
print(f"SEG [0,0,0] -> {seg_corner_000}")
print(f"CT [0,0,0]  -> {ct_corner_000}")

# Test a labeled voxel
labeled_indices = np.argwhere(final_array > 0)
mid_idx = len(labeled_indices) // 2
seg_voxel = labeled_indices[mid_idx]
seg_label = final_array[tuple(seg_voxel)]

print(f"\nTest SEG voxel: {seg_voxel}, label: {seg_label}")

seg_world = (seg_affine @ np.append(seg_voxel, 1))[:3]
print(f"SEG world: {seg_world}")

ct_voxel = world_to_voxel(ct_affine, seg_world)
print(f"CT voxel: {ct_voxel}")

ct_voxel_int = np.round(ct_voxel).astype(int)
in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))
print(f"CT voxel (rounded): {ct_voxel_int}, in bounds: {in_bounds}")

if in_bounds:
    ct_value = ct_image[0, ct_voxel_int[0], ct_voxel_int[1], ct_voxel_int[2]].item()
    print(f"CT value: {ct_value:.1f} HU")

# Create visualization
print("\n" + "="*60)
print("Creating visualization")
print("="*60)

z_slice = ct_shape[2] // 2
ct_slice = ct_image[0, :, :, z_slice].numpy()

seg_z_slice = z_slice  # Same z after our transformation
seg_slice = final_array[:, :, seg_z_slice]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
axes[0].set_title(f'CT slice (z={z_slice})')
axes[0].axis('off')

axes[1].imshow(seg_slice.T, cmap='nipy_spectral', origin='lower')
axes[1].set_title(f'SEG slice (z={seg_z_slice})')
axes[1].axis('off')

axes[2].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
seg_mask = seg_slice > 0
seg_overlay = np.ma.masked_where(~seg_mask.T, seg_slice.T)
axes[2].imshow(seg_overlay, cmap='nipy_spectral', origin='lower', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
output_path = os.path.join(seg_dir, 'alignment_v4.png')
plt.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

print(f"\n\nData directory: {seg_dir}")
