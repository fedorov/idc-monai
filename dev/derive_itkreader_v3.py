#!/usr/bin/env python
"""Derive the transformation - v3 with correct understanding of direction."""

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

seg_dir = tempfile.mkdtemp(prefix="idc_derive3_")
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
print("Step 2: Understanding the key insight")
print("="*60)

# The key insight: MONAI's ITKReader applies a CONSISTENT convention:
# - X increases to the RIGHT (patient's left, anatomical Left)
# - Y increases ANTERIOR (towards patient's front)
# - Z increases SUPERIOR (towards patient's head)
#
# When the original DICOM has direction != identity, ITKReader reorders
# the array so that the above holds, and adjusts the affine accordingly.
#
# The final affine diagonal signs indicate:
# - Negative X: voxel X increases -> world X decreases (R to L)
# - Negative Y: voxel Y increases -> world Y decreases (A to P)
# - Positive Z: voxel Z increases -> world Z increases (I to S)
#
# This is the RAI (Right-Anterior-Inferior to Left-Posterior-Superior) convention
# where voxel [0,0,0] is at the Right-Anterior-Inferior corner.

# For CT with direction [[1,0,0],[0,1,0],[0,0,1]] (identity):
# - Raw origin is at some corner
# - ITKReader flips X and Y arrays to get [-,-,+] pattern
# - Origin moves to the R-A-I corner

# For SEG with direction [[1,0,0],[0,-1,0],[0,0,-1]]:
# - X direction = +1: same as CT
# - Y direction = -1: OPPOSITE to CT's Y=+1
# - Z direction = -1: OPPOSITE to CT's Z=+1
#
# So: SEG X needs same treatment as CT X
#     SEG Y needs OPPOSITE treatment (array already in opposite order)
#     SEG Z needs OPPOSITE treatment

print("CT corner [0,0,0] -> world:", ct_affine @ np.array([0,0,0,1]))
print("CT corner [511,511,49] -> world:", ct_affine @ np.array([511,511,49,1]))

# The CT [0,0,0] is at world [179.6, 340.6, -328] which is the R-A-I corner
# The CT [511,511,49] is at world [-179.6, -18.6, -83] which is the L-P-S corner

print("\n" + "="*60)
print("Step 3: Match ITKReader convention for SEG")
print("="*60)

# SEG raw has direction [[1,0,0],[0,-1,0],[0,0,-1]]
# This means:
# - SEG voxel X increases -> world X increases (+X = L)
# - SEG voxel Y increases -> world Y decreases (-Y = P, so Y increases = P to A)
# - SEG voxel Z increases -> world Z decreases (-Z = I, so Z increases = I to S)
#
# Wait, that's confusing. Let me be more careful.
#
# The affine formula is: world = origin + direction @ (voxel * spacing)
# For diagonal direction matrix:
# world_i = origin_i + direction[i,i] * spacing[i] * voxel_i
#
# SEG: direction = [[1,0,0],[0,-1,0],[0,0,-1]]
# - world_x = origin_x + 1 * spacing_x * voxel_x  -> as voxel_x increases, world_x increases
# - world_y = origin_y + (-1) * spacing_y * voxel_y -> as voxel_y increases, world_y DECREASES
# - world_z = origin_z + (-1) * spacing_z * voxel_z -> as voxel_z increases, world_z DECREASES
#
# CT after ITKReader: diagonal = [-0.7, -0.7, +5]
# - world_x = origin_x + (-0.7) * voxel_x -> as voxel_x increases, world_x DECREASES
# - world_y = origin_y + (-0.7) * voxel_y -> as voxel_y increases, world_y DECREASES
# - world_z = origin_z + (+5) * voxel_z -> as voxel_z increases, world_z INCREASES
#
# So to match:
# - SEG X: world increases with voxel -> need to flip so world DECREASES with voxel
# - SEG Y: world decreases with voxel -> already matches! But wait...
# - SEG Z: world decreases with voxel -> need to flip so world INCREASES with voxel

# Let me verify with corners:
# SEG raw origin is [-179.6, 18.6, -83]
# SEG voxel [0,0,0] -> world = origin = [-179.6, 18.6, -83]
# SEG voxel [511,0,0] -> world = [-179.6 + 511*0.7, 18.6, -83] = [179.6, 18.6, -83]
# SEG voxel [0,511,0] -> world = [-179.6, 18.6 - 511*0.7, -83] = [-179.6, -340.6, -83]
# SEG voxel [0,0,49] -> world = [-179.6, 18.6, -83 - 49*5] = [-179.6, 18.6, -328]

print("\nSEG raw corners:")
print(f"  [0,0,0] -> {itkwasm_origin}")
print(f"  [511,0,0] -> X varies: {itkwasm_origin + np.array([511*itkwasm_spacing[0]*itkwasm_direction[0,0], 0, 0])}")
print(f"  [0,511,0] -> Y varies: {itkwasm_origin + np.array([0, 511*itkwasm_spacing[1]*itkwasm_direction[1,1], 0])}")
print(f"  [0,0,49] -> Z varies: {itkwasm_origin + np.array([0, 0, 49*itkwasm_spacing[2]*itkwasm_direction[2,2]])}")

# SEG world extent:
# X: -179.6 to 179.6
# Y: 18.6 to -340.6 (decreases as voxel increases due to dir=-1)
# Z: -83 to -328 (decreases as voxel increases due to dir=-1)

# CT world extent (after ITKReader):
# X: 179.6 to -179.6 (decreases as voxel increases)
# Y: 340.6 to -18.6 (decreases as voxel increases)
# Z: -328 to -83 (increases as voxel increases)

# So the world extents are:
# X: same extent, opposite direction in array
# Y: DIFFERENT extents! SEG Y: [18.6, -340.6], CT Y: [340.6, -18.6]
# Z: same extent, opposite direction in array

# The Y extents are shifted! SEG Y range is [18.6, -340.6], CT Y range is [340.6, -18.6]
# These are the SAME physical range just expressed differently:
# SEG starts at Y=18.6 and goes to Y=-340.6
# CT starts at Y=340.6 and goes to Y=-18.6
#
# Wait, those are different ranges... Let me check the total span:
# SEG Y span: 18.6 - (-340.6) = 359.2 (but expressed as [max, min] in different order)
# CT Y span: 340.6 - (-18.6) = 359.2

# Actually they're different ranges! SEG covers [−340.6, 18.6] and CT covers [−18.6, 340.6]
# Hmm, that can't be right if they're supposed to align...

# Let me recalculate. SEG origin is at voxel [0,0,0]:
# For Y with dir=-1, as voxel Y goes from 0 to 511:
# world_y goes from 18.6 to 18.6 + (-1)*0.7*511 = 18.6 - 357.7 = -339.1

# Ah I see the issue - the EXTENTS ARE DIFFERENT because we're not accounting for
# the array order properly. The raw array from ITKWasm is in (Z, Y, X) order.

print("\n" + "="*60)
print("Step 4: Correct approach - match array layout to CT")
print("="*60)

# The CT after ITKReader has:
# - voxel [0,0,0] at world [179.6, 340.6, -328] (R-A-I corner)
# - voxel [511,511,49] at world [-179.6, -18.6, -83] (L-P-S corner)
# - Array layout: voxel_x increases L to R (world_x decreases)
#                 voxel_y increases A to P (world_y decreases)
#                 voxel_z increases I to S (world_z increases)

# The SEG raw has:
# - voxel [0,0,0] at world [-179.6, 18.6, -83]
# - voxel [511,511,49] at world [179.6, -340.6, -328]
# - Array layout: voxel_x increases L direction (world_x increases) - OPPOSITE to CT
#                 voxel_y increases P direction (world_y decreases) - SAME as CT
#                 voxel_z increases I direction (world_z decreases) - OPPOSITE to CT

# Wait, but the world extent for Y is still different:
# SEG Y: 18.6 down to -340.6
# CT Y: 340.6 down to -18.6

# Let me verify by computing all corners of both...

def compute_corner(affine, voxel):
    return (affine @ np.append(voxel, 1))[:3]

# Transpose SEG to (X,Y,Z)
seg_array = np.transpose(itkwasm_array, (2, 1, 0))

# Build naive SEG affine
seg_affine_naive = np.eye(4)
seg_affine_naive[:3, :3] = itkwasm_direction @ np.diag(itkwasm_spacing)
seg_affine_naive[:3, 3] = itkwasm_origin

print("All 8 corners of CT volume in world coordinates:")
ct_shape = ct_image.shape[1:]
for i in [0, ct_shape[0]-1]:
    for j in [0, ct_shape[1]-1]:
        for k in [0, ct_shape[2]-1]:
            w = compute_corner(ct_affine, [i,j,k])
            print(f"  [{i:3d},{j:3d},{k:2d}] -> [{w[0]:8.2f}, {w[1]:8.2f}, {w[2]:8.2f}]")

print("\nAll 8 corners of SEG volume (transposed, naive affine) in world coordinates:")
seg_shape = seg_array.shape
for i in [0, seg_shape[0]-1]:
    for j in [0, seg_shape[1]-1]:
        for k in [0, seg_shape[2]-1]:
            w = compute_corner(seg_affine_naive, [i,j,k])
            print(f"  [{i:3d},{j:3d},{k:2d}] -> [{w[0]:8.2f}, {w[1]:8.2f}, {w[2]:8.2f}]")

print("\n" + "="*60)
print("Step 5: Find the correct transformation")
print("="*60)

# From the corners, we can see:
# CT [0,0,0] -> [179.6, 340.6, -328]
# SEG [511,0,49] -> [179.6, 18.6, -328]  (after naive affine)
#
# These should be the SAME point if the volumes align!
# But CT Y=340.6 vs SEG Y=18.6... difference of 322!
#
# Wait, that's exactly 511 * 0.63 ≈ 322, which suggests the Y index mapping is off.

# Let me think about this differently.
# The issue is that ITKWasm gives us:
# - Array in (Z, Y, X) order
# - Metadata in (X, Y, Z) order
# - Origin at the "first voxel" position
#
# After transposing to (X, Y, Z), the array indexing changes.
# Original array[z,y,x] becomes transposed_array[x,y,z]
# So transposed[0,0,0] = original[0,0,0] which was at origin
# And transposed[511,511,49] = original[49,511,511] which was at far corner

# The SEG origin [-179.6, 18.6, -83] is at the raw array position [0,0,0]
# which after transpose is STILL [0,0,0].

# So SEG voxel [0,0,0] (transposed) is at world [-179.6, 18.6, -83]
# And SEG voxel [511,511,49] (transposed) should be at the opposite corner

# For CT, voxel [0,0,0] is at world [179.6, 340.6, -328]
# These are clearly different starting points!

# The CT's [0,0,0] corner is at [179.6, 340.6, -328]
# The SEG's [0,0,0] corner is at [-179.6, 18.6, -83]
#
# For these to represent overlapping volumes, we need to find which SEG corner
# corresponds to CT's [0,0,0] corner...

print("\nLooking for SEG corner that matches CT [0,0,0] = [179.6, 340.6, -328]:")
ct_corner_000 = compute_corner(ct_affine, [0,0,0])
print(f"Target: {ct_corner_000}")

best_match = None
best_dist = float('inf')
for i in [0, seg_shape[0]-1]:
    for j in [0, seg_shape[1]-1]:
        for k in [0, seg_shape[2]-1]:
            w = compute_corner(seg_affine_naive, [i,j,k])
            dist = np.linalg.norm(w - ct_corner_000)
            print(f"  SEG [{i:3d},{j:3d},{k:2d}] -> [{w[0]:8.2f}, {w[1]:8.2f}, {w[2]:8.2f}], dist={dist:.2f}")
            if dist < best_dist:
                best_dist = dist
                best_match = [i, j, k]

print(f"\nBest match: SEG {best_match} with distance {best_dist:.2f}")

print("\n" + "="*60)
print("Step 6: The answer")
print("="*60)

# The closest SEG corner to CT [0,0,0] is SEG [511, 0, 49]
# with coordinates [179.6, 18.6, -328]
# The Y difference of 322 (340.6 - 18.6) suggests the SEG and CT
# don't have the same Y extent!

# But wait - the user said SEG fully covers CT. Let me check total world extent...
ct_extent = {
    'x': [compute_corner(ct_affine, [0,0,0])[0], compute_corner(ct_affine, [511,0,0])[0]],
    'y': [compute_corner(ct_affine, [0,0,0])[1], compute_corner(ct_affine, [0,511,0])[1]],
    'z': [compute_corner(ct_affine, [0,0,0])[2], compute_corner(ct_affine, [0,0,49])[2]],
}
seg_extent = {
    'x': [compute_corner(seg_affine_naive, [0,0,0])[0], compute_corner(seg_affine_naive, [511,0,0])[0]],
    'y': [compute_corner(seg_affine_naive, [0,0,0])[1], compute_corner(seg_affine_naive, [0,511,0])[1]],
    'z': [compute_corner(seg_affine_naive, [0,0,0])[2], compute_corner(seg_affine_naive, [0,0,49])[2]],
}

print(f"CT X extent: {ct_extent['x']} (range: {abs(ct_extent['x'][1]-ct_extent['x'][0]):.1f})")
print(f"SEG X extent: {seg_extent['x']} (range: {abs(seg_extent['x'][1]-seg_extent['x'][0]):.1f})")
print(f"CT Y extent: {ct_extent['y']} (range: {abs(ct_extent['y'][1]-ct_extent['y'][0]):.1f})")
print(f"SEG Y extent: {seg_extent['y']} (range: {abs(seg_extent['y'][1]-seg_extent['y'][0]):.1f})")
print(f"CT Z extent: {ct_extent['z']} (range: {abs(ct_extent['z'][1]-ct_extent['z'][0]):.1f})")
print(f"SEG Z extent: {seg_extent['z']} (range: {abs(seg_extent['z'][1]-seg_extent['z'][0]):.1f})")

# They should have the same ranges!
# CT Y: 340.6 to -18.6 = range 359.2
# SEG Y: 18.6 to -340.6 = range 359.2 (same!)

# So the ranges ARE the same, just shifted. This means the CT and SEG
# may have different physical positions, OR there's something wrong with
# the metadata interpretation.

print("\n" + "="*60)
print("Step 7: Check ITKWasm vs dcmqi output directly")
print("="*60)

# From earlier comparison, ITKWasm and dcmqi produce IDENTICAL output
# So the issue must be with how ITKReader handles CT differently than raw DICOM

# Let me load the CT with ITK directly (not through MONAI's ITKReader)
import itk

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

ct_itk_spacing = np.array(ct_itk_image.GetSpacing())
ct_itk_origin = np.array(ct_itk_image.GetOrigin())
ct_itk_direction = np.array(ct_itk_image.GetDirection())

print(f"\nCT via ITK directly:")
print(f"  Spacing: {ct_itk_spacing}")
print(f"  Origin: {ct_itk_origin}")
print(f"  Direction:\n{ct_itk_direction}")

ct_itk_affine = np.eye(4)
ct_itk_affine[:3, :3] = ct_itk_direction @ np.diag(ct_itk_spacing)
ct_itk_affine[:3, 3] = ct_itk_origin
print(f"  Affine:\n{ct_itk_affine}")

print("\nCT via MONAI ITKReader (for comparison):")
print(f"  Affine:\n{ct_affine}")

print("\n--- Using ITK CT affine instead of MONAI ---")
ct_corner_000_itk = compute_corner(ct_itk_affine, [0,0,0])
print(f"CT [0,0,0] via ITK: {ct_corner_000_itk}")

# Now find SEG corner matching ITK CT corner
print(f"\nLooking for SEG corner matching CT (ITK) [0,0,0] = {ct_corner_000_itk}:")
best_match = None
best_dist = float('inf')
for i in [0, seg_shape[0]-1]:
    for j in [0, seg_shape[1]-1]:
        for k in [0, seg_shape[2]-1]:
            w = compute_corner(seg_affine_naive, [i,j,k])
            dist = np.linalg.norm(w - ct_corner_000_itk)
            if dist < best_dist:
                best_dist = dist
                best_match = [i, j, k]

print(f"Best match: SEG {best_match} with distance {best_dist:.2f}")

# Test a random voxel alignment using ITK CT affine
print("\n" + "="*60)
print("Step 8: Test alignment with ITK CT affine")
print("="*60)

labeled_indices = np.argwhere(seg_array > 0)
mid_idx = len(labeled_indices) // 2
seg_voxel = labeled_indices[mid_idx]
print(f"Test SEG voxel: {seg_voxel}")

seg_world = seg_affine_naive @ np.append(seg_voxel, 1)
print(f"SEG world (naive affine): {seg_world[:3]}")

# Note: ITK's array is also in different order...
# ITK returns arrays in (Z, Y, X) format
# So ITK array[z,y,x] corresponds to our transposed array[x,y,z]
# For the same WORLD position!

ct_voxel_itk = np.linalg.inv(ct_itk_affine) @ seg_world
print(f"CT voxel (via ITK affine): {ct_voxel_itk[:3]}")

# Convert from ITK's (X,Y,Z) to ITK array's (Z,Y,X)
ct_voxel_zyx = ct_voxel_itk[:3][::-1]
print(f"CT array index (Z,Y,X for ITK array): {ct_voxel_zyx}")

ct_array = itk.GetArrayFromImage(ct_itk_image)
print(f"CT ITK array shape (Z,Y,X): {ct_array.shape}")

ct_voxel_int = np.round(ct_voxel_zyx).astype(int)
if all(0 <= ct_voxel_int[i] < ct_array.shape[i] for i in range(3)):
    ct_value = ct_array[tuple(ct_voxel_int)]
    print(f"CT value at [{ct_voxel_int}]: {ct_value} HU")
else:
    print(f"CT voxel {ct_voxel_int} out of bounds for shape {ct_array.shape}")

print(f"\n\nData directory: {seg_dir}")
