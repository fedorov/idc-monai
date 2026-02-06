#!/usr/bin/env python
"""Final transform v2: Account for different ImageOrientationPatient between CT and SEG."""

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

seg_dir = tempfile.mkdtemp(prefix="idc_final2_")
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
print("Key insight: Raw DICOM orientations differ")
print("="*60)

# CT DICOM: ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
#   Y direction = +1 (voxel Y increases -> world Y increases)
#   Origin Y = -340.6 (minimum Y)
#
# SEG DICOM: ImageOrientationPatient = [1, 0, 0, 0, -1, 0]
#   Y direction = -1 (voxel Y increases -> world Y decreases)
#   Origin Y = +18.6 (maximum Y)
#
# Both cover Y range [-340.6, +18.6], just expressed differently.

# ITKWasm gives us SEG with direction [1, -1, -1]
# meaning X+, Y-, Z-

# MONAI ITKReader transforms CT to have affine diagonal [-0.7, -0.7, +5]
# This means:
# - voxel X increases -> world X decreases (flipped from raw CT X+ direction)
# - voxel Y increases -> world Y decreases (same as raw CT Y+ direction BUT CT was flipped by ITKReader)
# - voxel Z increases -> world Z increases

# Wait, let me trace through what ITKReader actually does to the CT:
# Raw CT: direction [1,0,0,0,1,0,0,0,1] = identity, origin at [-179.6, -340.6, Z]
# After ITKReader: diagonal [-0.7, -0.7, +5], origin at [+179.6, +340.6, Z]
# This is consistent with flipping X and Y arrays.

# The question is: what transformation do we need for SEG to match?

# Let me think about this physically:
# CT voxel [0, 0, 0] is at world [179.6, 340.6, -328] after ITKReader
# This should correspond to the physical corner at (RIGHT, ANTERIOR, INFERIOR)
#
# SEG has raw origin at [-179.6, +18.6, -83] which is (LEFT, ANTERIOR, SUPERIOR)
# and direction [1, -1, -1]
# So SEG voxel [0, 0, 0] is at (LEFT, ANTERIOR, SUPERIOR) corner

# For SEG to match CT's convention, SEG voxel [0, 0, 0] should be at
# (RIGHT, ANTERIOR, INFERIOR) = max X, max Y, min Z

# Current SEG voxel [0,0,0] is at: origin = [-179.6, +18.6, -83] = (LEFT, ANT, SUP)
# We need SEG voxel [0,0,0] to be at: (RIGHT, ANT, INF) = [+179.6, +18.6, -328]
# Wait, that's not right either because the Y values should be the same physical location

# Let me compute the physical corners of both volumes:

print("\nCT physical corners (after ITKReader):")
ct_shape = ct_image.shape[1:]
for name, voxel in [("origin (0,0,0)", [0,0,0]), ("far corner", [ct_shape[0]-1, ct_shape[1]-1, ct_shape[2]-1])]:
    world = (ct_affine @ np.append(voxel, 1))[:3]
    print(f"  CT voxel {voxel} -> world {world}")

print("\nSEG physical corners (raw from ITKWasm, after transpose):")
seg_array = np.transpose(itkwasm_array, (2, 1, 0))
seg_shape = seg_array.shape
seg_affine_raw = np.eye(4)
seg_affine_raw[:3, :3] = itkwasm_direction @ np.diag(itkwasm_spacing)
seg_affine_raw[:3, 3] = itkwasm_origin
for name, voxel in [("origin (0,0,0)", [0,0,0]), ("far corner", [seg_shape[0]-1, seg_shape[1]-1, seg_shape[2]-1])]:
    world = (seg_affine_raw @ np.append(voxel, 1))[:3]
    print(f"  SEG voxel {voxel} -> world {world}")

# From raw DICOM, we know:
# CT covers: X [-179.6, 179.6], Y [-340.6, 18.6], Z [-243, -128] (from DICOM file positions)
# SEG covers: X [-179.6, 179.6], Y [18.6, -340.6] (same, different direction), Z [-328, -83]

# The Y ranges ARE the same, just the direction is different!
# CT raw has origin Y = -340.6 (min Y), direction +1 (increases toward +18.6)
# SEG raw has origin Y = +18.6 (max Y), direction -1 (decreases toward -340.6)

print("\n" + "="*60)
print("Correct approach: Match the world coordinate mapping")
print("="*60)

# The goal is: for any world coordinate that exists in both volumes,
# the voxel indices should give the correct data.

# MONAI ITKReader flips the CT so that:
# - CT affine diagonal is [-0.7, -0.7, +5]
# - CT voxel [0,0,0] is at world [179.6, 340.6, -328]
#
# Wait, the CT origin after ITKReader is [179.6, 340.6, -328]
# But raw CT DICOM has origin at [-179.6, -340.6, -243]
#
# The CT Z changed from -243 to -328. That's a range issue - there are 50 slices
# with 5mm spacing, so range is 245mm. From -243 to (-243 + 49*5) = +2?
# No wait, the raw CT files showed Z range -243 to -128, which is only 115mm for 50 slices?
# That's 115/49 = 2.3mm spacing, not 5mm.

# Actually looking at the data more carefully:
# CT Z range: -243 to -128 = 115mm
# But spacing is 5mm and we have 50 slices... that doesn't add up.

# Wait, the CT we see has shape (512, 512, 50) and ITKReader gives Z spacing of 5.
# So the Z extent should be 49 * 5 = 245mm.
# ITKReader origin Z is -328, so Z range is [-328, -328+245] = [-328, -83]

# That matches the SEG Z range! So Z is fine.

# The issue is with Y. Let me trace it again:
# - CT raw DICOM: origin [-179.6, -340.6, some_z], direction [1, 1, 1]
# - ITKReader flips X and Y, giving origin [179.6, 340.6, -328], diagonal [-0.7, -0.7, 5]

# For CT after ITKReader:
# voxel [0, 0, 0] -> world [179.6, 340.6, -328]
# voxel [0, 511, 0] -> world [179.6, 340.6 - 511*0.7, -328] = [179.6, -18.1, -328]
# voxel [511, 511, 0] -> world [-179.6, -18.1, -328]

# So CT Y range after ITKReader: 340.6 to -18.1 (as voxel Y goes 0 to 511)

# For SEG raw:
# voxel [0, 0, 0] -> world [-179.6, 18.6, -83]
# voxel [0, 511, 0] -> world [-179.6, 18.6 - 511*0.7, -83] = [-179.6, -340.1, -83]

# So SEG Y range: 18.6 to -340.1 (as voxel Y goes 0 to 511)

# CT Y range: [340.6, -18.1] = total 358.7
# SEG Y range: [18.6, -340.1] = total 358.7

# THESE ARE NOT THE SAME RANGE!
# CT Y: 340.6 to -18.1
# SEG Y: 18.6 to -340.1

# There's a ~322 offset. The volumes don't overlap in Y!

# But wait - the user said these should align. Let me check if maybe
# I'm misreading the raw DICOM...

# From the check_dicom_metadata.py output:
# CT First slice ImagePositionPatient: [-179.64844, -340.64844, -243]
# SEG First frame ImagePositionPatient: [-179.648438, 18.6484375, -323]

# These Y values are different: CT Y = -340.6, SEG Y = +18.6

# But the ImageOrientationPatient is different too:
# CT: [1, 0, 0, 0, 1, 0] -> row=[1,0,0], col=[0,1,0]
# SEG: [1, 0, 0, 0, -1, 0] -> row=[1,0,0], col=[0,-1,0]

# For CT: the column direction is +Y, so going down rows increases Y
# For SEG: the column direction is -Y, so going down rows decreases Y

# The ImagePositionPatient is the position of the FIRST pixel (top-left corner)
# CT first pixel is at Y = -340.6, and column direction is +Y
# So CT bottom pixel is at Y = -340.6 + 511*0.7 = -340.6 + 357.7 = +17.1 ≈ +18.6!

# SEG first pixel is at Y = +18.6, and column direction is -Y
# So SEG bottom pixel is at Y = +18.6 - 511*0.7 = +18.6 - 357.7 = -339.1 ≈ -340.6!

# THEY DO ALIGN! The CT and SEG cover the same Y range, just the pixel order is flipped.
# CT: top Y = -340.6, bottom Y = +18.6 (going down = increasing Y)
# SEG: top Y = +18.6, bottom Y = -340.6 (going down = decreasing Y)

# So when ITKReader flips the CT Y, it should match the SEG's natural order!

# Let me verify: after ITKReader flips CT Y:
# New CT origin is at what was the bottom of the image
# Original CT bottom pixel Y = -340.6 + 511*0.7 = +17.1
# After flip, this becomes voxel Y=0

# Hmm, but the ITKReader CT affine shows origin Y = +340.6, not +17.1

# I think the confusion is that ITKReader flips AND negates.
# If original CT goes from Y=-340.6 (top) to Y=+17.1 (bottom) with +Y direction,
# After flipping the array, voxel[0] now corresponds to what was the bottom.
# But the affine is built differently - it negates the spacing and adjusts origin.

# Let me just verify empirically with a labeled point:

print("\n" + "="*60)
print("Empirical verification with physical world coordinates")
print("="*60)

# Find a labeled voxel in SEG
labeled_indices = np.argwhere(seg_array > 0)
mid_idx = len(labeled_indices) // 2
seg_voxel = labeled_indices[mid_idx]
seg_label = seg_array[tuple(seg_voxel)]

print(f"\nTest SEG voxel: {seg_voxel}, label: {seg_label}")

# Get world coordinates using raw SEG affine
seg_world_raw = (seg_affine_raw @ np.append(seg_voxel, 1))[:3]
print(f"SEG world (raw affine): {seg_world_raw}")

# Now find the CT voxel at this world coordinate
ct_voxel_from_world = (np.linalg.inv(ct_affine) @ np.append(seg_world_raw, 1))[:3]
print(f"CT voxel (from world): {ct_voxel_from_world}")

ct_voxel_int = np.round(ct_voxel_from_world).astype(int)
in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))
print(f"CT voxel (rounded): {ct_voxel_int}, in bounds: {in_bounds}")

if in_bounds:
    ct_value = ct_image[0, ct_voxel_int[0], ct_voxel_int[1], ct_voxel_int[2]].item()
    print(f"CT value: {ct_value:.1f} HU")

# If not in bounds, let's understand why
if not in_bounds:
    print("\nAnalyzing the mismatch:")
    print(f"CT shape: {ct_shape}")
    print(f"CT voxel needed: {ct_voxel_int}")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        if ct_voxel_int[i] < 0:
            print(f"  {axis}: {ct_voxel_int[i]} is below 0")
        elif ct_voxel_int[i] >= ct_shape[i]:
            print(f"  {axis}: {ct_voxel_int[i]} is >= {ct_shape[i]} (out by {ct_voxel_int[i] - ct_shape[i] + 1})")

    # Check what world coordinates the CT covers
    ct_world_min = (ct_affine @ np.array([ct_shape[0]-1, ct_shape[1]-1, 0, 1]))[:3]
    ct_world_max = (ct_affine @ np.array([0, 0, ct_shape[2]-1, 1]))[:3]
    print(f"\nCT world extent:")
    print(f"  X: [{min(ct_world_min[0], ct_world_max[0]):.1f}, {max(ct_world_min[0], ct_world_max[0]):.1f}]")
    print(f"  Y: [{min(ct_world_min[1], ct_world_max[1]):.1f}, {max(ct_world_min[1], ct_world_max[1]):.1f}]")
    print(f"  Z: [{min(ct_world_min[2], ct_world_max[2]):.1f}, {max(ct_world_min[2], ct_world_max[2]):.1f}]")

    seg_world_min = (seg_affine_raw @ np.array([seg_shape[0]-1, seg_shape[1]-1, 0, 1]))[:3]
    seg_world_max = (seg_affine_raw @ np.array([0, 0, seg_shape[2]-1, 1]))[:3]
    print(f"\nSEG world extent (raw affine):")
    print(f"  X: [{min(seg_world_min[0], seg_world_max[0]):.1f}, {max(seg_world_min[0], seg_world_max[0]):.1f}]")
    print(f"  Y: [{min(seg_world_min[1], seg_world_max[1]):.1f}, {max(seg_world_min[1], seg_world_max[1]):.1f}]")
    print(f"  Z: [{min(seg_world_min[2], seg_world_max[2]):.1f}, {max(seg_world_min[2], seg_world_max[2]):.1f}]")

print(f"\n\nData directory: {seg_dir}")
