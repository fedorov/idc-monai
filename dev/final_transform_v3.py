#!/usr/bin/env python
"""Final transform v3: Correctly handle the Y axis flip."""

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

seg_dir = tempfile.mkdtemp(prefix="idc_final3_")
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

# The CT and SEG volumes overlap in physical space, but have different
# axis directions. We need to transform SEG to match CT's convention.
#
# CT after ITKReader:
#   - Affine diagonal: [-0.7, -0.7, +5]
#   - Origin: [179.6, 340.6, -328]
#   - Y extent: [340.6, -18.6] (going from max to min as voxel Y increases)
#
# SEG raw (ITKWasm):
#   - Affine diagonal: [+0.7, -0.7, -5]
#   - Origin: [-179.6, 18.6, -83]
#   - Y extent: [18.6, -340.6] (going from 18.6 down as voxel Y increases)
#
# The Y extents ARE the same physical range, just the start/end are different:
# CT Y: 340.6 -> -18.6 (span 359.2)
# SEG Y: 18.6 -> -340.6 (span 359.2)
#
# These are offset by ~322. The SEG origin Y (18.6) equals CT's Y at voxel Y≈458:
# CT Y at voxel 0: 340.6
# CT Y at voxel Y: 340.6 - Y * 0.7 = 18.6 => Y = 322/0.7 ≈ 460
#
# But the CT only has 512 Y voxels, so voxel 460 is valid.
# The SEG Y extent [18.6, -340.6] maps to CT voxels [460, 971].
# Since CT only has voxels [0, 511], only SEG Y values [18.6, -18.6] overlap with CT!
# That's just 37 / 0.7 ≈ 53 voxels of overlap.

# Wait, that doesn't match what the user said. Let me reconsider.
#
# The issue is that the DICOM coordinate system is being interpreted differently.
# Looking at raw DICOM:
# - CT ImageOrientationPatient: [1, 0, 0, 0, 1, 0] -> column direction is +Y
# - SEG ImageOrientationPatient: [1, 0, 0, 0, -1, 0] -> column direction is -Y
#
# CT origin pixel is at [-179.6, -340.6, Z]
# As you go down rows (increase voxel Y), world Y increases (because col dir is +Y)
# So CT Y range: -340.6 to -340.6 + 511*0.7 = -340.6 to +17.1
#
# SEG origin pixel is at [-179.6, +18.6, Z]
# As you go down rows (increase voxel Y), world Y decreases (because col dir is -Y)
# So SEG Y range: +18.6 to +18.6 - 511*0.7 = +18.6 to -339.1
#
# CT Y: [-340.6, +17.1]
# SEG Y: [+18.6, -339.1]
#
# These DO overlap almost perfectly! The slight difference is rounding.

# So the issue is that ITKReader does something unexpected with the CT.
# It flips the array and changes the origin, resulting in an affine where
# the Y goes from 340.6 down to -18.6, which is the OPPOSITE of raw CT's
# Y range of -340.6 to +17.1.

# I think ITKReader negates BOTH axes AND flips the origin, which effectively
# mirrors the coordinate system. Let me trace through what happens:

# Raw CT:
#   origin = [-179.6, -340.6, Z]
#   direction = identity
#   affine diagonal = [+0.7, +0.7, Z_spacing]
#   world at voxel [0,0,0] = [-179.6, -340.6, Z]
#   world at voxel [511,0,0] = [-179.6 + 511*0.7, -340.6, Z] = [179.3, -340.6, Z]

# After ITKReader:
#   origin = [179.6, 340.6, -328]
#   affine diagonal = [-0.7, -0.7, +5]
#   world at voxel [0,0,0] = [179.6, 340.6, -328]
#   world at voxel [511,0,0] = [179.6 - 511*0.7, 340.6, -328] = [-179.3, 340.6, -328]

# So ITKReader:
# 1. Flips X: origin goes from -179.6 to +179.6, spacing becomes negative
# 2. Flips Y: origin goes from -340.6 to +340.6, spacing becomes negative
# 3. Z might be flipped too depending on the original direction

# For our SEG with direction [+1, -1, -1]:
# - X is +1 (same as CT raw) -> needs flip to get -0.7
# - Y is -1 (opposite of CT raw +1) -> ???
# - Z is -1 -> needs flip to get +5

# Let me think about Y more carefully:
# CT raw Y direction = +1, ITKReader gives Y spacing = -0.7
# So ITKReader multiplies direction by spacing and then negates?
# Or ITKReader always wants negative X and Y, positive Z?

# Looking at the result: ITKReader always produces [-spacing_x, -spacing_y, +spacing_z]
# regardless of original direction.

# For SEG:
# - X direction = +1 -> need to flip array X and use -spacing_x
# - Y direction = -1 -> the product is already -1*spacing = -0.7, but we want
#   the SAME Y extent as CT after ITKReader
# - Z direction = -1 -> need to flip array Z and use +spacing_z

# The key insight from the user: the Y axis IS flipped, and the extents overlap.
# So we need to flip Y as well to match the CT convention!

# Let me try: flip ALL THREE axes and adjust origin accordingly.

# Step 1: Transpose from (Z, Y, X) to (X, Y, Z)
seg_array = np.transpose(itkwasm_array, (2, 1, 0))
print(f"After transpose to (X,Y,Z): {seg_array.shape}")

# Step 2: Flip axes to match ITKReader convention
# ITKReader convention: negative X, negative Y, positive Z in diagonal
# SEG raw (after transpose): direction [+1, -1, -1]
# So: X needs flip (+ to -), Y needs flip (- to +, then to - -> stays same? No...)

# Actually, the issue is the WORLD EXTENT matching.
# CT after ITKReader has world Y extent [340.6, -18.6] (max to min)
# SEG raw has world Y extent [18.6, -340.6] (different range!)

# But the user says they overlap with a flip. Let me re-examine:
# CT raw DICOM origin: [-179.6, -340.6, ...]
# CT raw DICOM with 512 Y voxels and spacing 0.7 and direction +1:
#   Y at voxel 0: -340.6
#   Y at voxel 511: -340.6 + 511*0.7 = +17.1

# SEG raw DICOM origin: [-179.6, +18.6, ...]
# SEG raw DICOM with 512 Y voxels and spacing 0.7 and direction -1:
#   Y at voxel 0: +18.6
#   Y at voxel 511: +18.6 - 511*0.7 = -339.1

# CT Y world extent: [-340.6, +17.1]
# SEG Y world extent: [+18.6, -339.1] = [-339.1, +18.6]

# These ARE nearly the same! [-340.6, +17.1] vs [-339.1, +18.6]
# The small difference is because 18.6 ≠ 17.1 (about 1.5 mm difference)

# So the volumes DO overlap. The issue is that ITKReader transforms the CT
# in a way that makes its world extent appear as [340.6, -18.6] instead of
# the raw [-340.6, +17.1]. ITKReader must be negating the Y coordinates!

# ITKReader does: new_Y = -old_Y
# So raw CT Y [-340.6, +17.1] becomes ITKReader CT Y [+340.6, -17.1]

# Similarly, SEG raw Y [+18.6, -339.1] should become [-18.6, +339.1] after
# the same transformation!

# So for SEG to match ITKReader CT, we need to negate Y coordinates.
# This is done by: flip Y array, negate Y in origin

final_array = seg_array.copy()
final_origin = itkwasm_origin.copy()
final_spacing = itkwasm_spacing.copy()

# Flip X (because direction is +1, we want -1 like CT)
final_array = np.flip(final_array, axis=0)
final_origin[0] = itkwasm_origin[0] + (itkwasm_size[0] - 1) * itkwasm_spacing[0] * itkwasm_direction[0, 0]
print(f"After X flip, origin X: {final_origin[0]}")

# Flip Y (because we need to negate the Y world coordinate system)
final_array = np.flip(final_array, axis=1)
final_origin[1] = itkwasm_origin[1] + (itkwasm_size[1] - 1) * itkwasm_spacing[1] * itkwasm_direction[1, 1]
# Also negate to match ITKReader's coordinate negation
final_origin[1] = -final_origin[1]
print(f"After Y flip + negate, origin Y: {final_origin[1]}")

# Flip Z (because direction is -1, we want +1 like CT)
final_array = np.flip(final_array, axis=2)
final_origin[2] = itkwasm_origin[2] + (itkwasm_size[2] - 1) * itkwasm_spacing[2] * itkwasm_direction[2, 2]
print(f"After Z flip, origin Z: {final_origin[2]}")

# Negate X origin too to match ITKReader's convention
final_origin[0] = -final_origin[0]
print(f"After X negate, origin X: {final_origin[0]}")

# Build final affine with ITKReader convention: [-spacing, -spacing, +spacing]
seg_affine = np.eye(4)
seg_affine[:3, :3] = np.diag([-itkwasm_spacing[0], -itkwasm_spacing[1], itkwasm_spacing[2]])
seg_affine[:3, 3] = final_origin

print(f"\nFinal SEG affine:\n{seg_affine}")
print(f"\nCT affine (for comparison):\n{ct_affine}")

print("\n" + "="*60)
print("Testing alignment")
print("="*60)

def voxel_to_world(affine, voxel):
    return (affine @ np.append(voxel, 1))[:3]

def world_to_voxel(affine, world):
    return (np.linalg.inv(affine) @ np.append(world, 1))[:3]

# Test a labeled voxel
labeled_indices = np.argwhere(final_array > 0)
mid_idx = len(labeled_indices) // 2
seg_voxel = labeled_indices[mid_idx]
seg_label = final_array[tuple(seg_voxel)]

print(f"\nTest SEG voxel: {seg_voxel}, label: {seg_label}")

seg_world = voxel_to_world(seg_affine, seg_voxel)
print(f"SEG world: {seg_world}")

ct_voxel = world_to_voxel(ct_affine, seg_world)
print(f"CT voxel: {ct_voxel}")

ct_shape = ct_image.shape[1:]
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

# Find corresponding SEG z slice
ct_world_z = voxel_to_world(ct_affine, [0, 0, z_slice])[2]
seg_voxel_z = world_to_voxel(seg_affine, [0, 0, ct_world_z])[2]
seg_z_slice = int(round(seg_voxel_z))

print(f"CT z-slice {z_slice} -> world z={ct_world_z:.1f} -> SEG z-slice {seg_z_slice}")

if 0 <= seg_z_slice < final_array.shape[2]:
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
    output_path = os.path.join(seg_dir, 'alignment_v3.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")

print(f"\n\nData directory: {seg_dir}")
