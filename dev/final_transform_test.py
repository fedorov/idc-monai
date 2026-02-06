#!/usr/bin/env python
"""Final test: Apply correct transformation to match MONAI ITKReader convention."""

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

seg_dir = tempfile.mkdtemp(prefix="idc_final_")
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
print(f"SEG size (X,Y,Z): {itkwasm_size}")
print(f"SEG origin: {itkwasm_origin}")
print(f"SEG direction:\n{itkwasm_direction}")

print("\n" + "="*60)
print("Applying ITKReader-compatible transformation")
print("="*60)

# The algorithm to match MONAI ITKReader:
# MONAI ITKReader applies a transformation that results in:
# - Affine diagonal: [-spacing_x, -spacing_y, +spacing_z]
# - Origin at the corner where all affine diagonal signs point TO
#   (i.e., the corner at max_x, max_y, min_z in world coordinates)
#
# For our SEG with direction [[1,0,0],[0,-1,0],[0,0,-1]]:
# The raw affine diagonal would be: [+0.7, -0.7, -5]
# Target diagonal: [-0.7, -0.7, +5]
#
# We need to:
# 1. Flip X (change sign from + to -)
# 2. Keep Y (already -)
# 3. Flip Z (change sign from - to +)
#
# When we flip an axis in the array, we must adjust the origin to maintain
# correct world coordinate mapping.

# Step 1: Transpose from (Z, Y, X) to (X, Y, Z)
seg_array = np.transpose(itkwasm_array, (2, 1, 0))
print(f"After transpose to (X,Y,Z): {seg_array.shape}")

# Step 2: Determine which axes need flipping
# For each axis: if current diagonal sign != target diagonal sign, flip
current_diagonal = np.diag(itkwasm_direction) * itkwasm_spacing
target_diagonal = np.array([-itkwasm_spacing[0], -itkwasm_spacing[1], +itkwasm_spacing[2]])

print(f"\nCurrent affine diagonal: {current_diagonal}")
print(f"Target affine diagonal: {target_diagonal}")

flip_axes = []
for i in range(3):
    if np.sign(current_diagonal[i]) != np.sign(target_diagonal[i]):
        flip_axes.append(i)

print(f"Need to flip axes: {flip_axes}")

# Step 3: Apply flips and adjust origin
final_array = seg_array.copy()
final_origin = itkwasm_origin.copy()

for axis in flip_axes:
    # Flip the array
    final_array = np.flip(final_array, axis=axis)
    # Adjust origin: move to the opposite corner along this axis
    # new_origin = old_origin + (size-1) * spacing * direction
    final_origin[axis] = itkwasm_origin[axis] + (itkwasm_size[axis] - 1) * itkwasm_spacing[axis] * itkwasm_direction[axis, axis]
    print(f"Flipped axis {axis}, new origin[{axis}] = {final_origin[axis]}")

# Step 4: Build final affine
seg_affine = np.eye(4)
seg_affine[:3, :3] = np.diag(target_diagonal)
seg_affine[:3, 3] = final_origin

print(f"\nFinal SEG array shape: {final_array.shape}")
print(f"Final SEG affine:\n{seg_affine}")

print("\n" + "="*60)
print("Testing alignment")
print("="*60)

def voxel_to_world(affine, voxel):
    return (affine @ np.append(voxel, 1))[:3]

def world_to_voxel(affine, world):
    return (np.linalg.inv(affine) @ np.append(world, 1))[:3]

# Test corner alignment
print("\nCorner alignment test:")
print("SEG [0,0,0] -> world -> CT voxel:")
seg_corner = voxel_to_world(seg_affine, [0, 0, 0])
ct_corner = world_to_voxel(ct_affine, seg_corner)
print(f"  SEG [0,0,0] -> world {seg_corner} -> CT {ct_corner}")

print("\nCT [0,0,0] -> world -> SEG voxel:")
ct_corner_world = voxel_to_world(ct_affine, [0, 0, 0])
seg_from_ct = world_to_voxel(seg_affine, ct_corner_world)
print(f"  CT [0,0,0] -> world {ct_corner_world} -> SEG {seg_from_ct}")

# Test labeled voxel alignment
print("\n" + "="*60)
print("Labeled voxel alignment test")
print("="*60)

labeled_indices = np.argwhere(final_array > 0)
if len(labeled_indices) > 0:
    # Test several points
    for idx in [len(labeled_indices)//4, len(labeled_indices)//2, 3*len(labeled_indices)//4]:
        seg_voxel = labeled_indices[idx]
        seg_label = final_array[tuple(seg_voxel)]

        seg_world = voxel_to_world(seg_affine, seg_voxel)
        ct_voxel = world_to_voxel(ct_affine, seg_world)
        ct_voxel_int = np.round(ct_voxel).astype(int)

        ct_shape = ct_image.shape[1:]
        in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))

        print(f"\nSEG voxel {seg_voxel} (label={seg_label}):")
        print(f"  -> world: {seg_world}")
        print(f"  -> CT voxel: {ct_voxel_int}, in bounds: {in_bounds}")

        if in_bounds:
            ct_value = ct_image[0, ct_voxel_int[0], ct_voxel_int[1], ct_voxel_int[2]].item()
            print(f"  -> CT value: {ct_value:.1f} HU")

print("\n" + "="*60)
print("Creating visualization")
print("="*60)

# Find a slice with both CT content and segmentation
# Use the middle Z slice
ct_shape = ct_image.shape[1:]
z_slice = ct_shape[2] // 2

# Get CT slice
ct_slice = ct_image[0, :, :, z_slice].numpy()

# Get corresponding SEG slice - need to find the world coordinates and map back
# All voxels in CT slice z_slice correspond to some z in world
ct_world_z = voxel_to_world(ct_affine, [0, 0, z_slice])[2]

# Find SEG slice at same world z
seg_voxel_z = world_to_voxel(seg_affine, [0, 0, ct_world_z])[2]
seg_z_slice = int(round(seg_voxel_z))
print(f"CT z-slice {z_slice} corresponds to world z={ct_world_z:.1f}, SEG z-slice {seg_z_slice}")

if 0 <= seg_z_slice < final_array.shape[2]:
    seg_slice = final_array[:, :, seg_z_slice]

    # Create overlay visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CT slice
    axes[0].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
    axes[0].set_title(f'CT slice (z={z_slice})')
    axes[0].axis('off')

    # SEG slice
    axes[1].imshow(seg_slice.T, cmap='nipy_spectral', origin='lower')
    axes[1].set_title(f'SEG slice (z={seg_z_slice})')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
    seg_mask = seg_slice > 0
    seg_overlay = np.ma.masked_where(~seg_mask.T, seg_slice.T)
    axes[2].imshow(seg_overlay, cmap='nipy_spectral', origin='lower', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    output_path = os.path.join(seg_dir, 'alignment_test_final.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")

print(f"\n\nData directory: {seg_dir}")
