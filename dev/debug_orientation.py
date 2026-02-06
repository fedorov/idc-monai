#!/usr/bin/env python
"""Debug script to verify SEG and CT spatial alignment."""

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from idc_index import IDCClient
from itkwasm_dicom import read_segmentation
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data.image_reader import ITKReader
from monai.data import MetaTensor

# Initialize client and find a CT with segmentation
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
seg_dir = tempfile.mkdtemp(prefix="idc_debug_")
print(f"Downloading to {seg_dir}...")
client.download_from_selection(
    seriesInstanceUID=[demo_pair['image_uid'], demo_pair['seg_uid']],
    downloadDir=seg_dir,
    dirTemplate="%SeriesInstanceUID"
)

# Load CT with ITKReader
from pathlib import Path
image_path = os.path.join(seg_dir, demo_pair['image_uid'])
ct_load = Compose([
    LoadImaged(keys=["image"], reader=ITKReader()),
    EnsureChannelFirstd(keys=["image"]),
])
ct_data = ct_load({"image": image_path})
ct_image = ct_data["image"]
ct_affine = ct_image.affine.numpy()

print(f"\nCT shape: {ct_image.shape}")
print(f"CT affine:\n{ct_affine}")

# Load SEG with ITKWasm - straightforward approach
seg_path = Path(os.path.join(seg_dir, demo_pair['seg_uid']))
dcm_files = list(seg_path.glob("*.dcm"))
seg_image, overlay_info = read_segmentation(dcm_files[0])

# Transpose from ITKWasm's (Z, Y, X) to (X, Y, Z)
seg_array = np.asarray(seg_image.data).copy()
seg_array = np.transpose(seg_array, (2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)

spacing = np.array(seg_image.spacing)
origin = np.array(seg_image.origin)
direction = np.array(seg_image.direction).reshape(3, 3)
size = np.array(seg_image.size)

# ITKReader applies a LPS to "ITK" coordinate transformation that effectively:
# - Negates X origin and X direction (L becomes R)
# - Negates Y origin and Y direction (P becomes A)
# This places the origin at the opposite corner and uses negative spacing

# Apply the same transformation to match ITKReader's output
lps_to_itk = np.diag([-1, -1, 1])  # Negate X and Y

# Transform origin
origin_itk = lps_to_itk @ origin

# Transform direction
direction_itk = lps_to_itk @ direction

# Compute the corner where ITKReader would place the origin
# ITKReader places origin at the max X, max Y corner (in its coordinate system)
# We need to find where voxel [size-1, size-1, 0] maps to after the transformation

# Actually, let's just match what ITKReader does for CT:
# CT has direction [1,1,1] in DICOM, and ITKReader produces origin at [179.6, 340.6, -328]
# with diagonal [-0.7, -0.7, 5]

# For SEG with DICOM direction [1,-1,-1], after the LPS->ITK transform:
# direction becomes [-1, 1, -1]

# The simplest fix: compute the affine such that world coordinates match
# We want SEG voxel [x, y, z] to map to the same world as CT voxel [x, y, z]
# (assuming they should overlay)

# Let's try: use the CT's affine convention but with SEG's spacing
# This assumes the arrays are in the same order after transposition

# Actually, the real issue might be that ITKReader is doing something more complex.
# Let me just try negating the signs to match CT's affine pattern:

# CT affine pattern: negative X, negative Y, positive Z in the diagonal
# SEG currently: positive X, negative Y, negative Z

# Flip X axis (negate X direction and adjust origin)
# Flip Z axis (negate Z direction and adjust origin)

seg_affine = np.eye(4)

# For X: flip if CT has negative and we have positive
# CT has -0.7 for X, SEG has +0.7 -> need to flip
seg_array = np.flip(seg_array, axis=0)
# New origin X = old origin X + (size-1) * spacing * direction
new_origin_x = origin[0] + (size[0] - 1) * spacing[0] * direction[0, 0]

# For Y: CT has -0.7, SEG has -0.7 -> same, no flip needed

# For Z: CT has +5, SEG has -5 -> need to flip
seg_array = np.flip(seg_array, axis=2)
new_origin_z = origin[2] + (size[2] - 1) * spacing[2] * direction[2, 2]

new_origin = np.array([new_origin_x, origin[1], new_origin_z])

# Now build affine with same sign pattern as CT: negative X, negative Y, positive Z
seg_affine[:3, :3] = np.diag([-spacing[0], -spacing[1], spacing[2]])
seg_affine[:3, 3] = new_origin

print(f"After flip, new SEG origin: {new_origin}")
print(f"SEG affine diagonal: {np.diag(seg_affine[:3, :3])}")

seg_tensor = MetaTensor(seg_array)
seg_tensor.affine = seg_affine

print(f"\nSEG shape: {seg_tensor.shape}")
print(f"SEG affine:\n{seg_affine}")

# Helper functions
def voxel_to_world(affine, voxel):
    """Convert voxel to world coordinates."""
    voxel_h = np.append(voxel, 1)
    world_h = affine @ voxel_h
    return world_h[:3]

def world_to_voxel(affine, world):
    """Convert world to voxel coordinates."""
    world_h = np.append(world, 1)
    voxel_h = np.linalg.inv(affine) @ world_h
    return voxel_h[:3]

# Find a labeled voxel in the segmentation (near the center)
seg_np = seg_tensor.numpy()
labeled_indices = np.argwhere(seg_np > 0)
if len(labeled_indices) > 0:
    # Pick a point near the middle of the labeled region
    mid_idx = len(labeled_indices) // 2
    seg_voxel = labeled_indices[mid_idx]
    seg_label = seg_np[tuple(seg_voxel)]

    print(f"\n" + "="*60)
    print("Alignment test: SEG voxel -> World -> CT voxel")
    print("="*60)
    print(f"SEG voxel: {seg_voxel}, label: {seg_label}")

    # Convert to world coordinates
    seg_world = voxel_to_world(seg_affine, seg_voxel)
    print(f"World coordinates (from SEG): {seg_world}")

    # Convert to CT voxel coordinates
    ct_voxel = world_to_voxel(ct_affine, seg_world)
    print(f"CT voxel (from world): {ct_voxel}")

    # Check if CT voxel is within bounds
    ct_shape = ct_image.shape[1:]  # Remove channel dim
    ct_voxel_int = np.round(ct_voxel).astype(int)

    in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))
    print(f"CT voxel (rounded): {ct_voxel_int}, in bounds: {in_bounds}")

    if in_bounds:
        ct_value = ct_image[0, ct_voxel_int[0], ct_voxel_int[1], ct_voxel_int[2]].item()
        print(f"CT value at this location: {ct_value:.1f} HU")
    else:
        print("CT voxel is out of bounds - SEG and CT may not overlap or have alignment issue")

    # Visualize the slice
    print("\n" + "="*60)
    print("Creating visualization...")
    print("="*60)

    # Find Z slice that has this point
    z_seg = seg_voxel[2]
    z_ct = ct_voxel_int[2] if in_bounds else ct_shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CT slice
    ct_slice = ct_image[0, :, :, z_ct].numpy()
    axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
    if in_bounds:
        axes[0].plot(ct_voxel_int[0], ct_voxel_int[1], 'rx', markersize=15, markeredgewidth=2)
    axes[0].set_title(f'CT slice z={z_ct}')
    axes[0].axis('off')

    # SEG slice
    seg_slice = seg_np[:, :, z_seg]
    axes[1].imshow(seg_slice.T, cmap='nipy_spectral', origin='lower')
    axes[1].plot(seg_voxel[0], seg_voxel[1], 'wx', markersize=15, markeredgewidth=2)
    axes[1].set_title(f'SEG slice z={z_seg}')
    axes[1].axis('off')

    # Info text
    axes[2].text(0.1, 0.8, f"SEG voxel: {seg_voxel}", fontsize=12)
    axes[2].text(0.1, 0.6, f"World coords: [{seg_world[0]:.1f}, {seg_world[1]:.1f}, {seg_world[2]:.1f}]", fontsize=12)
    axes[2].text(0.1, 0.4, f"CT voxel: {ct_voxel_int}", fontsize=12)
    axes[2].text(0.1, 0.2, f"In bounds: {in_bounds}", fontsize=12)
    axes[2].axis('off')
    axes[2].set_title('Mapping Info')

    plt.tight_layout()
    plt.savefig(os.path.join(seg_dir, 'alignment_test.png'), dpi=100)
    print(f"Saved: {os.path.join(seg_dir, 'alignment_test.png')}")

print(f"\nData directory: {seg_dir}")
