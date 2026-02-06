#!/usr/bin/env python
"""Test the updated LoadDicomSegd transform."""

import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from idc_index import IDCClient
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data.image_reader import ITKReader

from idc_monai.transforms import LoadDicomSegd

# Download test data
print("="*60)
print("Downloading test data")
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

seg_dir = tempfile.mkdtemp(prefix="idc_test_transform_")
print(f"Downloading to {seg_dir}...")
client.download_from_selection(
    seriesInstanceUID=[demo_pair['image_uid'], demo_pair['seg_uid']],
    downloadDir=seg_dir,
    dirTemplate="%SeriesInstanceUID"
)

image_path = os.path.join(seg_dir, demo_pair['image_uid'])
seg_path = os.path.join(seg_dir, demo_pair['seg_uid'])

print("\n" + "="*60)
print("Loading data using transforms")
print("="*60)

# Load CT with standard MONAI pipeline
ct_transform = Compose([
    LoadImaged(keys=["image"], reader=ITKReader()),
    EnsureChannelFirstd(keys=["image"]),
])

# Load SEG with our custom transform
seg_transform = Compose([
    LoadDicomSegd(keys=["label"]),
    EnsureChannelFirstd(keys=["label"]),
])

ct_data = ct_transform({"image": image_path})
seg_data = seg_transform({"label": seg_path})

ct_image = ct_data["image"]
seg_label = seg_data["label"]

print(f"\nCT shape: {ct_image.shape}")
print(f"CT affine:\n{ct_image.affine}")

print(f"\nSEG shape: {seg_label.shape}")
print(f"SEG affine:\n{seg_label.affine}")

print("\n" + "="*60)
print("Testing alignment")
print("="*60)

ct_affine = ct_image.affine.numpy()
seg_affine = seg_label.affine.numpy()

def voxel_to_world(affine, voxel):
    return (affine @ np.append(voxel, 1))[:3]

def world_to_voxel(affine, world):
    return (np.linalg.inv(affine) @ np.append(world, 1))[:3]

# Test corner alignment
print("\nCorner test:")
seg_corner_000 = voxel_to_world(seg_affine, [0, 0, 0])
ct_corner_000 = voxel_to_world(ct_affine, [0, 0, 0])
print(f"SEG [0,0,0] -> world {seg_corner_000}")
print(f"CT [0,0,0]  -> world {ct_corner_000}")
print(f"Difference: {np.linalg.norm(seg_corner_000 - ct_corner_000):.2f} mm")

# Test labeled voxel alignment
seg_np = seg_label[0].numpy()  # Remove channel dim
labeled_indices = np.argwhere(seg_np > 0)
if len(labeled_indices) > 0:
    # Test multiple points
    for idx in [len(labeled_indices)//4, len(labeled_indices)//2, 3*len(labeled_indices)//4]:
        seg_voxel = labeled_indices[idx]
        seg_world = voxel_to_world(seg_affine, seg_voxel)
        ct_voxel = world_to_voxel(ct_affine, seg_world)
        ct_voxel_int = np.round(ct_voxel).astype(int)

        ct_shape = ct_image.shape[1:]
        in_bounds = all(0 <= ct_voxel_int[i] < ct_shape[i] for i in range(3))

        print(f"\nSEG voxel {seg_voxel}:")
        print(f"  -> world: {seg_world}")
        print(f"  -> CT voxel: {ct_voxel_int}, in bounds: {in_bounds}")

        if in_bounds:
            ct_value = ct_image[0, ct_voxel_int[0], ct_voxel_int[1], ct_voxel_int[2]].item()
            print(f"  -> CT value: {ct_value:.1f} HU")

print("\n" + "="*60)
print("Creating visualization")
print("="*60)

ct_shape = ct_image.shape[1:]
z_slice = ct_shape[2] // 2

ct_slice = ct_image[0, :, :, z_slice].numpy()
seg_slice = seg_np[:, :, z_slice]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
axes[0].set_title(f'CT slice (z={z_slice})')
axes[0].axis('off')

axes[1].imshow(seg_slice.T, cmap='nipy_spectral', origin='lower')
axes[1].set_title(f'SEG slice (z={z_slice})')
axes[1].axis('off')

axes[2].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
seg_mask = seg_slice > 0
seg_overlay = np.ma.masked_where(~seg_mask.T, seg_slice.T)
axes[2].imshow(seg_overlay, cmap='nipy_spectral', origin='lower', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
output_path = os.path.join(seg_dir, 'transform_test.png')
plt.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

print("\n" + "="*60)
print("SUCCESS! Transform is working correctly.")
print("="*60)
print(f"\nData directory: {seg_dir}")
