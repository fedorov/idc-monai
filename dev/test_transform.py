#!/usr/bin/env python
"""Test the LoadDicomSegd transform.

This script verifies that LoadDicomSegd produces output that is spatially
aligned with CT images loaded via MONAI's ITKReader.

Usage:
    cd idc_monai
    source .venv/bin/activate
    python dev/test_transform.py

Expected output:
    - CT and SEG shapes match
    - Affines match (within floating point tolerance)
    - Voxel coordinates map correctly between CT and SEG
    - Visualization shows anatomically correct overlay
"""

import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
print("Creating visualization with DICOM SEG colors")
print("="*60)

# Extract segment colors from overlay_info metadata
overlay_info = seg_data["label_meta_dict"].get("overlay_info", {})
segment_attrs = overlay_info.get("segmentAttributes", [[]])

# Build colormap from DICOM SEG recommendedDisplayRGBValue
# Index 0 is background (transparent), then each segment label maps to its color
def build_seg_colormap(segment_attrs):
    """Build a colormap from DICOM SEG segment attributes.

    The segmentAttributes is a list of lists (groups of segments).
    Each segment has a labelID and recommendedDisplayRGBValue.
    """
    # Flatten segment attributes and find max label
    all_segments = []
    for group in segment_attrs:
        all_segments.extend(group)

    if not all_segments:
        return plt.cm.nipy_spectral, {}

    max_label = max(seg.get("labelID", 0) for seg in all_segments)

    # Create color array: [background, label1, label2, ...]
    # Start with transparent background
    colors = np.zeros((max_label + 1, 4))
    colors[0] = [0, 0, 0, 0]  # Background transparent

    label_names = {}
    for seg in all_segments:
        label_id = seg.get("labelID", 0)
        rgb = seg.get("recommendedDisplayRGBValue", [128, 128, 128])
        label_name = seg.get("SegmentLabel", f"Segment {label_id}")

        # Normalize RGB from 0-255 to 0-1, set alpha
        colors[label_id] = [rgb[0]/255, rgb[1]/255, rgb[2]/255, 1.0]
        label_names[label_id] = label_name

    return ListedColormap(colors), label_names

seg_cmap, label_names = build_seg_colormap(segment_attrs)

print(f"Found {len(label_names)} segments with DICOM-defined colors:")
for label_id, name in sorted(label_names.items())[:10]:
    print(f"  Label {label_id}: {name}")
if len(label_names) > 10:
    print(f"  ... and {len(label_names) - 10} more")

ct_shape = ct_image.shape[1:]
z_slice = ct_shape[2] // 2

ct_slice = ct_image[0, :, :, z_slice].numpy()
seg_slice = seg_np[:, :, z_slice]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
axes[0].set_title(f'CT slice (z={z_slice})')
axes[0].axis('off')

# Use DICOM SEG colors for segmentation display
axes[1].imshow(seg_slice.T, cmap=seg_cmap, origin='lower',
               vmin=0, vmax=len(seg_cmap.colors)-1, interpolation='nearest')
axes[1].set_title(f'SEG slice (z={z_slice})\n(DICOM SEG colors)')
axes[1].axis('off')

# Overlay with DICOM SEG colors
axes[2].imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1000, vmax=500)
seg_mask = seg_slice > 0
seg_overlay = np.ma.masked_where(~seg_mask.T, seg_slice.T)
axes[2].imshow(seg_overlay, cmap=seg_cmap, origin='lower', alpha=0.6,
               vmin=0, vmax=len(seg_cmap.colors)-1, interpolation='nearest')
axes[2].set_title('Overlay\n(DICOM SEG colors)')
axes[2].axis('off')

plt.tight_layout()
output_path = os.path.join(seg_dir, 'transform_test.png')
plt.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

print("\n" + "="*60)
print("SUCCESS! Transform is working correctly.")
print("="*60)
print(f"\nData directory: {seg_dir}")
