# DICOM-SEG Loading in IDC-MONAI

This document explains the design decisions and research behind DICOM-SEG loading support in IDC-MONAI.

## Background

DICOM Segmentation (DICOM-SEG, SOP Class `1.2.840.10008.5.1.4.1.1.66.4`) is an enhanced multiframe DICOM format for storing segmentation masks. Unlike regular DICOM series (one file per slice), DICOM-SEG stores all slices in a single file with rich segment metadata.

### Key Characteristics of DICOM-SEG

- **Enhanced multiframe object**: All frames/slices in a single `.dcm` file
- **Segment metadata**: Each segment has a label, description, algorithm info, and anatomical codes
- **Per-frame spatial information**: Origin, orientation stored per-frame in nested sequences
- **Multi-segment support**: Can contain overlapping or non-overlapping segments

## Why Standard Readers Don't Work

### ITKReader/SimpleITK

SimpleITK and MONAI's ITKReader cannot read DICOM-SEG files. These readers expect standard DICOM image series (multiple files, one per slice) and do not handle the enhanced multiframe format or the nested `PerFrameFunctionalGroupsSequence` that contains spatial metadata.

**Error you'll see**: Loading fails or produces incorrect/empty data.

### MONAI's PydicomReader

MONAI's `PydicomReader` has experimental DICOM-SEG support (see `monai/data/image_reader.py`), but it has been reported to be unreliable:

- **Affine matrix issues**: May fabricate or incorrectly compute affine matrices
- **Spatial metadata problems**: Per-frame position information may not be correctly extracted
- **Size mismatches**: Resulting segmentation volume may not match source image dimensions

## Evaluated Solutions

We evaluated four approaches for loading DICOM-SEG in Python:

### 1. pydicom-seg

**Repository**: https://github.com/razorx89/pydicom-seg

**Pros**:
- Pure Python implementation
- Designed specifically for DICOM-SEG
- Returns numpy arrays and SimpleITK images

**Cons**:
- Less actively maintained
- Separate from main DICOM tooling ecosystem
- SimpleITK dependency for image construction

**API**:
```python
import pydicom
from pydicom_seg import MultiClassReader

dcm = pydicom.dcmread('segmentation.dcm')
reader = MultiClassReader()
result = reader.read(dcm)

seg_array = result.data      # numpy array
sitk_image = result.image    # SimpleITK image
```

### 2. highdicom

**Repository**: https://github.com/ImagingDataCommons/highdicom

**Pros**:
- Comprehensive library from IDC team
- Better spatial metadata handling
- Used by MONAI Deploy for DICOM-SEG writing
- Active development

**Cons**:
- More complex API
- Focused on both reading and creation

**API**:
```python
import highdicom as hd

seg = hd.seg.segread('/path/to/seg.dcm')
vol = seg.get_volume(combine_segments=True)
seg_array = vol.array
affine = vol.affine
```

### 3. itkwasm-dicom (Selected)

**Repository**: https://github.com/InsightSoftwareConsortium/ITK-Wasm

**Pros**:
- Wraps dcmqi (the reference implementation for DICOM-SEG)
- WebAssembly-based, portable across platforms
- Actively maintained by Kitware/ITK team
- Returns ITK-compatible Image objects with proper spatial metadata
- Python-native (no subprocess calls needed)

**Cons**:
- WebAssembly runtime dependency
- Relatively new Python API

**API**:
```python
from itkwasm_dicom import read_segmentation
import numpy as np

# Read DICOM-SEG file
seg_image, overlay_info = read_segmentation("/path/to/seg.dcm")

# Access data
seg_array = np.asarray(seg_image.data)  # numpy array
spacing = seg_image.spacing              # (sx, sy, sz) tuple
origin = seg_image.origin                # (ox, oy, oz) tuple
direction = seg_image.direction          # flattened 3x3 direction cosines

# overlay_info contains segment labels and descriptions
```

### 4. dcmqi CLI

**Repository**: https://github.com/QIICR/dcmqi

**Pros**:
- Gold standard reference implementation
- Most reliable conversion

**Cons**:
- Requires external C++ binaries
- Not Python-native (subprocess calls needed)
- Platform-specific installation

**API**:
```bash
segimage2itkimage --inputDICOM seg.dcm --outputDirectory ./output
```

## Decision: Use itkwasm-dicom

We selected `itkwasm-dicom` for the following reasons:

1. **Reliability**: Wraps dcmqi, which is the reference implementation for DICOM-SEG created by the QIICR team. This is the same code used in 3D Slicer and other trusted medical imaging tools.

2. **Spatial metadata accuracy**: Correctly extracts origin, spacing, and direction from the complex nested DICOM sequences.

3. **ITK ecosystem**: Part of the ITK ecosystem, ensuring long-term maintenance and compatibility.

4. **Python-native**: No subprocess calls or external binary installation required. Uses WebAssembly for portability.

5. **MONAI compatibility**: Returns numpy arrays that can be easily wrapped as MONAI `MetaTensor` with proper affine matrices.

6. **Active development**: Supported by Kitware with regular releases (v7.6.4 as of November 2025).

## Implementation

### LoadDicomSegd Transform

We created a MONAI-compatible transform in `idc_monai.transforms.LoadDicomSegd`:

```python
from idc_monai.transforms import LoadDicomSegd
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd

transforms = Compose([
    # For CT images (regular DICOM series) - use standard LoadImaged
    LoadImaged(keys=["image"]),

    # For DICOM-SEG - use LoadDicomSegd
    LoadDicomSegd(keys=["label"]),

    EnsureChannelFirstd(keys=["image", "label"]),
    # ... rest of pipeline
])
```

### Key Features

1. **Directory handling**: Accepts either a directory path (finds `.dcm` file inside) or direct file path
2. **Array transposition**: ITKWasm returns arrays in (Z, Y, X) order but metadata in (X, Y, Z) order. We transpose to (X, Y, Z) to match ITKReader and MONAI conventions.
3. **ITKReader-compatible affine**: Builds proper 4x4 affine matrix that matches MONAI's ITKReader output format, ensuring spatial alignment between CT and SEG
4. **Automatic axis flipping**: Flips array axes as needed to match the coordinate transformation applied by ITKReader
5. **MetaTensor output**: Returns MONAI `MetaTensor` with affine and metadata attached
6. **Overlay info**: Preserves segment labels/descriptions in metadata

### Coordinate System Alignment

DICOM images are stored in LPS (Left-Posterior-Superior) coordinates, but MONAI's ITKReader applies a transformation that results in:
- Affine diagonal: `[-spacing_x, -spacing_y, +spacing_z]`
- Origin adjusted to match the transformed array

The `LoadDicomSegd` transform applies the same transformation to DICOM-SEG data, ensuring that:
- CT and SEG voxel coordinates map to the same world coordinates
- Overlay visualizations align correctly
- Spatial operations (resampling, registration) work as expected

### Usage Note

Since `LoadDicomSegd` produces affines compatible with ITKReader, you typically do NOT need `Orientationd` for basic alignment. Both CT (via ITKReader) and SEG (via LoadDicomSegd) will have matching affines:

```python
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from idc_monai.transforms import LoadDicomSegd

transforms = Compose([
    LoadImaged(keys=["image"]),  # Uses ITKReader by default
    LoadDicomSegd(keys=["label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    # CT and SEG are now aligned - same affine, same voxel grid
])
```

If you need a specific orientation (e.g., RAS for neuroimaging), add `Orientationd`:

```python
from monai.transforms import Compose, LoadImaged, Orientationd, EnsureChannelFirstd
from idc_monai.transforms import LoadDicomSegd

transforms = Compose([
    LoadImaged(keys=["image"]),
    LoadDicomSegd(keys=["label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),  # Reorient both to RAS
    # ... rest of pipeline
])
```

## Alternative: Using highdicom

If you encounter issues with itkwasm-dicom, highdicom is a good alternative:

```python
import highdicom as hd
import numpy as np
from monai.data import MetaTensor

# Read DICOM-SEG
seg = hd.seg.segread('/path/to/seg.dcm')
vol = seg.get_volume(combine_segments=True)

# Convert to MONAI MetaTensor
meta_tensor = MetaTensor(vol.array)
meta_tensor.affine = vol.affine
```

## References

- [DICOM-SEG Standard (Part 3, Section C.8.20)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html)
- [dcmqi Documentation](https://qiicr.gitbook.io/dcmqi-guide/)
- [ITKWasm DICOM Documentation](https://wasm.itk.org/en/latest/introduction/file_formats/dicom.html)
- [Kitware Blog: Reading DICOM in Python](https://www.kitware.com/reading-dicom-images-and-non-image-sop-classes-in-javascript-and-python/)
- [pydicom-seg Documentation](https://razorx89.github.io/pydicom-seg/)
- [highdicom SEG Documentation](https://highdicom.readthedocs.io/en/latest/seg.html)
