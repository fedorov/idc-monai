# Plan: Fix LoadDicomSegd — Remove Axis Flips, Use Proper Affine from DICOM Metadata

## Context

The current `LoadDicomSegd._build_affine` implementation is architecturally wrong. It hardcodes the output affine diagonal to `[-spacing[0], -spacing[1], spacing[2]]` and then physically reorders (flips) the SEG array to "match MONAI ITKReader convention." This approach bakes in an assumption about data ordering that violates the principle that orientation is fully defined in DICOM metadata. The correct behavior is to load the SEG array as-is from itkwasm (after the required `(Z,Y,X)→(X,Y,Z)` transpose, which is a layout convention not an orientation flip) and derive the affine directly from the direction matrix, spacing, and origin that itkwasm returns.

**itkwasm-dicom conventions (confirmed from source):**
- `seg_image.data` shape is `(Z, Y, X)` — C-order numpy with Z slowest
- `seg_image.size` is `(X, Y, Z)` — ITK convention
- `seg_image.origin`, `seg_image.spacing`, `seg_image.direction` are all in **LPS** coordinate system
- `seg_image.direction` is a flattened 9-element array; `reshape(3,3)` gives a matrix D where `D[i,j]` = component of voxel-axis-j's unit vector along physical LPS axis i
- Affine formula (ITK standard): `world_lps = D @ diag(spacing) @ voxel + origin`

**MONAI convention (from empirical ITKReader analysis):**
- Applies LPS-to-RAS: negate X and Y world coordinates
- Affine becomes: `world_ras = diag([-1,-1,1]) @ (D @ diag(spacing) @ voxel + origin)`

## Critical Files

- `src/idc_monai/transforms.py` — `_build_affine` (lines 204–259) and `__call__` (lines 261–303): the only file to modify
- `dev/test_transform.py` — existing functional test to adapt
- `docs/dicom_seg_loading.md` — documentation to update

## Implementation

### Step 1: Replace `_build_affine` in `src/idc_monai/transforms.py`

Replace the entire `_build_affine` method (lines 204–259) with:

```python
def _build_affine(self, spacing, origin, direction) -> np.ndarray:
    """Build 4x4 affine matrix from DICOM spatial metadata.

    Converts from ITK/DICOM LPS convention to MONAI's RAS-like convention
    (negating X and Y world coordinates) to match the output of ITKReader.
    No axis flips are applied — orientation is fully encoded in the affine.

    Args:
        spacing:   Voxel spacing (X, Y, Z) from itkwasm
        origin:    Physical coordinates of voxel [0,0,0] in LPS from itkwasm
        direction: 3x3 direction cosine matrix (D[i,j] = component of
                   voxel-axis-j unit vector along LPS physical axis i)

    Returns:
        4x4 affine matrix in MONAI convention
    """
    lps_to_ras = np.diag([-1., -1., 1.])
    affine = np.eye(4)
    affine[:3, :3] = lps_to_ras @ direction @ np.diag(spacing)
    affine[:3, 3] = lps_to_ras @ origin
    return affine
```

### Step 2: Update `__call__` in the same file

Remove the flip loop and the `flip_axes` return value. The updated call site becomes:

```python
affine = self._build_affine(spacing, origin, direction)

# No axis flips — orientation is encoded in the affine
seg_array = np.ascontiguousarray(seg_array)
```

The `np.ascontiguousarray` call is retained because the array may not be contiguous after the transpose. Remove the `for axis in flip_axes` block entirely.

### Step 3: Update the docstring on `_build_affine`

The signature also changes (remove `size` parameter, no longer returns a tuple).

## Verification

### Environment setup

```bash
cd /Users/af61/github/idc-monai
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Mathematical check (no network needed)

For an axis-aligned SEG with `direction = [[1,0,0],[0,-1,0],[0,0,-1]]`, `spacing = [0.7, 0.7, 5]`, `origin = [-179.6, 18.6, -83]`:

Expected affine:
```
[[-0.7,  0,   0,   179.6 ],
 [ 0,   0.7,  0,  -18.6  ],
 [ 0,   0,  -5,  -83.0   ],
 [ 0,   0,   0,   1      ]]
```

Voxel [0, 511, 49] should map to world ≈ [179.6, 339.1, -328], which is the same physical location as CT voxel [0, 0, 0] (within ~1 mm, due to independent SEG slice positioning). Verify with:
```python
seg_affine @ [0, 511, 49, 1]   # ≈ [179.6, 339.1, -328, 1]
```

For identity-direction SEG (`direction = I`), `origin = [-179.6, -340.6, -328]`:
```
[[-0.7,  0,  0,  179.6 ],
 [ 0,  -0.7, 0,  340.6 ],
 [ 0,   0,   5, -328.0 ],
 [ 0,   0,   0,   1    ]]
```
This matches the ITKReader output for a CT with the same geometry exactly. ✓

### Real data test

```bash
python dev/test_transform.py
```

The test downloads a TotalSegmentator CT+SEG pair from IDC, loads both, and checks that SEG voxels map to valid CT locations via the affines. After the fix:
- CT and SEG affines will **not** be identical (because their direction matrices differ)
- The existing labeled-voxel round-trip check (SEG voxel → world → CT voxel → in-bounds) is the correct verification
- Remove or relax any assertion that checks CT affine == SEG affine

### Confirming no-flip behavior

After the fix, `seg_array` should equal `np.transpose(np.asarray(seg_image.data), (2,1,0))` with no further reordering:
```python
raw_transposed = np.transpose(np.asarray(seg_image.data), (2, 1, 0))
assert np.array_equal(seg_label[0].numpy(), raw_transposed)
```

## What changes downstream

Users who relied on the axis-flip behavior to get matching affines between CT and SEG need to use `Orientationd` for voxel-aligned operations. `docs/dicom_seg_loading.md` already mentions `Orientationd` as the recommended approach; update the "Coordinate System Alignment" section to explain that the affine now faithfully represents the DICOM metadata rather than being forced to match the CT's orientation.
