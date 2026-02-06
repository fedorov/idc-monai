# IDC-MONAI: Using Imaging Data Commons with MONAI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ImagingDataCommons/idc-monai/blob/main/idc_monai/monai_contribution/idc_dataset.ipynb)

A comprehensive guide and toolkit for using [NCI Imaging Data Commons (IDC)](https://portal.imaging.datacommons.cancer.gov/) data with [MONAI](https://monai.io/) for medical imaging AI research.

## Overview

This project helps MONAI users leverage the vast public cancer imaging datasets available through the Imaging Data Commons. IDC provides free access to ~100 TB of radiology and pathology images without authentication, making it an excellent resource for training and evaluating medical imaging AI models.

### What You'll Learn

- Query and discover relevant imaging data from IDC
- Download DICOM data efficiently for AI training
- Load IDC data into MONAI pipelines with proper preprocessing
- Load DICOM Segmentation (DICOM-SEG) files with proper spatial alignment
- Extract segment metadata (names, categories, colors) from DICOM SEG
- Build end-to-end training workflows using public cancer imaging data

## Project Structure

```
idc_monai/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── src/
│   └── idc_monai/
│       ├── __init__.py
│       ├── dataset.py        # IDCDataset class for MONAI
│       ├── transforms.py     # IDC-specific transforms (LoadDicomSegd, CTWindowd)
│       └── utils.py          # Utility functions
├── tutorials/
│   ├── 01_getting_started.ipynb      # Introduction to IDC + MONAI
│   ├── 02_ct_segmentation.ipynb      # CT segmentation workflow
│   └── 03_working_with_annotations.ipynb  # Using IDC annotations
├── monai_contribution/
│   └── idc_dataset.ipynb     # Self-contained notebook for MONAI tutorials
└── dev/
    └── test_transform.py     # Development/testing scripts
```

## Learning Path

### Beginner Track
1. **[Getting Started](tutorials/01_getting_started.ipynb)** - Learn to query IDC, download data, and load it into MONAI
2. **[CT Segmentation](tutorials/02_ct_segmentation.ipynb)** - Build a complete segmentation pipeline with IDC data

### Intermediate Track
3. **[Working with Annotations](tutorials/03_working_with_annotations.ipynb)** - Use AI-generated and expert annotations from IDC

### Self-Contained Tutorial
- **[IDC Dataset Notebook](monai_contribution/idc_dataset.ipynb)** - Complete standalone tutorial (opens in Colab)

## Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/ImagingDataCommons/idc-monai.git
cd idc-monai

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Minimal Example

```python
from idc_index import IDCClient
from idc_monai import IDCDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd

# 1. Query IDC for chest CT data
client = IDCClient()
series = client.sql_query("""
    SELECT SeriesInstanceUID, PatientID, collection_id
    FROM index
    WHERE Modality = 'CT'
      AND BodyPartExamined = 'CHEST'
      AND collection_id = 'nlst'
    LIMIT 10
""")

# 2. Create MONAI-compatible dataset
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0)),
])

dataset = IDCDataset(
    series_uids=list(series['SeriesInstanceUID']),
    download_dir="./data",
    transform=transforms,
)

# 3. Use with MONAI DataLoader
from monai.data import DataLoader
loader = DataLoader(dataset, batch_size=2, num_workers=2)
```

## Key Features

### IDCDataset Class

A MONAI-compatible dataset class that handles:
- Automatic download of DICOM data from IDC
- Caching to avoid re-downloading
- Integration with MONAI's transform pipeline
- Support for images with paired segmentations

### LoadDicomSegd Transform

A specialized MONAI transform for loading DICOM Segmentation files:
- Uses `itkwasm-dicom` for robust DICOM-SEG reading
- Produces affines compatible with MONAI's ITKReader (spatial alignment with CT)
- Extracts segment metadata including:
  - Segment names and labels
  - Category/type codes (SNOMED-CT, etc.)
  - Recommended display colors (`recommendedDisplayRGBValue`)
- Enables direct overlay visualization without manual reorientation

### CTWindowd Transform

Apply CT windowing with preset windows:
- Lung, soft tissue, bone, brain, liver presets
- Custom window center/width support

### Data Discovery Utilities

Helper functions to find:
- Collections by cancer type, modality, or body part
- Paired image-segmentation data via `seg_index`
- Data with specific licenses (CC-BY for commercial use)

## IDC Data Highlights for MONAI Users

| Collection | Modality | Use Case | Size |
|------------|----------|----------|------|
| NLST | CT | Lung cancer screening | ~26K patients |
| TCGA-LUAD | CT | Lung adenocarcinoma | ~500 patients |
| LIDC-IDRI | CT | Lung nodule detection | 1,018 patients |
| BraTS collections | MRI | Brain tumor segmentation | Varies |
| ACRIN-NSCLC-FDG-PET | PET/CT | Lung cancer staging | 242 patients |

**Analysis Results (AI Segmentations):**
- TotalSegmentator - 104 anatomical structures
- nnU-Net trained models
- Expert annotations from research studies

## Requirements

- Python 3.9+
- MONAI 1.0+
- idc-index 0.8+
- PyTorch 1.9+
- itkwasm-dicom (for DICOM-SEG support)
- pydicom
- ITK

## License

This project is licensed under the Apache License 2.0.

**Important:** IDC data has various licenses (CC-BY, CC-BY-NC). Always check `license_short_name` before using data for commercial purposes.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{idc_monai,
  title = {IDC-MONAI: Using Imaging Data Commons with MONAI},
  year = {2026},
  url = {https://github.com/ImagingDataCommons/idc-monai}
}
```

Also cite IDC:
```bibtex
@article{fedorov2023idc,
  title={National Cancer Institute Imaging Data Commons: Toward Transparency,
         Reproducibility, and Scalability in Imaging Artificial Intelligence},
  author={Fedorov, Andrey and others},
  journal={RadioGraphics},
  volume={43},
  number={12},
  year={2023},
  doi={10.1148/rg.230180}
}
```

## Resources

- [IDC Portal](https://portal.imaging.datacommons.cancer.gov/)
- [IDC Documentation](https://learn.canceridc.dev/)
- [MONAI Documentation](https://docs.monai.io/)
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [IDC User Forum](https://discourse.canceridc.dev/)
- [DICOM SEG Standard](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html)
- [ITKWasm DICOM](https://wasm.itk.org/en/latest/introduction/file_formats/dicom.html)
