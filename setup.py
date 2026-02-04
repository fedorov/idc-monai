# Copyright 2026 Imaging Data Commons
# Licensed under the Apache License, Version 2.0

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="idc-monai",
    version="0.1.0",
    author="Imaging Data Commons",
    author_email="support@canceridc.dev",
    description="Tools for using NCI Imaging Data Commons with MONAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ImagingDataCommons/idc-monai",
    project_urls={
        "Bug Tracker": "https://github.com/ImagingDataCommons/idc-monai/issues",
        "Documentation": "https://github.com/ImagingDataCommons/idc-monai",
        "IDC Portal": "https://portal.imaging.datacommons.cancer.gov/",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "monai>=1.0.0",
        "idc-index>=0.8.0",
        "torch>=1.9.0",
        "pydicom>=2.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "tutorials": [
            "matplotlib>=3.4.0",
            "jupyterlab",
            "SimpleITK>=2.1.0",
        ],
    },
)
