"""Package installer for dental_pipeline."""

from setuptools import find_packages, setup

setup(
    name="dental-cbct-segmentation",
    version="1.0.0",
    description="End-to-end dental CBCT segmentation pipeline using nnU-Net v2",
    author="Dobbe AI",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0",
        "nnunetv2>=2.5",
        "nibabel>=5.0",
        "SimpleITK>=2.3",
        "numpy>=1.24",
        "scipy>=1.11",
        "scikit-image>=0.21",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "plotly>=5.18",
        "PyYAML>=6.0",
        "tqdm>=4.65",
        "rich>=13.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "dental-validate=scripts.validate_dataset:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
