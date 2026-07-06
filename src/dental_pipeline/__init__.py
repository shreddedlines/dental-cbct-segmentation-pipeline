"""
dental_pipeline — Production dental CBCT segmentation pipeline.

Built on nnU-Net v2 for the ToothFairy3 dataset (532 CBCT volumes, 77 classes).
"""

__version__ = "1.0.0"
__author__ = "Dobbe AI"

from dental_pipeline.config import (
    LABEL_CATEGORIES,
    PipelineConfig,
    get_label_names,
    load_config,
    setup_logging,
)
from dental_pipeline.dataset_validator import DatasetValidator
from dental_pipeline.label_remapping import LabelRemapper

__all__ = [
    "LABEL_CATEGORIES",
    "DatasetValidator",
    "LabelRemapper",
    "PipelineConfig",
    "get_label_names",
    "load_config",
    "setup_logging",
]
