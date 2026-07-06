"""
Configuration loader for the dental CBCT segmentation pipeline.

Loads a YAML config file into typed dataclasses, resolves relative paths to
absolute paths from the project root, and provides helpers for logging setup
and label-name retrieval.

Usage
-----
    from dental_pipeline.config import load_config

    config = load_config("configs/pipeline_config.yaml")
    print(config.paths.dataset_root)
    print(config.get_label_names())
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ============================================================================
# Dataclass hierarchy
# ============================================================================


@dataclass
class ProjectConfig:
    """Project-level metadata."""

    name: str = "dental-cbct-segmentation"
    description: str = ""
    seed: int = 42


@dataclass
class PathsConfig:
    """All filesystem paths used by the pipeline."""

    dataset_root: str = "/path/to/ToothFairy3"
    output_root: str = "./outputs"
    nnunet_raw: str = "./outputs/nnunet_raw"
    nnunet_preprocessed: str = "./outputs/nnunet_preprocessed"
    nnunet_results: str = "./outputs/nnunet_results"
    predictions: str = "./outputs/predictions"
    metrics: str = "./outputs/metrics"
    visualizations: str = "./outputs/visualizations"
    validation_report: str = "./outputs/validation_report"
    logs: str = "./outputs/logs"


@dataclass
class DatasetConfig:
    """Dataset-specific metadata matching dataset.json."""

    id: int = 100
    name: str = "Dataset100_ToothFairy3"
    num_classes: int = 77
    num_volumes: int = 532
    file_ending: str = ".nii.gz"
    channel_names: Dict[int, str] = field(default_factory=lambda: {0: "CBCT"})
    labels: Dict[int, str] = field(default_factory=dict)


@dataclass
class NNUNetConfig:
    """nnU-Net training and inference parameters."""

    configuration: str = "3d_fullres"
    trainer: str = "nnUNetTrainer"
    plans: str = "nnUNetPlans"
    folds: List[int] = field(default_factory=lambda: [0])
    num_processes: int = 8


@dataclass
class PostprocessingConfig:
    """Post-processing parameters for predicted segmentation masks."""

    min_component_size: int = 100
    apply_connected_components: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation / inference settings."""

    postprocessing: PostprocessingConfig = field(
        default_factory=PostprocessingConfig
    )
    test_time_augmentation: bool = False
    save_softmax: bool = False


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    epochs: int = 1000
    continue_training: bool = False
    compile: bool = False


@dataclass
class VisualizationConfig:
    """Visualization defaults."""

    num_cases: int = 3
    opacity: float = 0.3
    colormap: str = "tab20"
    dpi: int = 150
    slice_axis: str = "axial"
    save_format: str = "png"


@dataclass
class PipelineConfig:
    """Top-level container that aggregates every config section."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    nnunet: NNUNetConfig = field(default_factory=NNUNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(
        default_factory=VisualizationConfig
    )

    # Populated after resolve_paths
    _project_root: Optional[Path] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    def resolve_paths(self, project_root: Path) -> None:
        """Convert every relative path in ``PathsConfig`` to an absolute path
        anchored at *project_root*.  ``dataset_root`` is left untouched if it
        is already absolute.

        Parameters
        ----------
        project_root:
            The root directory of the project (typically where ``configs/``
            lives as a child).
        """
        self._project_root = project_root.resolve()
        path_fields = [
            "output_root",
            "nnunet_raw",
            "nnunet_preprocessed",
            "nnunet_results",
            "predictions",
            "metrics",
            "visualizations",
            "validation_report",
            "logs",
        ]
        for fname in path_fields:
            raw_value = getattr(self.paths, fname)
            resolved = self._resolve_single(raw_value)
            setattr(self.paths, fname, str(resolved))

        # dataset_root: resolve only if relative
        ds_root = Path(self.paths.dataset_root)
        if not ds_root.is_absolute():
            self.paths.dataset_root = str(
                (self._project_root / ds_root).resolve()
            )

    def _resolve_single(self, raw: str) -> Path:
        """Resolve a single path string relative to project root."""
        p = Path(raw)
        if p.is_absolute():
            return p
        assert self._project_root is not None
        return (self._project_root / p).resolve()

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------
    def get_label_names(self) -> Dict[int, str]:
        """Return ``{label_id: label_name}`` from the config.

        Falls back to reading ``dataset.json`` from *dataset_root* if the
        config labels dict is empty.
        """
        if self.dataset.labels:
            return dict(self.dataset.labels)

        dataset_json_path = Path(self.paths.dataset_root) / "dataset.json"
        if dataset_json_path.is_file():
            with open(dataset_json_path, "r", encoding="utf-8") as fh:
                dj = json.load(fh)
            raw_labels: Dict[str, str] = dj.get("labels", {})
            return {int(k): v for k, v in raw_labels.items()}

        logger.warning(
            "No label mapping found in config or dataset.json. "
            "Returning empty dict."
        )
        return {}

    def get_valid_label_ids(self) -> List[int]:
        """Sorted list of all valid (expected) label IDs."""
        return sorted(self.get_label_names().keys())

    # ------------------------------------------------------------------
    # nnU-Net environment variables
    # ------------------------------------------------------------------
    def set_nnunet_env(self) -> None:
        """Export the three nnU-Net path env-vars so CLI tools find them."""
        os.environ["nnUNet_raw"] = self.paths.nnunet_raw
        os.environ["nnUNet_preprocessed"] = self.paths.nnunet_preprocessed
        os.environ["nnUNet_results"] = self.paths.nnunet_results
        logger.info("nnU-Net environment variables set.")
        logger.debug("  nnUNet_raw            = %s", self.paths.nnunet_raw)
        logger.debug(
            "  nnUNet_preprocessed   = %s", self.paths.nnunet_preprocessed
        )
        logger.debug("  nnUNet_results        = %s", self.paths.nnunet_results)

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------
    def create_output_dirs(self) -> None:
        """Create every output directory listed in ``PathsConfig``."""
        for fname in (
            "output_root",
            "nnunet_raw",
            "nnunet_preprocessed",
            "nnunet_results",
            "predictions",
            "metrics",
            "visualizations",
            "validation_report",
            "logs",
        ):
            p = Path(getattr(self.paths, fname))
            p.mkdir(parents=True, exist_ok=True)
        logger.info("All output directories ensured.")


# ============================================================================
# YAML → dataclass deserialization helpers
# ============================================================================


def _build_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a plain dict.

    Handles nested dataclasses and basic type coercions (e.g. YAML reads
    dict keys as int when they look numeric, but our dataclass might
    expect ``Dict[int, str]``).
    """
    if data is None:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs: Dict[str, Any] = {}

    for key, value in data.items():
        if key.startswith("_"):
            continue  # skip private fields
        if key not in field_types:
            logger.debug("Ignoring unknown config key '%s' for %s", key, cls.__name__)
            continue

        ft = field_types[key]

        # Resolve string annotations → actual types
        if isinstance(ft, str):
            ft = _resolve_annotation(ft)

        # Nested dataclass
        if isinstance(ft, type) and hasattr(ft, "__dataclass_fields__"):
            kwargs[key] = _build_dataclass(ft, value or {})
        else:
            kwargs[key] = value

    return cls(**kwargs)


def _resolve_annotation(annotation: str) -> type:
    """Best-effort resolution of stringified type annotations."""
    _mapping = {
        "PostprocessingConfig": PostprocessingConfig,
        "ProjectConfig": ProjectConfig,
        "PathsConfig": PathsConfig,
        "DatasetConfig": DatasetConfig,
        "NNUNetConfig": NNUNetConfig,
        "TrainingConfig": TrainingConfig,
        "EvaluationConfig": EvaluationConfig,
        "VisualizationConfig": VisualizationConfig,
    }
    for name, cls in _mapping.items():
        if name in annotation:
            return cls
    return str  # fallback


# ============================================================================
# Public API
# ============================================================================


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file and return a fully-resolved
    ``PipelineConfig`` instance.

    Parameters
    ----------
    config_path:
        Path to the YAML config file (e.g. ``configs/pipeline_config.yaml``).

    Returns
    -------
    PipelineConfig
        Populated and path-resolved configuration object.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    config = _build_dataclass(PipelineConfig, raw)
    assert isinstance(config, PipelineConfig)

    # Resolve paths relative to the project root (parent of configs/)
    project_root = config_path.parent.parent
    config.resolve_paths(project_root)

    logger.info("Configuration loaded from %s", config_path)
    return config


# ============================================================================
# Logging setup
# ============================================================================


def setup_logging(
    config: PipelineConfig,
    level: int = logging.INFO,
    log_filename: str = "pipeline.log",
) -> None:
    """Configure the root logger to write to both console and a log file.

    Parameters
    ----------
    config:
        Pipeline config (used to locate the logs directory).
    level:
        Logging level for the root logger.
    log_filename:
        Name of the log file inside the logs directory.
    """
    log_dir = Path(config.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if root.handlers:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logger.info("Logging initialised — file: %s", log_file)


# ============================================================================
# Module-level constants and helpers
# ============================================================================

LABEL_CATEGORIES: Dict[str, List[int]] = {
    "Anatomical": [1, 2, 3, 4, 5, 6, 7],
    "Restorations": [8, 9, 10],
    "Teeth": [
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    ],
    "Canals": [103, 104, 105],
    "Pulps": [
        111, 112, 113, 114, 115, 116, 117, 118,
        121, 122, 123, 124, 125, 126, 127, 128,
        131, 132, 133, 134, 135, 136, 137, 138,
        141, 142, 143, 144, 145, 146, 147, 148,
    ],
}

# Module-level default label names (used when no config is available).
_DEFAULT_LABEL_NAMES: Dict[int, str] = {
    0: "background",
    1: "Lower Jawbone", 2: "Upper Jawbone",
    3: "Left Inferior Alveolar Canal", 4: "Right Inferior Alveolar Canal",
    5: "Left Maxillary Sinus", 6: "Right Maxillary Sinus", 7: "Pharynx",
    8: "Bridge", 9: "Crown", 10: "Implant",
    11: "Upper Right Central Incisor", 12: "Upper Right Lateral Incisor",
    13: "Upper Right Canine", 14: "Upper Right First Premolar",
    15: "Upper Right Second Premolar", 16: "Upper Right First Molar",
    17: "Upper Right Second Molar", 18: "Upper Right Third Molar",
    21: "Upper Left Central Incisor", 22: "Upper Left Lateral Incisor",
    23: "Upper Left Canine", 24: "Upper Left First Premolar",
    25: "Upper Left Second Premolar", 26: "Upper Left First Molar",
    27: "Upper Left Second Molar", 28: "Upper Left Third Molar",
    31: "Lower Left Central Incisor", 32: "Lower Left Lateral Incisor",
    33: "Lower Left Canine", 34: "Lower Left First Premolar",
    35: "Lower Left Second Premolar", 36: "Lower Left First Molar",
    37: "Lower Left Second Molar", 38: "Lower Left Third Molar",
    41: "Lower Right Central Incisor", 42: "Lower Right Lateral Incisor",
    43: "Lower Right Canine", 44: "Lower Right First Premolar",
    45: "Lower Right Second Premolar", 46: "Lower Right First Molar",
    47: "Lower Right Second Molar", 48: "Lower Right Third Molar",
    103: "Left Mandibular Incisive Canal", 104: "Right Mandibular Incisive Canal",
    105: "Lingual Canal",
    111: "Upper Right Central Incisor Pulp", 112: "Upper Right Lateral Incisor Pulp",
    113: "Upper Right Canine Pulp", 114: "Upper Right First Premolar Pulp",
    115: "Upper Right Second Premolar Pulp", 116: "Upper Right First Molar Pulp",
    117: "Upper Right Second Molar Pulp", 118: "Upper Right Third Molar Pulp",
    121: "Upper Left Central Incisor Pulp", 122: "Upper Left Lateral Incisor Pulp",
    123: "Upper Left Canine Pulp", 124: "Upper Left First Premolar Pulp",
    125: "Upper Left Second Premolar Pulp", 126: "Upper Left First Molar Pulp",
    127: "Upper Left Second Molar Pulp", 128: "Upper Left Third Molar Pulp",
    131: "Lower Left Central Incisor Pulp", 132: "Lower Left Lateral Incisor Pulp",
    133: "Lower Left Canine Pulp", 134: "Lower Left First Premolar Pulp",
    135: "Lower Left Second Premolar Pulp", 136: "Lower Left First Molar Pulp",
    137: "Lower Left Second Molar Pulp", 138: "Lower Left Third Molar Pulp",
    141: "Lower Right Central Incisor Pulp", 142: "Lower Right Lateral Incisor Pulp",
    143: "Lower Right Canine Pulp", 144: "Lower Right First Premolar Pulp",
    145: "Lower Right Second Premolar Pulp", 146: "Lower Right First Molar Pulp",
    147: "Lower Right Second Molar Pulp", 148: "Lower Right Third Molar Pulp",
}


def get_label_names() -> Dict[int, str]:
    """Return the default ``{label_id: label_name}`` mapping.

    This module-level helper is useful when callers need label names
    without access to a ``PipelineConfig`` instance.
    """
    return dict(_DEFAULT_LABEL_NAMES)
