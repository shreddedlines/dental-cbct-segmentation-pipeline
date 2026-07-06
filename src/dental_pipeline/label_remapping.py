"""
Label remapping for nnU-Net v2 compatibility.

nnU-Net v2 requires labels to be consecutive integers starting from 0.
The ToothFairy3 dataset uses non-contiguous FDI-based label IDs
(0, 1–10, 11–18, 21–28, 31–38, 41–48, 103–105, 111–148).

This module:
  1. Builds a bijective mapping  original_id ↔ consecutive_id.
  2. Remaps all label masks in ``labelsTr/`` to consecutive IDs (writing
     to the nnU-Net raw directory — the source dataset is never modified).
  3. Generates a new ``dataset.json`` with the consecutive label scheme.
  4. Persists the mapping as ``label_mapping.json`` so that predictions can
     be reverse-mapped back to original IDs for evaluation and reporting.

Usage
-----
    from dental_pipeline.label_remapping import LabelRemapper
    from dental_pipeline.config import load_config

    config = load_config("configs/pipeline_config.yaml")
    remapper = LabelRemapper(config)
    remapper.run()                       # remap everything
    original_seg = remapper.reverse_map_volume(consecutive_seg)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from dental_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-function helpers (top-level so they are picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _remap_single_file(
    src_path: str,
    dst_path: str,
    lut: np.ndarray,
    max_original_id: int,
    unknown_voxel_threshold: int = 1000,
) -> Tuple[str, bool, str]:
    """Remap a single NIfTI label file using a look-up table.

    Parameters
    ----------
    src_path : str
        Source label file (original IDs).
    dst_path : str
        Destination file (consecutive IDs).
    lut : np.ndarray
        1-D array of length ``max_original_id + 1`` where
        ``lut[original_id] = consecutive_id``.
    max_original_id : int
        Maximum original label ID (used for bounds checking).
    unknown_voxel_threshold : int
        If the total number of voxels carrying unknown label IDs is
        **below** this threshold they are silently mapped to background
        (0) and a warning is returned.  If the count meets or exceeds
        the threshold the file is rejected as an error.

    Returns
    -------
    tuple[str, bool, str]
        ``(filename, success, message)``
    """
    fname = Path(src_path).name
    try:
        img = nib.load(src_path)
        data = np.asarray(img.dataobj, dtype=np.int32)

        # Identify voxels whose label ID falls outside the LUT range or
        # is not part of the known label set (LUT maps unknowns to 0 by
        # construction, but IDs *beyond* the LUT length would cause an
        # IndexError).
        out_of_range_mask = data > max_original_id
        unknown_count = int(out_of_range_mask.sum())

        if unknown_count > 0:
            bad_ids = sorted(set(data[out_of_range_mask].flat))

            if unknown_count >= unknown_voxel_threshold:
                return (
                    fname,
                    False,
                    f"Label IDs out of range: {bad_ids[:10]} "
                    f"({unknown_count} voxels — exceeds threshold "
                    f"of {unknown_voxel_threshold})",
                )

            # Below threshold → clamp unknown voxels to background
            data[out_of_range_mask] = 0
            warning_msg = (
                f"WARNING: mapped {unknown_count} voxels with unknown "
                f"label(s) {bad_ids} to background (below threshold "
                f"of {unknown_voxel_threshold})"
            )
        else:
            warning_msg = None

        # Apply look-up table (vectorised, fast)
        remapped = lut[data]

        # Save with same affine and header
        new_img = nib.Nifti1Image(
            remapped.astype(np.uint8) if lut.max() < 256 else remapped.astype(np.int16),
            affine=img.affine,
            header=img.header,
        )
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, dst_path)

        return (fname, True, warning_msg or "OK")
    except Exception as exc:
        return (fname, False, str(exc))


class LabelRemapper:
    """Bidirectional label remapping between original and consecutive IDs.

    Parameters
    ----------
    config : PipelineConfig
        Resolved pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dataset_src = Path(config.paths.dataset_root)
        self.dataset_dst = (
            Path(config.paths.nnunet_raw) / config.dataset.name
        )
        self.num_workers = config.nnunet.num_processes

        # Build mappings from the original dataset.json
        self.original_labels: Dict[int, str] = self._load_original_labels()
        (
            self.original_to_consecutive,
            self.consecutive_to_original,
            self.consecutive_label_names,
        ) = self._build_mapping()

        # Pre-compute a numpy LUT for fast vectorised remapping
        self._max_original_id = max(self.original_labels.keys())
        self._forward_lut = self._build_lut(
            self.original_to_consecutive, self._max_original_id
        )
        self._max_consecutive_id = max(self.consecutive_to_original.keys())
        self._reverse_lut = self._build_lut(
            self.consecutive_to_original, self._max_consecutive_id
        )

    # ------------------------------------------------------------------
    # Mapping construction
    # ------------------------------------------------------------------

    def _load_original_labels(self) -> Dict[int, str]:
        """Read the original ``dataset.json`` from the source dataset.

        The file uses ``{name: id}`` format.  We return ``{id: name}``.
        """
        dj_path = self.dataset_src / "dataset.json"
        if not dj_path.is_file():
            raise FileNotFoundError(
                f"dataset.json not found at {dj_path}"
            )
        with open(dj_path, "r", encoding="utf-8") as fh:
            dj = json.load(fh)

        raw_labels = dj.get("labels", {})
        # dataset.json format: {"name": id, ...} → we want {id: "name"}
        return {int(v): k for k, v in raw_labels.items()}

    def _build_mapping(
        self,
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
        """Build bijective original ↔ consecutive mappings.

        Returns
        -------
        tuple
            (original_to_consecutive, consecutive_to_original,
             consecutive_label_names)
        """
        sorted_ids = sorted(self.original_labels.keys())

        o2c: Dict[int, int] = {}
        c2o: Dict[int, int] = {}
        c_names: Dict[int, str] = {}

        for consecutive_id, original_id in enumerate(sorted_ids):
            o2c[original_id] = consecutive_id
            c2o[consecutive_id] = original_id
            c_names[consecutive_id] = self.original_labels[original_id]

        logger.info(
            "Label mapping built: %d classes, "
            "original range [%d, %d] → consecutive [0, %d]",
            len(sorted_ids),
            sorted_ids[0],
            sorted_ids[-1],
            len(sorted_ids) - 1,
        )
        return o2c, c2o, c_names

    @staticmethod
    def _build_lut(mapping: Dict[int, int], max_key: int) -> np.ndarray:
        """Build a numpy look-up table from a dict mapping."""
        lut = np.zeros(max_key + 1, dtype=np.int32)
        for src, dst in mapping.items():
            lut[src] = dst
        return lut

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Execute the full remapping pipeline.

        1. Remap all ``labelsTr/*.nii.gz`` → consecutive IDs.
        2. Generate a new ``dataset.json`` with consecutive labels.
        3. Save ``label_mapping.json`` for reverse mapping.

        Returns
        -------
        Path
            Path to the generated ``label_mapping.json``.
        """
        logger.info("=" * 60)
        logger.info("Starting label remapping (original → consecutive)")
        logger.info("=" * 60)

        self.remap_labels_directory()
        self.generate_dataset_json()
        mapping_path = self.save_mapping()

        logger.info("=" * 60)
        logger.info("Label remapping complete")
        logger.info("=" * 60)

        return mapping_path

    def remap_labels_directory(self) -> None:
        """Remap every label file from ``labelsTr/`` to consecutive IDs.

        Reads from the original (read-only) dataset and writes remapped
        files into the nnU-Net raw directory.
        """
        src_dir = self.dataset_src / "labelsTr"
        dst_dir = self.dataset_dst / "labelsTr"

        if not src_dir.is_dir():
            raise FileNotFoundError(f"Source labelsTr not found: {src_dir}")

        dst_dir.mkdir(parents=True, exist_ok=True)

        label_files = sorted(src_dir.glob("*.nii.gz"))
        if not label_files:
            raise FileNotFoundError(f"No .nii.gz files in {src_dir}")

        logger.info(
            "Remapping %d label files: %s → %s",
            len(label_files), src_dir, dst_dir,
        )

        # Parallel remapping
        tasks = [
            (str(f), str(dst_dir / f.name), self._forward_lut, self._max_original_id)
            for f in label_files
        ]

        successes = 0
        failures: List[Tuple[str, str]] = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_remap_single_file, *t): t[0]
                for t in tasks
            }
            with tqdm(total=len(futures), desc="Remapping labels", unit="vol") as pbar:
                for future in as_completed(futures):
                    fname, ok, msg = future.result()
                    if ok:
                        successes += 1
                        if msg != "OK":
                            logger.warning("  %s: %s", fname, msg)
                    else:
                        failures.append((fname, msg))
                        logger.error("  FAILED %s: %s", fname, msg)
                    pbar.update(1)

        logger.info(
            "Remapping finished: %d/%d succeeded, %d failed",
            successes, len(tasks), len(failures),
        )

        if failures:
            raise RuntimeError(
                f"{len(failures)} label file(s) failed remapping. "
                "Check the log for details."
            )

    def generate_dataset_json(self) -> Path:
        """Generate a new ``dataset.json`` with consecutive label IDs.

        This file is written to the nnU-Net raw dataset directory and
        replaces any symlinked ``dataset.json``.

        Returns
        -------
        Path
            Path to the generated file.
        """
        dj_path = self.dataset_dst / "dataset.json"

        # Read original for metadata
        with open(self.dataset_src / "dataset.json", "r", encoding="utf-8") as fh:
            original_dj = json.load(fh)

        # Build consecutive labels dict: {"name": consecutive_id, ...}
        consecutive_labels = {
            name: cid
            for cid, name in self.consecutive_label_names.items()
        }

        new_dj = {
            "name": original_dj.get("name", "ToothFairy 3"),
            "description": original_dj.get("description", ""),
            "reference": original_dj.get("reference", ""),
            "license": original_dj.get("license", ""),
            "release": original_dj.get("release", ""),
            "tensorImageSize": original_dj.get("tensorImageSize", "4D"),
            "labels": consecutive_labels,
            "numTraining": original_dj.get("numTraining", 532),
            "numTest": original_dj.get("numTest", 0),
            "file_ending": original_dj.get("file_ending", ".nii.gz"),
            "channel_names": original_dj.get("channel_names", {"0": "CBCT"}),
        }

        # Remove any existing symlink before writing the real file
        if dj_path.is_symlink():
            dj_path.unlink()
            logger.info("Removed existing dataset.json symlink")

        with open(dj_path, "w", encoding="utf-8") as fh:
            json.dump(new_dj, fh, indent=2)

        logger.info("Generated consecutive dataset.json → %s", dj_path)
        return dj_path

    def save_mapping(self) -> Path:
        """Persist the bidirectional label mapping to JSON.

        Saves to ``outputs/label_mapping.json`` and also to the nnU-Net
        raw dataset directory for co-location with the data.

        Returns
        -------
        Path
            Path to the primary mapping file.
        """
        mapping_data = {
            "description": (
                "Bidirectional mapping between original ToothFairy3 FDI label IDs "
                "and consecutive nnU-Net-compatible label IDs."
            ),
            "num_classes": len(self.original_to_consecutive),
            "original_to_consecutive": {
                str(k): v for k, v in sorted(self.original_to_consecutive.items())
            },
            "consecutive_to_original": {
                str(k): v for k, v in sorted(self.consecutive_to_original.items())
            },
            "consecutive_label_names": {
                str(k): v for k, v in sorted(self.consecutive_label_names.items())
            },
            "original_label_names": {
                str(k): v for k, v in sorted(self.original_labels.items())
            },
        }

        # Save to output root
        output_root = Path(self.config.paths.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        primary_path = output_root / "label_mapping.json"
        with open(primary_path, "w", encoding="utf-8") as fh:
            json.dump(mapping_data, fh, indent=2)
        logger.info("Label mapping saved → %s", primary_path)

        # Also save alongside the dataset
        secondary_path = self.dataset_dst / "label_mapping.json"
        with open(secondary_path, "w", encoding="utf-8") as fh:
            json.dump(mapping_data, fh, indent=2)

        return primary_path

    # ------------------------------------------------------------------
    # Reverse mapping (for inference results)
    # ------------------------------------------------------------------

    def reverse_map_volume(self, consecutive_seg: np.ndarray) -> np.ndarray:
        """Map a segmentation volume from consecutive IDs back to original IDs.

        Parameters
        ----------
        consecutive_seg : np.ndarray
            Segmentation array with consecutive label IDs (0, 1, 2, …, 76).

        Returns
        -------
        np.ndarray
            Segmentation array with original ToothFairy3 label IDs.
        """
        return self._reverse_lut[consecutive_seg]

    def reverse_map_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Reverse-map a single NIfTI prediction file.

        Parameters
        ----------
        input_path : Path
            Prediction with consecutive IDs.
        output_path : Path
            Where to save the file with original IDs.
        """
        img = nib.load(str(input_path))
        data = np.asarray(img.dataobj, dtype=np.int32)
        remapped = self.reverse_map_volume(data)

        new_img = nib.Nifti1Image(
            remapped.astype(np.int16),
            affine=img.affine,
            header=img.header,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, str(output_path))

    def reverse_map_directory(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> None:
        """Reverse-map all NIfTI files in a directory.

        Parameters
        ----------
        input_dir : Path
            Directory with consecutive-ID predictions.
        output_dir : Path
            Where to save original-ID predictions.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_files = sorted(Path(input_dir).glob("*.nii.gz"))

        if not pred_files:
            logger.warning("No .nii.gz files found in %s", input_dir)
            return

        logger.info(
            "Reverse-mapping %d files: %s → %s",
            len(pred_files), input_dir, output_dir,
        )

        for f in tqdm(pred_files, desc="Reverse-mapping", unit="vol"):
            self.reverse_map_file(f, output_dir / f.name)

        logger.info("Reverse-mapping complete.")

    # ------------------------------------------------------------------
    # Class method to load an existing mapping from disk
    # ------------------------------------------------------------------

    @classmethod
    def from_mapping_file(
        cls,
        mapping_path: Path,
        config: Optional[PipelineConfig] = None,
    ) -> "LabelRemapper":
        """Create a LabelRemapper from a saved ``label_mapping.json``.

        This is useful when you only need reverse-mapping (e.g. during
        evaluation) and the original dataset is not available.

        Parameters
        ----------
        mapping_path : Path
            Path to ``label_mapping.json``.
        config : PipelineConfig, optional
            If provided, used for process count etc.
        """
        with open(mapping_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        instance = object.__new__(cls)
        instance.config = config
        instance.original_to_consecutive = {
            int(k): int(v)
            for k, v in data["original_to_consecutive"].items()
        }
        instance.consecutive_to_original = {
            int(k): int(v)
            for k, v in data["consecutive_to_original"].items()
        }
        instance.consecutive_label_names = {
            int(k): v
            for k, v in data["consecutive_label_names"].items()
        }
        instance.original_labels = {
            int(k): v
            for k, v in data["original_label_names"].items()
        }
        instance.num_workers = config.nnunet.num_processes if config else 4

        max_orig = max(instance.original_to_consecutive.keys())
        instance._max_original_id = max_orig
        instance._forward_lut = cls._build_lut(
            instance.original_to_consecutive, max_orig
        )
        max_cons = max(instance.consecutive_to_original.keys())
        instance._max_consecutive_id = max_cons
        instance._reverse_lut = cls._build_lut(
            instance.consecutive_to_original, max_cons
        )

        instance.dataset_src = Path(config.paths.dataset_root) if config else Path(".")
        instance.dataset_dst = Path(".")

        logger.info(
            "LabelRemapper loaded from %s (%d classes)",
            mapping_path, len(instance.original_to_consecutive),
        )
        return instance
