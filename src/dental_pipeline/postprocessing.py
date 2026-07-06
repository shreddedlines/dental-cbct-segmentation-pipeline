"""Post-processing module for dental CBCT segmentation predictions.

Applies connected-component filtering, anatomical constraint enforcement,
and batch processing of NIfTI segmentation volumes produced by nnU-Net.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label

from dental_pipeline.config import (
    LABEL_CATEGORIES,
    PipelineConfig,
    get_label_names,
)

logger = logging.getLogger(__name__)

# Tooth position label IDs where implants can legitimately replace a tooth.
# All 32 FDI tooth positions (11-18, 21-28, 31-38, 41-48).
_VALID_IMPLANT_TOOTH_POSITIONS: set[int] = set(LABEL_CATEGORIES["Teeth"])

# Canal ↔ tooth associations (canal label → set of tooth labels it belongs to).
_CANAL_TOOTH_MAP: dict[int, set[int]] = {
    103: {33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47, 48},  # L IAC region (lower left)
    104: {33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47, 48},  # R IAC region (lower right)
    105: {33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47, 48},  # generic canal
}


class PostProcessor:
    """Post-processing pipeline for dental CBCT segmentation masks.

    Applies per-label connected component filtering and anatomical
    constraint rules to clean up raw nnU-Net predictions.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration loaded from YAML.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.min_component_size: int = config.evaluation.postprocessing.min_component_size
        self.apply_cc: bool = config.evaluation.postprocessing.apply_connected_components
        self.label_names: dict[int, str] = get_label_names()
        logger.info(
            "PostProcessor initialised – min_component_size=%d, apply_cc=%s",
            self.min_component_size,
            self.apply_cc,
        )

    # ------------------------------------------------------------------
    # Core algorithms
    # ------------------------------------------------------------------

    def remove_small_components(
        self,
        segmentation: np.ndarray,
        min_size: int | None = None,
    ) -> np.ndarray:
        """Remove small connected components from each label independently.

        For every non-background label present in *segmentation*, a 3-D
        connected-component analysis is run.  Components whose voxel
        count is strictly below *min_size* are set to background (0).

        Parameters
        ----------
        segmentation : np.ndarray
            Integer label volume (H, W, D) or (D, H, W).
        min_size : int, optional
            Minimum component size in voxels.  Falls back to
            ``config.evaluation.postprocessing.min_component_size``.

        Returns
        -------
        np.ndarray
            Cleaned segmentation with the same shape and dtype.
        """
        if min_size is None:
            min_size = self.min_component_size

        cleaned = segmentation.copy()
        unique_labels = np.unique(cleaned)
        # Skip background (0)
        unique_labels = unique_labels[unique_labels != 0]

        removed_total = 0
        for lbl in unique_labels:
            binary_mask = cleaned == lbl
            component_array, num_components = ndimage_label(binary_mask)

            if num_components <= 1:
                # 0 or 1 component – nothing to prune
                continue

            for comp_id in range(1, num_components + 1):
                comp_mask = component_array == comp_id
                comp_size = int(comp_mask.sum())
                if comp_size < min_size:
                    cleaned[comp_mask] = 0
                    removed_total += 1

        if removed_total > 0:
            logger.info(
                "Removed %d small components (min_size=%d).",
                removed_total,
                min_size,
            )
        return cleaned

    def apply_anatomical_constraints(
        self,
        segmentation: np.ndarray,
    ) -> np.ndarray:
        """Enforce anatomy-aware rules on the segmentation.

        Currently implemented rules
        ---------------------------
        1. **Implant position validity** – Implant voxels (label 10) that
           do not spatially overlap with any valid tooth-position label
           in the *original* prediction are removed.  This guards against
           hallucinated implant predictions in impossible locations (e.g.,
           in the maxillary sinus or pharynx).

        2. **Bilateral structure symmetry check** – If one of the paired
           anatomical structures (L/R IAC, L/R Max Sinus) is present but
           the contralateral is suspiciously tiny (< 5 % of the larger
           side), a warning is logged.  No voxels are removed — this is
           an advisory check.

        3. **Pulp inside tooth** – Pulp labels (111-148) must be
           spatially contained within the corresponding tooth label
           (11-48).  Pulp voxels outside their parent tooth are removed.

        Parameters
        ----------
        segmentation : np.ndarray
            Integer label volume.

        Returns
        -------
        np.ndarray
            Constrained segmentation.
        """
        constrained = segmentation.copy()

        # --- Rule 1: implant must overlap a tooth position ----------------
        implant_label = 10
        if np.any(constrained == implant_label):
            # Build a mask of all tooth-position voxels (dilated by 3 to
            # allow slight boundary misalignment).
            from scipy.ndimage import binary_dilation

            tooth_mask = np.isin(constrained, list(_VALID_IMPLANT_TOOTH_POSITIONS))
            # Dilate tooth regions to give a tolerance zone
            struct = np.ones((5, 5, 5), dtype=bool)
            dilated_tooth_mask = binary_dilation(tooth_mask, structure=struct, iterations=1)

            implant_mask = constrained == implant_label
            invalid_implant = implant_mask & ~dilated_tooth_mask
            n_removed = int(invalid_implant.sum())
            if n_removed > 0:
                constrained[invalid_implant] = 0
                logger.info(
                    "Rule 1 – removed %d implant voxels outside valid tooth positions.",
                    n_removed,
                )

        # --- Rule 2: bilateral symmetry advisory -------------------------
        bilateral_pairs: list[tuple[int, int, str]] = [
            (3, 4, "IAC"),       # L IAC / R IAC
            (5, 6, "Max Sinus"),  # L Max Sinus / R Max Sinus
        ]
        for left_lbl, right_lbl, name in bilateral_pairs:
            left_count = int(np.sum(constrained == left_lbl))
            right_count = int(np.sum(constrained == right_lbl))
            if left_count == 0 and right_count == 0:
                continue
            larger = max(left_count, right_count)
            smaller = min(left_count, right_count)
            if larger > 0 and smaller / larger < 0.05:
                side = "left" if left_count < right_count else "right"
                logger.warning(
                    "Rule 2 – %s %s is <5%% of contralateral side "
                    "(%d vs %d voxels). Possible segmentation artefact.",
                    side,
                    name,
                    smaller,
                    larger,
                )

        # --- Rule 3: pulp must be inside its parent tooth -----------------
        pulp_labels: list[int] = LABEL_CATEGORIES["Pulps"]
        for pulp_lbl in pulp_labels:
            if not np.any(constrained == pulp_lbl):
                continue
            # Derive parent tooth label: pulp 1XX → tooth XX
            tooth_lbl = pulp_lbl - 100
            tooth_mask = constrained == tooth_lbl
            if not np.any(tooth_mask):
                # Parent tooth absent – remove orphan pulp entirely
                n_removed = int(np.sum(constrained == pulp_lbl))
                constrained[constrained == pulp_lbl] = 0
                logger.info(
                    "Rule 3 – removed %d pulp-%d voxels (parent tooth %d absent).",
                    n_removed,
                    pulp_lbl,
                    tooth_lbl,
                )
                continue

            # Dilate tooth mask slightly to tolerate boundary effects
            from scipy.ndimage import binary_dilation as _bd

            struct = np.ones((3, 3, 3), dtype=bool)
            dilated_tooth = _bd(tooth_mask, structure=struct, iterations=1)
            pulp_mask = constrained == pulp_lbl
            outside = pulp_mask & ~dilated_tooth
            n_outside = int(outside.sum())
            if n_outside > 0:
                constrained[outside] = 0
                logger.info(
                    "Rule 3 – removed %d pulp-%d voxels outside tooth %d.",
                    n_outside,
                    pulp_lbl,
                    tooth_lbl,
                )

        return constrained

    def process(self, segmentation: np.ndarray) -> np.ndarray:
        """Run the full post-processing pipeline.

        1. Connected-component small-object removal (if enabled).
        2. Anatomical constraint enforcement.

        Parameters
        ----------
        segmentation : np.ndarray
            Raw integer label volume from nnU-Net prediction.

        Returns
        -------
        np.ndarray
            Post-processed segmentation.
        """
        result = segmentation.copy()

        if self.apply_cc:
            result = self.remove_small_components(result)
        else:
            logger.info("Connected-component filtering disabled – skipping.")

        result = self.apply_anatomical_constraints(result)
        return result

    # ------------------------------------------------------------------
    # File-level helpers
    # ------------------------------------------------------------------

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Load a NIfTI segmentation, post-process, and save.

        Parameters
        ----------
        input_path : Path
            Path to the input ``.nii.gz`` segmentation file.
        output_path : Path
            Destination path for the processed file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info("Processing %s → %s", input_path.name, output_path)

        nii = nib.load(str(input_path))
        seg_data: np.ndarray = np.asarray(nii.dataobj, dtype=np.int16)

        processed = self.process(seg_data)

        out_img = nib.Nifti1Image(processed, affine=nii.affine, header=nii.header)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, str(output_path))
        logger.info("Saved processed segmentation to %s", output_path)

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> list[Path]:
        """Batch post-process all ``.nii.gz`` files in a directory.

        Parameters
        ----------
        input_dir : Path
            Directory containing raw prediction NIfTI files.
        output_dir : Path
            Directory where processed files will be written.

        Returns
        -------
        list[Path]
            Paths to all successfully processed output files.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nifti_files = sorted(input_dir.glob("*.nii.gz"))
        if not nifti_files:
            logger.warning("No .nii.gz files found in %s", input_dir)
            return []

        logger.info(
            "Batch post-processing %d files from %s",
            len(nifti_files),
            input_dir,
        )

        processed_paths: list[Path] = []
        for idx, fpath in enumerate(nifti_files, 1):
            out_path = output_dir / fpath.name
            try:
                self.process_file(fpath, out_path)
                processed_paths.append(out_path)
            except Exception:
                logger.exception("Failed to process %s", fpath.name)

            if idx % 50 == 0 or idx == len(nifti_files):
                logger.info("Progress: %d / %d files processed.", idx, len(nifti_files))

        logger.info(
            "Batch complete – %d / %d files processed successfully.",
            len(processed_paths),
            len(nifti_files),
        )
        return processed_paths
