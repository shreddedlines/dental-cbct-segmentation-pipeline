"""
Dataset validator for the ToothFairy3 dental CBCT dataset.

Performs exhaustive integrity, geometry, and statistics checks on the raw
NIfTI volumes *before* any nnU-Net preprocessing.  Results are collected into
a structured dict and rendered as a Markdown report with matplotlib figures.

Usage
-----
    from dental_pipeline.config import load_config
    from dental_pipeline.dataset_validator import DatasetValidator

    config = load_config("configs/pipeline_config.yaml")
    validator = DatasetValidator(config)
    results = validator.run_all()
    validator.generate_report()
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dental_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level worker functions (must be picklable for multiprocessing)
# ============================================================================


def _load_header(filepath: str) -> Dict[str, Any]:
    """Load only the NIfTI header (fast, no voxel data)."""
    img = nib.load(filepath)
    hdr = img.header
    return {
        "path": filepath,
        "shape": tuple(img.shape),
        "affine": img.affine.tolist(),
        "spacing": tuple(float(v) for v in hdr.get_zooms()[:3]),
        "dtype": str(hdr.get_data_dtype()),
    }


def _validate_shape_pair(
    image_path: str, label_path: str
) -> Dict[str, Any]:
    """Check that an image and its label share shape and affine."""
    img = nib.load(image_path)
    lbl = nib.load(label_path)
    shape_match = img.shape[:3] == lbl.shape[:3]
    affine_match = bool(np.allclose(img.affine, lbl.affine, atol=1e-4))
    return {
        "case": Path(image_path).name,
        "image_shape": tuple(img.shape),
        "label_shape": tuple(lbl.shape),
        "shape_match": shape_match,
        "affine_match": affine_match,
    }


def _validate_label_values(
    label_path: str, valid_ids: List[int]
) -> Dict[str, Any]:
    """Check that every voxel value in a label file is in the valid set."""
    lbl = nib.load(label_path).get_fdata(dtype=np.float32)
    unique = set(int(v) for v in np.unique(lbl))
    valid_set = set(valid_ids)
    unexpected = unique - valid_set
    return {
        "case": Path(label_path).name,
        "unique_labels": sorted(unique),
        "unexpected_labels": sorted(unexpected),
        "is_valid": len(unexpected) == 0,
    }


def _compute_class_counts(label_path: str) -> Dict[int, int]:
    """Count voxels per class in a single label volume."""
    lbl = nib.load(label_path).get_fdata(dtype=np.float32).astype(np.int32)
    unique, counts = np.unique(lbl, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def _compute_intensity_stats(image_path: str) -> Dict[str, float]:
    """Compute basic intensity statistics for one image volume."""
    data = nib.load(image_path).get_fdata(dtype=np.float32)
    return {
        "case": Path(image_path).name,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "has_nan": bool(np.isnan(data).any()),
        "has_inf": bool(np.isinf(data).any()),
    }


# ============================================================================
# DatasetValidator
# ============================================================================


class DatasetValidator:
    """Comprehensive validation suite for the ToothFairy3 dataset.

    Parameters
    ----------
    config:
        A fully-resolved ``PipelineConfig`` instance.
    num_workers:
        Number of parallel processes for heavy I/O tasks.  Defaults to
        ``config.nnunet.num_processes``.
    """

    def __init__(
        self, config: PipelineConfig, num_workers: Optional[int] = None
    ) -> None:
        self.config = config
        self.num_workers = num_workers or config.nnunet.num_processes

        self.dataset_root = Path(config.paths.dataset_root)
        self.images_dir = self.dataset_root / "imagesTr"
        self.labels_dir = self.dataset_root / "labelsTr"
        self.dataset_json_path = self.dataset_root / "dataset.json"
        self.report_dir = Path(config.paths.validation_report)

        self.valid_label_ids: List[int] = config.get_valid_label_ids()
        self.label_names: Dict[int, str] = config.get_label_names()

        # Populated lazily
        self._image_files: Optional[List[Path]] = None
        self._label_files: Optional[List[Path]] = None
        self._results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    @property
    def image_files(self) -> List[Path]:
        if self._image_files is None:
            ending = self.config.dataset.file_ending
            self._image_files = sorted(self.images_dir.glob(f"*{ending}"))
        return self._image_files

    @property
    def label_files(self) -> List[Path]:
        if self._label_files is None:
            ending = self.config.dataset.file_ending
            self._label_files = sorted(self.labels_dir.glob(f"*{ending}"))
        return self._label_files

    def _case_id(self, path: Path) -> str:
        """Extract the case identifier from a filename.

        Handles both ``CASE_ID_0000.nii.gz`` (images) and
        ``CASE_ID.nii.gz`` (labels).
        """
        name = path.name
        # Strip file ending
        ending = self.config.dataset.file_ending
        stem = name[: -len(ending)] if name.endswith(ending) else path.stem
        # Strip nnU-Net channel suffix (_0000, _0001, …)
        stem = re.sub(r"_\d{4}$", "", stem)
        return stem

    def _sample(self, items: List[Any], n: int) -> List[Any]:
        """Deterministic sub-sample of *items* (seeded)."""
        if n >= len(items):
            return list(items)
        rng = np.random.RandomState(self.config.project.seed)
        indices = rng.choice(len(items), size=n, replace=False)
        return [items[i] for i in sorted(indices)]

    # ------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------

    def validate_file_pairing(self) -> Dict[str, Any]:
        """Check 1-to-1 mapping between imagesTr and labelsTr.

        Returns
        -------
        dict with keys:
            num_images, num_labels, paired, images_only, labels_only, passed
        """
        logger.info("Checking imagesTr ↔ labelsTr file pairing …")

        image_ids: Set[str] = {self._case_id(p) for p in self.image_files}
        label_ids: Set[str] = {self._case_id(p) for p in self.label_files}

        paired = image_ids & label_ids
        images_only = image_ids - label_ids
        labels_only = label_ids - image_ids

        result = {
            "num_images": len(image_ids),
            "num_labels": len(label_ids),
            "num_paired": len(paired),
            "images_without_labels": sorted(images_only),
            "labels_without_images": sorted(labels_only),
            "passed": len(images_only) == 0 and len(labels_only) == 0,
        }

        if result["passed"]:
            logger.info(
                "  ✓ File pairing OK — %d paired cases.", result["num_paired"]
            )
        else:
            logger.warning(
                "  ✗ File pairing FAILED — %d image-only, %d label-only.",
                len(images_only),
                len(labels_only),
            )

        self._results["file_pairing"] = result
        return result

    def validate_shapes(self, sample_size: int = 30) -> Dict[str, Any]:
        """Verify that images and labels share the same shape & affine.

        Parameters
        ----------
        sample_size:
            Number of cases to check (0 = all).
        """
        logger.info("Validating shapes (sample=%d) …", sample_size)

        pairs = self._matched_pairs()
        pairs = self._sample(pairs, sample_size) if sample_size else pairs

        mismatches: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_validate_shape_pair, str(ip), str(lp)): (ip, lp)
                for ip, lp in pairs
            }
            for fut in as_completed(futures):
                res = fut.result()
                if not res["shape_match"] or not res["affine_match"]:
                    mismatches.append(res)

        passed = len(mismatches) == 0
        result = {
            "checked": len(pairs),
            "mismatches": mismatches,
            "passed": passed,
        }

        if passed:
            logger.info("  ✓ All %d checked cases have matching shapes.", len(pairs))
        else:
            logger.warning(
                "  ✗ %d shape/affine mismatches found.", len(mismatches)
            )

        self._results["shapes"] = result
        return result

    def validate_labels(self, sample_size: int = 30) -> Dict[str, Any]:
        """Check that every voxel label is in the valid set from dataset.json.

        Parameters
        ----------
        sample_size:
            Number of label volumes to check (0 = all).
        """
        logger.info("Validating label values (sample=%d) …", sample_size)

        files = self._sample(self.label_files, sample_size) if sample_size else self.label_files
        invalid_cases: List[Dict[str, Any]] = []
        all_found_labels: Set[int] = set()

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(
                    _validate_label_values, str(f), self.valid_label_ids
                ): f
                for f in files
            }
            for fut in as_completed(futures):
                res = fut.result()
                all_found_labels.update(res["unique_labels"])
                if not res["is_valid"]:
                    invalid_cases.append(res)

        passed = len(invalid_cases) == 0
        missing_labels = set(self.valid_label_ids) - all_found_labels
        result = {
            "checked": len(files),
            "invalid_cases": invalid_cases,
            "all_observed_labels": sorted(all_found_labels),
            "missing_from_sample": sorted(missing_labels),
            "passed": passed,
        }

        if passed:
            logger.info(
                "  ✓ All %d checked volumes have valid label values.", len(files)
            )
        else:
            logger.warning(
                "  ✗ %d volumes contain unexpected label values.",
                len(invalid_cases),
            )

        self._results["labels"] = result
        return result

    def compute_class_distribution(
        self, sample_size: int = 30
    ) -> Dict[str, Any]:
        """Compute voxel counts per class across sampled label volumes.

        Parameters
        ----------
        sample_size:
            Number of label volumes to aggregate (0 = all).
        """
        logger.info("Computing class distribution (sample=%d) …", sample_size)

        files = (
            self._sample(self.label_files, sample_size)
            if sample_size
            else self.label_files
        )
        total_counts: Counter = Counter()

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_compute_class_counts, str(f)): f for f in files
            }
            for fut in as_completed(futures):
                counts = fut.result()
                total_counts.update(counts)

        # Convert to sorted dict
        distribution = {
            k: total_counts.get(k, 0) for k in sorted(self.valid_label_ids)
        }

        result = {
            "sample_size": len(files),
            "distribution": distribution,
            "zero_count_classes": [
                k for k, v in distribution.items() if v == 0
            ],
        }

        logger.info(
            "  %d / %d classes have non-zero voxel counts.",
            sum(1 for v in distribution.values() if v > 0),
            len(distribution),
        )

        self._results["class_distribution"] = result
        return result

    def compute_spacing_statistics(self) -> Dict[str, Any]:
        """Compute voxel-spacing statistics across ALL image volumes.

        Only reads NIfTI headers (no voxel data), so this is fast.
        """
        logger.info("Computing spacing statistics (all volumes) …")

        spacings: List[Tuple[float, float, float]] = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_load_header, str(f)): f for f in self.image_files
            }
            for fut in as_completed(futures):
                hdr = fut.result()
                spacings.append(hdr["spacing"])

        arr = np.array(spacings)  # (N, 3)
        result = {
            "num_volumes": len(spacings),
            "spacing_min": arr.min(axis=0).tolist(),
            "spacing_max": arr.max(axis=0).tolist(),
            "spacing_mean": arr.mean(axis=0).tolist(),
            "spacing_std": arr.std(axis=0).tolist(),
            "spacing_median": np.median(arr, axis=0).tolist(),
            "unique_spacings": len(set(spacings)),
            "all_spacings": spacings,
        }

        logger.info(
            "  Spacing — mean: (%.3f, %.3f, %.3f)  |  %d unique combos.",
            *result["spacing_mean"],
            result["unique_spacings"],
        )

        self._results["spacing"] = result
        return result

    def compute_intensity_statistics(
        self, sample_size: int = 10
    ) -> Dict[str, Any]:
        """Compute intensity statistics across a sample of image volumes.

        Parameters
        ----------
        sample_size:
            Number of image volumes to load (0 = all — slow!).
        """
        logger.info(
            "Computing intensity statistics (sample=%d) …", sample_size
        )

        files = (
            self._sample(self.image_files, sample_size)
            if sample_size
            else self.image_files
        )
        stats_list: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_compute_intensity_stats, str(f)): f
                for f in files
            }
            for fut in as_completed(futures):
                stats_list.append(fut.result())

        mins = [s["min"] for s in stats_list]
        maxs = [s["max"] for s in stats_list]
        means = [s["mean"] for s in stats_list]

        result = {
            "sample_size": len(files),
            "global_min": float(np.min(mins)),
            "global_max": float(np.max(maxs)),
            "mean_of_means": float(np.mean(means)),
            "per_case": stats_list,
        }

        logger.info(
            "  Intensity range: [%.1f, %.1f]  mean-of-means: %.1f",
            result["global_min"],
            result["global_max"],
            result["mean_of_means"],
        )

        self._results["intensity"] = result
        return result

    def check_nan_inf(self, sample_size: int = 10) -> Dict[str, Any]:
        """Check for NaN / Inf values in image volumes.

        Parameters
        ----------
        sample_size:
            Number of image volumes to check (0 = all).
        """
        logger.info("Checking for NaN/Inf (sample=%d) …", sample_size)

        files = (
            self._sample(self.image_files, sample_size)
            if sample_size
            else self.image_files
        )

        # Re-use intensity stats since they already compute nan/inf
        corrupt: List[str] = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_compute_intensity_stats, str(f)): f
                for f in files
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res["has_nan"] or res["has_inf"]:
                    corrupt.append(res["case"])

        passed = len(corrupt) == 0
        result = {
            "checked": len(files),
            "corrupt_cases": corrupt,
            "passed": passed,
        }

        if passed:
            logger.info("  ✓ No NaN/Inf found in %d checked volumes.", len(files))
        else:
            logger.warning(
                "  ✗ %d volumes contain NaN or Inf values.", len(corrupt)
            )

        self._results["nan_inf"] = result
        return result

    def compute_fov_breakdown(self) -> Dict[str, Any]:
        """Classify cases by field-of-view type based on filename prefix.

        ToothFairy3 filenames start with ``F_``, ``P_``, or ``S_`` indicating
        the field-of-view category.
        """
        logger.info("Computing FOV breakdown …")

        prefix_counts: Counter = Counter()
        unclassified: List[str] = []

        for f in self.image_files:
            case_id = self._case_id(f)
            # Try to extract leading letter(s) before first underscore
            match = re.match(r"^([A-Za-z]+)_", case_id)
            if match:
                prefix_counts[match.group(1).upper()] += 1
            else:
                unclassified.append(case_id)

        result = {
            "prefix_counts": dict(prefix_counts.most_common()),
            "unclassified": unclassified,
            "total": len(self.image_files),
        }

        for prefix, cnt in prefix_counts.most_common():
            logger.info("  %s: %d volumes", prefix, cnt)
        if unclassified:
            logger.info("  Unclassified: %d volumes", len(unclassified))

        self._results["fov_breakdown"] = result
        return result

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------

    def run_all(
        self,
        sample_size: int = 30,
        intensity_sample: int = 10,
    ) -> Dict[str, Any]:
        """Execute every validation check and return a unified results dict.

        Parameters
        ----------
        sample_size:
            Sample size for shape, label-value, and class-distribution checks.
        intensity_sample:
            Sample size for intensity statistics and NaN/Inf checks.

        Returns
        -------
        dict
            Aggregated results from all checks, keyed by check name.
        """
        t0 = time.time()
        logger.info("=" * 70)
        logger.info("Starting full dataset validation …")
        logger.info("  Dataset root : %s", self.dataset_root)
        logger.info("  Images dir   : %s", self.images_dir)
        logger.info("  Labels dir   : %s", self.labels_dir)
        logger.info("=" * 70)

        self.validate_file_pairing()
        self.validate_shapes(sample_size=sample_size)
        self.validate_labels(sample_size=sample_size)
        self.compute_class_distribution(sample_size=sample_size)
        self.compute_spacing_statistics()
        self.compute_intensity_statistics(sample_size=intensity_sample)
        self.check_nan_inf(sample_size=intensity_sample)
        self.compute_fov_breakdown()

        elapsed = time.time() - t0
        self._results["elapsed_seconds"] = elapsed
        logger.info("=" * 70)
        logger.info("Validation complete in %.1f s.", elapsed)
        logger.info("=" * 70)

        return self._results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> Path:
        """Create a Markdown validation report with embedded matplotlib
        figures, saved to the configured ``validation_report`` directory.

        Returns
        -------
        Path
            Path to the generated ``validation_report.md`` file.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = self.report_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        _a = lines.append  # shorthand

        _a("# Dataset Validation Report")
        _a(f"\n**Dataset root:** `{self.dataset_root}`\n")

        if "elapsed_seconds" in self._results:
            _a(f"**Total validation time:** {self._results['elapsed_seconds']:.1f} s\n")

        # ---- Summary table ---------------------------------------------------
        _a("## Summary\n")
        _a("| Check | Status |")
        _a("|-------|--------|")
        check_keys = [
            ("file_pairing", "File Pairing"),
            ("shapes", "Shape / Affine Match"),
            ("labels", "Label Value Validity"),
            ("nan_inf", "NaN / Inf Check"),
        ]
        all_passed = True
        for key, name in check_keys:
            if key in self._results:
                status = "✅ PASS" if self._results[key].get("passed") else "❌ FAIL"
                if not self._results[key].get("passed"):
                    all_passed = False
                _a(f"| {name} | {status} |")
        _a("")

        # ---- File pairing ----------------------------------------------------
        if "file_pairing" in self._results:
            fp = self._results["file_pairing"]
            _a("## File Pairing\n")
            _a(f"- Images found: **{fp['num_images']}**")
            _a(f"- Labels found: **{fp['num_labels']}**")
            _a(f"- Paired:       **{fp['num_paired']}**")
            if fp["images_without_labels"]:
                _a(f"- ⚠️ Images without labels: {fp['images_without_labels'][:10]}")
            if fp["labels_without_images"]:
                _a(f"- ⚠️ Labels without images: {fp['labels_without_images'][:10]}")
            _a("")

        # ---- Shapes ----------------------------------------------------------
        if "shapes" in self._results:
            sh = self._results["shapes"]
            _a("## Shape & Affine Validation\n")
            _a(f"- Cases checked: **{sh['checked']}**")
            _a(f"- Mismatches: **{len(sh['mismatches'])}**")
            if sh["mismatches"]:
                _a("\n| Case | Image Shape | Label Shape | Shape OK | Affine OK |")
                _a("|------|-------------|-------------|----------|-----------|")
                for m in sh["mismatches"][:20]:
                    _a(
                        f"| {m['case']} | {m['image_shape']} | {m['label_shape']} "
                        f"| {'✓' if m['shape_match'] else '✗'} "
                        f"| {'✓' if m['affine_match'] else '✗'} |"
                    )
            _a("")

        # ---- Spacing ---------------------------------------------------------
        if "spacing" in self._results:
            sp = self._results["spacing"]
            _a("## Voxel Spacing\n")
            _a(f"- Volumes analysed: **{sp['num_volumes']}**")
            _a(f"- Unique spacing combos: **{sp['unique_spacings']}**")
            _a(f"- Mean spacing: `{[f'{v:.4f}' for v in sp['spacing_mean']]}`")
            _a(f"- Min  spacing: `{[f'{v:.4f}' for v in sp['spacing_min']]}`")
            _a(f"- Max  spacing: `{[f'{v:.4f}' for v in sp['spacing_max']]}`")
            _a("")

            # Spacing histogram
            try:
                self._plot_spacing_histogram(
                    sp["all_spacings"], figures_dir / "spacing_histogram.png"
                )
                _a("![Spacing distribution](figures/spacing_histogram.png)\n")
            except Exception as exc:
                logger.warning("Could not plot spacing histogram: %s", exc)

        # ---- Intensity -------------------------------------------------------
        if "intensity" in self._results:
            it = self._results["intensity"]
            _a("## Intensity Statistics\n")
            _a(f"- Sample size: **{it['sample_size']}**")
            _a(f"- Global min: **{it['global_min']:.1f}**")
            _a(f"- Global max: **{it['global_max']:.1f}**")
            _a(f"- Mean of means: **{it['mean_of_means']:.1f}**")
            _a("")

        # ---- Class distribution ----------------------------------------------
        if "class_distribution" in self._results:
            cd = self._results["class_distribution"]
            _a("## Class Distribution\n")
            _a(f"- Sample size: **{cd['sample_size']}** volumes\n")

            if cd["zero_count_classes"]:
                _a(
                    f"- ⚠️ Classes with zero voxels in sample: "
                    f"`{cd['zero_count_classes']}`\n"
                )

            # Table (top 30 by count)
            dist = cd["distribution"]
            sorted_classes = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            _a("| Label ID | Name | Voxel Count |")
            _a("|----------|------|-------------|")
            for lid, cnt in sorted_classes[:40]:
                name = self.label_names.get(lid, "???")
                _a(f"| {lid} | {name} | {cnt:,} |")
            if len(sorted_classes) > 40:
                _a(f"| … | *({len(sorted_classes) - 40} more)* | … |")
            _a("")

            # Bar chart
            try:
                self._plot_class_distribution(
                    dist, figures_dir / "class_distribution.png"
                )
                _a("![Class distribution](figures/class_distribution.png)\n")
            except Exception as exc:
                logger.warning("Could not plot class distribution: %s", exc)

        # ---- FOV breakdown ---------------------------------------------------
        if "fov_breakdown" in self._results:
            fb = self._results["fov_breakdown"]
            _a("## Field-of-View Breakdown\n")
            _a(f"- Total volumes: **{fb['total']}**\n")
            _a("| Prefix | Count |")
            _a("|--------|-------|")
            for prefix, cnt in fb["prefix_counts"].items():
                _a(f"| {prefix} | {cnt} |")
            if fb["unclassified"]:
                _a(f"\n- Unclassified: {len(fb['unclassified'])} volumes")
            _a("")

            # Pie chart
            try:
                self._plot_fov_pie(
                    fb["prefix_counts"], figures_dir / "fov_breakdown.png"
                )
                _a("![FOV breakdown](figures/fov_breakdown.png)\n")
            except Exception as exc:
                logger.warning("Could not plot FOV breakdown: %s", exc)

        # ---- Label validity --------------------------------------------------
        if "labels" in self._results:
            lb = self._results["labels"]
            _a("## Label Validity\n")
            _a(f"- Volumes checked: **{lb['checked']}**")
            _a(f"- Invalid volumes: **{len(lb['invalid_cases'])}**")
            if lb["invalid_cases"]:
                for ic in lb["invalid_cases"][:10]:
                    _a(
                        f"  - `{ic['case']}`: unexpected labels "
                        f"`{ic['unexpected_labels']}`"
                    )
            if lb["missing_from_sample"]:
                _a(
                    f"- Labels not seen in sample: `{lb['missing_from_sample']}`"
                )
            _a("")

        # ---- NaN / Inf -------------------------------------------------------
        if "nan_inf" in self._results:
            ni = self._results["nan_inf"]
            _a("## NaN / Inf Check\n")
            _a(f"- Volumes checked: **{ni['checked']}**")
            if ni["passed"]:
                _a("- ✅ No corrupt values detected.")
            else:
                _a(f"- ❌ Corrupt volumes: `{ni['corrupt_cases']}`")
            _a("")

        # ---- Write -----------------------------------------------------------
        report_path = self.report_dir / "validation_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report written to %s", report_path)

        return report_path

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _plot_spacing_histogram(
        self,
        spacings: List[Tuple[float, float, float]],
        save_path: Path,
    ) -> None:
        """Histogram of voxel spacings per axis."""
        arr = np.array(spacings)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axis_names = ["X (sagittal)", "Y (coronal)", "Z (axial)"]
        for i, (ax, name) in enumerate(zip(axes, axis_names)):
            ax.hist(arr[:, i], bins=40, edgecolor="black", alpha=0.7)
            ax.set_xlabel(f"Spacing (mm) — {name}")
            ax.set_ylabel("Count")
            ax.set_title(name)
        fig.suptitle("Voxel Spacing Distribution", fontsize=14)
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)

    def _plot_class_distribution(
        self, distribution: Dict[int, int], save_path: Path
    ) -> None:
        """Horizontal bar chart of voxel counts per class."""
        # Filter out background for better visualisation range
        items = {
            k: v for k, v in distribution.items() if k != 0 and v > 0
        }
        if not items:
            return

        sorted_items = sorted(items.items(), key=lambda x: x[1])
        labels_list = [
            f"{lid}: {self.label_names.get(lid, '?')}" for lid, _ in sorted_items
        ]
        counts = [c for _, c in sorted_items]

        fig_height = max(6, len(labels_list) * 0.28)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.barh(range(len(counts)), counts, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(len(labels_list)))
        ax.set_yticklabels(labels_list, fontsize=7)
        ax.set_xlabel("Voxel Count")
        ax.set_title("Class Distribution (excl. background)")
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)

    def _plot_fov_pie(
        self, prefix_counts: Dict[str, int], save_path: Path
    ) -> None:
        """Pie chart of FOV prefix distribution."""
        if not prefix_counts:
            return
        labels_list = list(prefix_counts.keys())
        sizes = list(prefix_counts.values())

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            sizes,
            labels=labels_list,
            autopct="%1.1f%%",
            startangle=140,
            colors=plt.cm.Set3.colors[: len(sizes)],
        )
        ax.set_title("Field-of-View Breakdown")
        fig.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matched_pairs(self) -> List[Tuple[Path, Path]]:
        """Return list of (image_path, label_path) for all matched cases."""
        label_map: Dict[str, Path] = {
            self._case_id(p): p for p in self.label_files
        }
        pairs: List[Tuple[Path, Path]] = []
        for img in self.image_files:
            cid = self._case_id(img)
            if cid in label_map:
                pairs.append((img, label_map[cid]))
        return pairs

    # ------------------------------------------------------------------
    # Pass / fail summary
    # ------------------------------------------------------------------

    def has_critical_failures(self) -> bool:
        """Return ``True`` if any critical check failed."""
        critical_keys = ["file_pairing", "shapes", "labels", "nan_inf"]
        for key in critical_keys:
            if key in self._results and not self._results[key].get("passed", True):
                return True
        return False

    def print_summary(self) -> None:
        """Print a concise pass/fail summary to stdout."""
        checks = [
            ("file_pairing", "File Pairing"),
            ("shapes", "Shape / Affine Match"),
            ("labels", "Label Values"),
            ("nan_inf", "NaN / Inf"),
        ]
        print("\n" + "=" * 50)
        print("DATASET VALIDATION SUMMARY")
        print("=" * 50)
        for key, name in checks:
            if key in self._results:
                ok = self._results[key].get("passed", True)
                icon = "✅" if ok else "❌"
                print(f"  {icon}  {name}")
        print("=" * 50)
        if self.has_critical_failures():
            print("  RESULT: ❌ CRITICAL FAILURES DETECTED")
        else:
            print("  RESULT: ✅ ALL CHECKS PASSED")
        print("=" * 50 + "\n")
