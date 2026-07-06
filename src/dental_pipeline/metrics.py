"""Evaluation metrics module for dental CBCT segmentation.

Computes per-class Dice coefficient, 95th-percentile Hausdorff distance,
sensitivity, and precision.  Aggregates results across cases and label
categories, and generates CSV reports with matplotlib visualisations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server / CI use
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from dental_pipeline.config import (
    LABEL_CATEGORIES,
    PipelineConfig,
    get_label_names,
)

logger = logging.getLogger(__name__)

def _compute_case_worker(args):
    """
    Worker process for one case.
    """
    pred_path, gt_path, config = args

    calc = MetricsCalculator(config)
    return calc.compute_case_metrics(pred_path, gt_path)

class MetricsCalculator:
    """Compute and report segmentation evaluation metrics.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration loaded from YAML.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.label_names: dict[int, str] = get_label_names()
        # All non-background label IDs
        self.all_label_ids: list[int] = sorted(
            lbl for lbl in self.label_names if lbl != 0
        )
        logger.info(
            "MetricsCalculator initialised – tracking %d labels.",
            len(self.all_label_ids),
        )

    # ------------------------------------------------------------------
    # Per-class metric functions
    # ------------------------------------------------------------------

    @staticmethod
    def dice_coefficient(
        pred: np.ndarray,
        gt: np.ndarray,
        label: int,
    ) -> float:
        """Compute the Sørensen–Dice coefficient for a single label.

        Parameters
        ----------
        pred, gt : np.ndarray
            Integer label volumes (same shape).
        label : int
            The label ID to evaluate.

        Returns
        -------
        float
            Dice score in [0, 1].  Returns ``np.nan`` when the label is
            absent in both *pred* and *gt*.
        """
        pred_mask = pred == label
        gt_mask = gt == label

        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()

        if pred_sum == 0 and gt_sum == 0:
            return np.nan  # label absent from both – undefined

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return float(2.0 * intersection / (pred_sum + gt_sum))

    @staticmethod
    def hausdorff_distance_95(
        pred: np.ndarray,
        gt: np.ndarray,
        label: int,
        voxel_spacing: tuple[float, ...] | None = None,
    ) -> float:
        """95th-percentile Hausdorff distance for a single label.

        Uses the Euclidean distance transform for efficiency.

        Parameters
        ----------
        pred, gt : np.ndarray
            Integer label volumes (same shape).
        label : int
            The label ID to evaluate.
        voxel_spacing : tuple[float, ...], optional
            Physical spacing per axis.  Defaults to isotropic (1, 1, 1).

        Returns
        -------
        float
            HD95 in physical units, or ``np.nan`` if the label is absent
            from either volume.
        """
        pred_mask = pred == label
        gt_mask = gt == label

        if not pred_mask.any() or not gt_mask.any():
            return np.nan

        if voxel_spacing is None:
            voxel_spacing = (1.0,) * pred.ndim

        # Surface voxels (boundary of the binary region)
        from scipy.ndimage import binary_erosion

        pred_border = pred_mask ^ binary_erosion(pred_mask)
        gt_border = gt_mask ^ binary_erosion(gt_mask)

        if not pred_border.any() or not gt_border.any():
            # Fallback: structures are a single voxel thick
            pred_border = pred_mask
            gt_border = gt_mask

        # Distance transform of the complementary masks
        dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
        dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)

        # Directed distances
        dist_gt_to_pred = dt_pred[gt_border]
        dist_pred_to_gt = dt_gt[pred_border]

        all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
        return float(np.percentile(all_distances, 95))

    @staticmethod
    def sensitivity(
        pred: np.ndarray,
        gt: np.ndarray,
        label: int,
    ) -> float:
        """Sensitivity (recall / true positive rate) for a single label.

        Returns
        -------
        float
            Sensitivity in [0, 1], or ``np.nan`` if the label is absent
            from *gt*.
        """
        gt_mask = gt == label
        if not gt_mask.any():
            return np.nan
        pred_mask = pred == label
        tp = np.logical_and(pred_mask, gt_mask).sum()
        return float(tp / gt_mask.sum())

    @staticmethod
    def precision_score(
        pred: np.ndarray,
        gt: np.ndarray,
        label: int,
    ) -> float:
        """Precision (positive predictive value) for a single label.

        Returns
        -------
        float
            Precision in [0, 1], or ``np.nan`` if the label is absent
            from *pred*.
        """
        pred_mask = pred == label
        if not pred_mask.any():
            return np.nan
        gt_mask = gt == label
        tp = np.logical_and(pred_mask, gt_mask).sum()
        return float(tp / pred_mask.sum())

    # ------------------------------------------------------------------
    # Case-level aggregation
    # ------------------------------------------------------------------

    def compute_case_metrics(
        self,
        pred_path: Path | str,
        gt_path: Path | str,
    ) -> dict[str, Any]:
        """Compute all metrics for every label in a single case.

        Parameters
        ----------
        pred_path : Path
            Path to the predicted NIfTI segmentation.
        gt_path : Path
            Path to the ground-truth NIfTI segmentation.

        Returns
        -------
        dict
            ``{"case_id": str, "labels": {label_id: {metric: value}}}``
        """
        pred_path = Path(pred_path)
        gt_path = Path(gt_path)

        pred_nii = nib.load(str(pred_path))
        gt_nii = nib.load(str(gt_path))
        pred_data = np.asarray(pred_nii.dataobj, dtype=np.int16)
        gt_data = np.asarray(gt_nii.dataobj, dtype=np.int16)

        # Determine voxel spacing from the ground-truth header
        spacing: tuple[float, ...] = tuple(float(s) for s in gt_nii.header.get_zooms()[:3])

        case_id = pred_path.name.replace(".nii.gz", "")

        # Determine which labels are present in either volume
        present_labels = set(np.unique(pred_data)) | set(np.unique(gt_data))
        present_labels.discard(0)  # skip background

        label_metrics: dict[int, dict[str, float]] = {}
        for lbl in sorted(present_labels):
            label_metrics[lbl] = {
                "dice": self.dice_coefficient(pred_data, gt_data, lbl),
                "hd95": self.hausdorff_distance_95(
                    pred_data, gt_data, lbl, voxel_spacing=spacing
                ),
                "sensitivity": self.sensitivity(pred_data, gt_data, lbl),
                "precision": self.precision_score(pred_data, gt_data, lbl),
            }

        logger.debug(
            "Case %s – computed metrics for %d labels.",
            case_id,
            len(label_metrics),
        )


           
        return {"case_id": case_id, "labels": label_metrics}

    # ------------------------------------------------------------------
    # Dataset-level aggregation
    # ------------------------------------------------------------------

    
    def compute_dataset_metrics(
        self,
        pred_dir: Path | str,
        gt_dir: Path | str,
        num_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        Parallel computation of metrics across cases.
        """

        pred_dir = Path(pred_dir)
        gt_dir = Path(gt_dir)

        if num_workers is None:
            num_workers = min(24, os.cpu_count() or 24)

        pred_files = sorted(pred_dir.glob("*.nii.gz"))

        if not pred_files:
            logger.warning("No prediction files found in %s", pred_dir)
            return pd.DataFrame()

        jobs = []

        for pred_file in pred_files:
            gt_file = gt_dir / pred_file.name

            if not gt_file.exists():
                logger.warning(
                    "Ground truth missing for %s",
                    pred_file.name,
                )
                continue

            jobs.append(
                (
                    pred_file,
                    gt_file,
                    self.config,
                )
            )

        logger.info(
            "Computing metrics using %d CPU workers...",
            num_workers,
        )

        rows = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                executor.submit(
                    _compute_case_worker,
                    job,
                ): job[0].name
                for job in jobs
            }

            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="Computing metrics",
                unit="case",
            ):

                try:
                    case_result = future.result()
                except Exception as e:
                    logger.exception(
                        "Failed on %s",
                        future_map[future],
                    )
                    continue

                case_id = case_result["case_id"]

                for lbl_id, metrics in case_result["labels"].items():
                    rows.append(
                        {
                            "case_id": case_id,
                            "label_id": lbl_id,
                            "label_name": self.label_names.get(
                                lbl_id,
                                f"Label_{lbl_id}",
                            ),
                            "dice": metrics["dice"],
                            "hd95": metrics["hd95"],
                            "sensitivity": metrics["sensitivity"],
                            "precision": metrics["precision"],
                        }
                    )

        df = pd.DataFrame(rows)

        logger.info(
            "Dataset metrics computed – %d rows across %d cases.",
            len(df),
            df["case_id"].nunique() if len(df) else 0,
        )
        return df






    # ------------------------------------------------------------------
    # Category-level summary
    # ------------------------------------------------------------------

    @staticmethod
    def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean metrics grouped by ``LABEL_CATEGORIES``.

        Parameters
        ----------
        df : pd.DataFrame
            Per-case, per-label metrics produced by
            :meth:`compute_dataset_metrics`.

        Returns
        -------
        pd.DataFrame
            Columns: ``category, dice_mean, dice_std, hd95_mean,
            hd95_std, sensitivity_mean, sensitivity_std,
            precision_mean, precision_std``.
        """
        if df.empty:
            return pd.DataFrame()

        summary_rows: list[dict[str, Any]] = []
        for category, label_ids in LABEL_CATEGORIES.items():
            cat_df = df[df["label_id"].isin(label_ids)]
            if cat_df.empty:
                continue
            summary_rows.append(
                {
                    "category": category,
                    "dice_mean": cat_df["dice"].mean(skipna=True),
                    "dice_std": cat_df["dice"].std(skipna=True),
                    "hd95_mean": cat_df["hd95"].mean(skipna=True),
                    "hd95_std": cat_df["hd95"].std(skipna=True),
                    "sensitivity_mean": cat_df["sensitivity"].mean(skipna=True),
                    "sensitivity_std": cat_df["sensitivity"].std(skipna=True),
                    "precision_mean": cat_df["precision"].mean(skipna=True),
                    "precision_std": cat_df["precision"].std(skipna=True),
                }
            )

        # Overall row
        summary_rows.append(
            {
                "category": "Overall",
                "dice_mean": df["dice"].mean(skipna=True),
                "dice_std": df["dice"].std(skipna=True),
                "hd95_mean": df["hd95"].mean(skipna=True),
                "hd95_std": df["hd95"].std(skipna=True),
                "sensitivity_mean": df["sensitivity"].mean(skipna=True),
                "sensitivity_std": df["sensitivity"].std(skipna=True),
                "precision_mean": df["precision"].mean(skipna=True),
                "precision_std": df["precision"].std(skipna=True),
            }
        )

        return pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        df: pd.DataFrame,
        output_dir: Path | str,
    ) -> None:
        """Save CSV reports and matplotlib visualisations.

        Generates
        ---------
        - ``metrics_per_case_label.csv`` – full per-case, per-label table.
        - ``metrics_mean_per_label.csv`` – mean across cases per label.
        - ``metrics_category_summary.csv`` – category-level summary.
        - ``dice_per_class.png`` – bar chart of mean Dice per class.
        - ``dice_category_summary.png`` – bar chart of category Dice.
        - ``dice_boxplot.png`` – box plot of Dice distribution per class.
        - ``hd95_boxplot.png`` – box plot of HD95 distribution per class.

        Parameters
        ----------
        df : pd.DataFrame
            Per-case, per-label metrics from :meth:`compute_dataset_metrics`.
        output_dir : Path
            Directory where all output files are written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            logger.warning("Empty DataFrame – skipping report generation.")
            return

        # ---- CSV reports -------------------------------------------------
        csv_full = output_dir / "metrics_per_case_label.csv"
        df.to_csv(csv_full, index=False, float_format="%.4f")
        logger.info("Saved full metrics CSV → %s", csv_full)

        # Mean per label across cases
        mean_df = (
            df.groupby(["label_id", "label_name"])
            .agg(
                dice_mean=("dice", "mean"),
                dice_std=("dice", "std"),
                hd95_mean=("hd95", "mean"),
                hd95_std=("hd95", "std"),
                sensitivity_mean=("sensitivity", "mean"),
                sensitivity_std=("sensitivity", "std"),
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                n_cases=("case_id", "count"),
            )
            .reset_index()
            .sort_values("label_id")
        )
        csv_mean = output_dir / "metrics_mean_per_label.csv"
        mean_df.to_csv(csv_mean, index=False, float_format="%.4f")
        logger.info("Saved per-label mean CSV → %s", csv_mean)

        # Category summary
        cat_df = self.compute_category_summary(df)
        csv_cat = output_dir / "metrics_category_summary.csv"
        cat_df.to_csv(csv_cat, index=False, float_format="%.4f")
        logger.info("Saved category summary CSV → %s", csv_cat)

        # ---- Visualisations ---------------------------------------------
        self._plot_dice_per_class(mean_df, output_dir)
        self._plot_category_summary(cat_df, output_dir)
        self._plot_boxplot(df, metric="dice", output_dir=output_dir)
        self._plot_boxplot(df, metric="hd95", output_dir=output_dir)

        logger.info("Report generation complete → %s", output_dir)

    # ------------------------------------------------------------------
    # Private plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_dice_per_class(
        mean_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Bar chart of mean Dice per class."""
        if mean_df.empty:
            return
        fig, ax = plt.subplots(figsize=(max(14, len(mean_df) * 0.35), 6))
        colors: list[str] = []
        category_colors = {
            "Anatomical": "#1f77b4",
            "Restorations": "#ff7f0e",
            "Teeth": "#2ca02c",
            "Canals": "#d62728",
            "Pulps": "#9467bd",
        }
        for lbl_id in mean_df["label_id"]:
            found_cat = "Other"
            for cat, ids in LABEL_CATEGORIES.items():
                if lbl_id in ids:
                    found_cat = cat
                    break
            colors.append(category_colors.get(found_cat, "#7f7f7f"))

        x_labels = [
            f"{row.label_name}\n({row.label_id})"
            for row in mean_df.itertuples()
        ]
        bars = ax.bar(
            range(len(mean_df)),
            mean_df["dice_mean"].fillna(0),
            yerr=mean_df["dice_std"].fillna(0),
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
        )
        ax.set_xticks(range(len(mean_df)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6, ha="center")
        ax.set_ylabel("Dice Coefficient")
        ax.set_title("Mean Dice Score per Class")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=mean_df["dice_mean"].mean(skipna=True), color="red",
                    linestyle="--", linewidth=0.8, label="Overall mean")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = output_dir / "dice_per_class.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        logger.info("Saved dice_per_class.png")

    @staticmethod
    def _plot_category_summary(
        cat_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Bar chart of Dice by label category."""
        if cat_df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        category_colors = {
            "Anatomical": "#1f77b4",
            "Restorations": "#ff7f0e",
            "Teeth": "#2ca02c",
            "Canals": "#d62728",
            "Pulps": "#9467bd",
            "Overall": "#17becf",
        }
        colors = [category_colors.get(c, "#7f7f7f") for c in cat_df["category"]]
        ax.bar(
            cat_df["category"],
            cat_df["dice_mean"],
            yerr=cat_df["dice_std"],
            color=colors,
            edgecolor="white",
            capsize=4,
        )
        ax.set_ylabel("Dice Coefficient")
        ax.set_title("Mean Dice by Label Category")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        out_path = output_dir / "dice_category_summary.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        logger.info("Saved dice_category_summary.png")

    @staticmethod
    def _plot_boxplot(
        df: pd.DataFrame,
        metric: str,
        output_dir: Path,
    ) -> None:
        """Box plot of a metric's distribution across classes."""
        if df.empty:
            return
        # Group by label_name, collect metric values
        groups: dict[str, list[float]] = {}
        for label_name, grp in df.groupby("label_name"):
            vals = grp[metric].dropna().tolist()
            if vals:
                groups[str(label_name)] = vals

        if not groups:
            return

        sorted_names = sorted(groups.keys())
        data = [groups[n] for n in sorted_names]

        fig, ax = plt.subplots(figsize=(max(14, len(sorted_names) * 0.35), 6))
        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(sorted_names) + 1))
        ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)
        ylabel = "Dice Coefficient" if metric == "dice" else "HD95 (mm)"
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} Distribution per Class")
        fig.tight_layout()
        out_path = output_dir / f"{metric}_boxplot.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        logger.info("Saved %s_boxplot.png", metric)
