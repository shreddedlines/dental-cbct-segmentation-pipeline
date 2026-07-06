#!/usr/bin/env python
"""Run evaluation: inference on the validation fold, post-processing, metrics.

nnU-Net predictions use consecutive label IDs (0, 1, 2, …, 76). This script
automatically reverse-maps them back to the original ToothFairy3 FDI label IDs
before computing metrics against the original ground truth.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dental_pipeline.config import load_config, setup_logging
from dental_pipeline.label_remapping import LabelRemapper
from dental_pipeline.metrics import MetricsCalculator
from dental_pipeline.postprocessing import PostProcessor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on the validation set, apply post-processing, and compute metrics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to evaluate (default: 0)",
    )
    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="Skip post-processing step",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip nnU-Net inference (use existing predictions)",
    )
    parser.add_argument(
        "--skip-reverse-map",
        action="store_true",
        help="Skip reverse label mapping (predictions already have original IDs)",
    )
    return parser.parse_args()


def run_inference(config, fold: int) -> Path:
    """Run nnUNetv2_predict on the validation fold."""
    config.set_nnunet_env()

    dataset_id = config.dataset.id
    configuration = config.nnunet.configuration
    trainer = config.nnunet.trainer
    plans = config.nnunet.plans

    pred_dir = Path(config.paths.predictions) / f"fold_{fold}" / "raw"
    pred_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(config.paths.nnunet_raw) / config.dataset.name / "imagesTr"

    cmd = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(pred_dir),
        "-d",
        str(dataset_id),
        "-c",
        configuration,
        "-tr",
        trainer,
        "-p",
        plans,
        "-f",
        str(fold),
    ]

    logger.info("Running inference: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("nnUNetv2_predict not found. Ensure nnU-Net v2 is installed.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        logger.error("Inference failed with exit code %d", exc.returncode)
        sys.exit(1)

    return pred_dir


def _find_mapping_file(config) -> Path:
    """Locate the label_mapping.json file."""
    candidates = [
        Path(config.paths.output_root) / "label_mapping.json",
        Path(config.paths.nnunet_raw) / config.dataset.name / "label_mapping.json",
    ]

    for p in candidates:
        if p.is_file():
            return p

    raise FileNotFoundError(
        "label_mapping.json not found. Run 02_setup_nnunet.py first. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def main() -> int:
    args = parse_args()

    config = load_config(args.config)
    config.create_output_dirs()
    setup_logging(config, log_filename="05_evaluate.log")

    logger.info("=" * 70)
    logger.info("PHASE 5 — Evaluation (fold %d)", args.fold)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Inference
    # ------------------------------------------------------------------
    if args.skip_inference:
        raw_pred_dir = Path(config.paths.predictions) / f"fold_{args.fold}" / "raw"

        if not raw_pred_dir.exists():
            logger.error("Prediction directory not found: %s", raw_pred_dir)
            return 1

        num_preds = len(list(raw_pred_dir.glob("*.nii.gz")))

        if num_preds == 0:
            logger.error("No prediction files found in %s", raw_pred_dir)
            return 1

        logger.info(
            "Skipping inference — using %d predictions from %s",
            num_preds,
            raw_pred_dir,
        )

    else:
        raw_pred_dir = run_inference(config, args.fold)

    # ------------------------------------------------------------------
    # Step 2: Reverse mapping
    # ------------------------------------------------------------------
    if args.skip_reverse_map:
        logger.info("Reverse mapping skipped by user request.")
        original_id_dir = raw_pred_dir
    else:
        logger.info("Reverse-mapping predictions to original label IDs...")

        mapping_path = _find_mapping_file(config)

        remapper = LabelRemapper.from_mapping_file(mapping_path, config)

        original_id_dir = (
            Path(config.paths.predictions)
            / f"fold_{args.fold}"
            / "original_ids"
        )

        original_id_dir.mkdir(parents=True, exist_ok=True)

        remapper.reverse_map_directory(
            raw_pred_dir,
            original_id_dir,
        )

        logger.info(
            "Reverse-mapped predictions → %s",
            original_id_dir,
        )

    # ------------------------------------------------------------------
    # Step 3: Post-processing
    # ------------------------------------------------------------------
    if args.no_postprocessing:
        logger.info("Post-processing skipped by user request.")
        eval_pred_dir = original_id_dir

    else:
        logger.info("Applying post-processing...")

        pp_dir = (
            Path(config.paths.predictions)
            / f"fold_{args.fold}"
            / "postprocessed"
        )

        pp_dir.mkdir(parents=True, exist_ok=True)

        postprocessor = PostProcessor(config)
        postprocessor.process_directory(original_id_dir, pp_dir)

        eval_pred_dir = pp_dir

    # ------------------------------------------------------------------
    # Step 4: Metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics...")

    gt_dir = Path(config.paths.dataset_root) / "labelsTr"

    metrics_calc = MetricsCalculator(config)

    df = metrics_calc.compute_dataset_metrics(
        eval_pred_dir,
        gt_dir,
    )

    metrics_dir = Path(config.paths.metrics) / f"fold_{args.fold}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "per_case_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Per-case metrics saved → %s", csv_path)

    summary_df = metrics_calc.compute_category_summary(df)

    summary_path = metrics_dir / "category_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info("Category summary saved → %s", summary_path)

    metrics_calc.generate_report(df, metrics_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY (fold %d)", args.fold)
    logger.info("=" * 70)
    logger.info("\n%s", summary_df.to_string(index=False))
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())