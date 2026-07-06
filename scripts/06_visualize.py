#!/usr/bin/env python
"""Generate interactive 3D visualizations of CBCT volumes and segmentation.

Usage
-----
    python scripts/06_visualize.py --config configs/pipeline_config.yaml
    python scripts/06_visualize.py --config configs/pipeline_config.yaml --case-ids ToothFairy3P_001 ToothFairy3F_001
    python scripts/06_visualize.py --config configs/pipeline_config.yaml --num-cases 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dental_pipeline.config import load_config, setup_logging
from dental_pipeline.visualization import VolumeVisualizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D visualizations of dental CBCT segmentation.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/pipeline_config.yaml",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--case-ids", nargs="+", default=None,
        help="Specific case IDs to visualize (e.g. ToothFairy3P_001)",
    )
    parser.add_argument(
        "--num-cases", type=int, default=None,
        help="Number of cases to visualize (overrides config)",
    )
    parser.add_argument(
        "--pred-dir", type=str, default=None,
        help="Override prediction directory (default: from config)",
    )
    parser.add_argument(
        "--use-ground-truth", action="store_true",
        help="Visualize ground-truth labels instead of predictions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config.create_output_dirs()
    setup_logging(config, log_filename="06_visualize.log")

    logger.info("=" * 70)
    logger.info("PHASE 6 — Visualization")
    logger.info("=" * 70)

    viz = VolumeVisualizer(config)

    # Image directory
    image_dir = Path(config.paths.dataset_root) / "imagesTr"
    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        return 1

    # Segmentation directory (predictions or ground truth)
    if args.use_ground_truth:
        seg_dir = Path(config.paths.dataset_root) / "labelsTr"
        logger.info("Using ground-truth labels from %s", seg_dir)
    elif args.pred_dir:
        seg_dir = Path(args.pred_dir)
    else:
        # Try to find the latest prediction folder.
        # Priority: postprocessed (original IDs) > original_ids > raw
        pred_base = Path(config.paths.predictions)
        fold_dirs = sorted(pred_base.glob("fold_*"))
        if fold_dirs:
            pp_dir = fold_dirs[-1] / "postprocessed"
            orig_dir = fold_dirs[-1] / "original_ids"
            raw_dir = fold_dirs[-1] / "raw"
            if pp_dir.is_dir() and any(pp_dir.glob("*.nii.gz")):
                seg_dir = pp_dir
            elif orig_dir.is_dir() and any(orig_dir.glob("*.nii.gz")):
                seg_dir = orig_dir
            elif raw_dir.is_dir() and any(raw_dir.glob("*.nii.gz")):
                seg_dir = raw_dir
                logger.warning(
                    "Using raw predictions (consecutive IDs). "
                    "Run 05_evaluate.py first for reverse-mapped labels."
                )
            else:
                seg_dir = fold_dirs[-1]
        else:
            # Fallback to ground truth
            seg_dir = Path(config.paths.dataset_root) / "labelsTr"
            logger.warning(
                "No predictions found — falling back to ground truth for visualization."
            )

    if not seg_dir.is_dir():
        logger.error("Segmentation directory not found: %s", seg_dir)
        return 1

    logger.info("Image dir:         %s", image_dir)
    logger.info("Segmentation dir:  %s", seg_dir)

    output_dir = Path(config.paths.visualizations)
    num_cases = args.num_cases or config.visualization.num_cases

    generated = viz.generate_visualizations(
        image_dir=image_dir,
        pred_dir=seg_dir,
        output_dir=output_dir,
        num_cases=num_cases,
        case_ids=args.case_ids,
    )

    if not generated:
        logger.warning("No visualizations were generated.")
        return 1

    logger.info("")
    logger.info("=" * 70)
    logger.info("VISUALIZATION COMPLETE — %d files generated:", len(generated))
    for p in generated:
        logger.info("  → %s", p)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
