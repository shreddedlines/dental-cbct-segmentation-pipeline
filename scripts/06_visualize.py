#!/usr/bin/env python
"""Generate professional 3D visualizations of dental CBCT segmentation.

Produces interactive HTML, multi-view PNGs, structure-specific renders,
and a 360° rotation animation from existing prediction files.

Usage
-----
    # Default: process num_cases from config, auto-detect prediction dir
    python scripts/06_visualize.py --config configs/pipeline_config.yaml

    # Specific cases
    python scripts/06_visualize.py --config configs/pipeline_config.yaml \\
        --case-ids ToothFairy3P_001 ToothFairy3F_001

    # More cases, custom prediction dir
    python scripts/06_visualize.py --config configs/pipeline_config.yaml \\
        --num-cases 5 --pred-dir outputs/predictions/fold_0/postprocessed

    # Use ground-truth labels (for validation visualization)
    python scripts/06_visualize.py --config configs/pipeline_config.yaml \\
        --use-ground-truth --num-cases 2

Output Structure (per case)
---------------------------
    outputs/visualizations/<case_id>/
    ├── interactive.html          Interactive Plotly viewer
    ├── front_view.png            Multi-view exports
    ├── left_view.png
    ├── right_view.png
    ├── top_view.png
    ├── isometric_view.png
    ├── teeth_only.png            Structure-specific
    ├── teeth_restorations.png
    ├── jaw_teeth.png
    ├── complete_anatomy.png
    ├── rotation.gif              360° animation
    └── rotation.mp4              (if ffmpeg available)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dental_pipeline.config import load_config, setup_logging
from dental_pipeline.visualization import ProfessionalVisualizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate professional 3D visualizations of dental CBCT "
            "segmentation (interactive HTML, multi-view PNGs, "
            "structure renders, 360° animation)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Override prediction/segmentation directory",
    )
    parser.add_argument(
        "--use-ground-truth", action="store_true",
        help="Visualize ground-truth labels instead of predictions",
    )
    parser.add_argument(
        "--skip-animation", action="store_true",
        help="Skip 360° rotation animation (much faster)",
    )
    return parser.parse_args()


def _find_seg_dir(config, args) -> Path:
    """Resolve the segmentation directory to use."""
    if args.use_ground_truth:
        seg_dir = Path(config.paths.dataset_root) / "labelsTr"
        logger.info("Using ground-truth labels from %s", seg_dir)
        return seg_dir

    if args.pred_dir:
        return Path(args.pred_dir)

    # Auto-detect: postprocessed > original_ids > raw > ground truth
    pred_base = Path(config.paths.predictions)
    fold_dirs = sorted(pred_base.glob("fold_*"))
    if fold_dirs:
        candidates = [
            ("postprocessed", fold_dirs[-1] / "postprocessed"),
            ("original_ids",  fold_dirs[-1] / "original_ids"),
            ("raw",           fold_dirs[-1] / "raw"),
        ]
        for label, cand in candidates:
            if cand.is_dir() and any(cand.glob("*.nii.gz")):
                if label == "raw":
                    logger.warning(
                        "Using raw predictions (consecutive IDs). "
                        "Run 05_evaluate.py first for reverse-mapped labels."
                    )
                return cand

    # Fallback
    seg_dir = Path(config.paths.dataset_root) / "labelsTr"
    logger.warning(
        "No predictions found — falling back to ground truth for visualization."
    )
    return seg_dir


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config.create_output_dirs()
    setup_logging(config, log_filename="06_visualize.log")

    logger.info("=" * 70)
    logger.info("PHASE 6 — Professional Visualization")
    logger.info("=" * 70)

    seg_dir = _find_seg_dir(config, args)
    if not seg_dir.is_dir():
        logger.error("Segmentation directory not found: %s", seg_dir)
        return 1

    logger.info("Segmentation dir:  %s", seg_dir)

    output_dir = Path(config.paths.visualizations)
    num_cases = args.num_cases or config.visualization.num_cases

    viz = ProfessionalVisualizer(config)

    # If --skip-animation, monkey-patch to no-op (avoids modifying class)
    if args.skip_animation:
        logger.info("Animation generation disabled (--skip-animation).")
        viz.export_rotation = lambda *a, **kw: []

    generated = viz.generate_visualizations(
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
    logger.info("")
    for p in generated:
        logger.info("  → %s", p)
    logger.info("")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
