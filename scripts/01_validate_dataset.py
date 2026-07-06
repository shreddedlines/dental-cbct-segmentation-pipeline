#!/usr/bin/env python
"""Validate the ToothFairy3 dataset before any processing.

Performs comprehensive integrity checks, computes dataset statistics,
and generates a markdown report with figures.

Usage
-----
    python scripts/01_validate_dataset.py --config configs/pipeline_config.yaml
    python scripts/01_validate_dataset.py --config configs/pipeline_config.yaml --full-sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the src package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dental_pipeline.config import load_config, setup_logging
from dental_pipeline.dataset_validator import DatasetValidator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the ToothFairy3 dataset and generate a report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline YAML config (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of volumes to sample for heavy checks (default: 30)",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Run all checks on every volume (slow — hours for 532 volumes)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config.create_output_dirs()
    setup_logging(config, log_filename="01_validate_dataset.log")

    logger.info("=" * 70)
    logger.info("PHASE 1 — Dataset Validation")
    logger.info("=" * 70)

    sample_size = None if args.full_sweep else args.sample_size
    logger.info(
        "Sample size: %s",
        "ALL (full sweep)" if sample_size is None else sample_size,
    )

    validator = DatasetValidator(config)

    # Run all checks
    results = validator.run_all(sample_size=sample_size)

    # Write JSON results
    report_dir = Path(config.paths.validation_report)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "validation_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    logger.info("JSON results saved → %s", json_path)

    # Generate markdown report (compatible with both validator versions)
    try:
        validator.generate_report()
    except TypeError:
        validator.generate_report(report_dir)

    # Print summary
    critical_failures = results.get("critical_failures", [])
    warnings = results.get("warnings", [])

    logger.info("")
    logger.info("=" * 70)

    if critical_failures:
        logger.error(
            "VALIDATION FAILED — %d critical issue(s):",
            len(critical_failures),
        )
        for failure in critical_failures:
            logger.error("  ✗ %s", failure)

        logger.info("=" * 70)
        return 1

    logger.info("VALIDATION PASSED ✓")

    if warnings:
        logger.warning("%d warning(s):", len(warnings))
        for warning in warnings:
            logger.warning("  ⚠ %s", warning)

    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())