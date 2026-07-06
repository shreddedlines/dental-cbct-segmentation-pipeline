#!/usr/bin/env python3
"""Run nnU-Net v2 planning and preprocessing for ToothFairy3.

Sets nnU-Net environment variables, then invokes
``nnUNetv2_plan_and_preprocess`` via subprocess. Optionally run in
verify-only mode to check dataset integrity without preprocessing.

Usage
-----
    python scripts/03_preprocess.py --config configs/pipeline_config.yaml
    python scripts/03_preprocess.py --config configs/pipeline_config.yaml --verify-only
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project package is importable when running from the repo root.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dental_pipeline.config import load_config  # noqa: E402
from dental_pipeline.nnunet_setup import NNUNetSetup  # noqa: E402


def _setup_logging(log_dir: Path) -> None:
    """Configure root logger to write to both console and a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "03_preprocess.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


def _copy_plans_file(config, log_dir: Path, logger: logging.Logger) -> None:
    """Copy generated nnUNet plans file into logs folder."""
    plans_name = config.nnunet.plans + ".json"

    plans_src = (
        Path(config.paths.nnunet_preprocessed)
        / config.dataset.name
        / plans_name
    )

    if not plans_src.exists():
        logger.warning("Plans file not found at %s", plans_src)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plans_dst = log_dir / f"{plans_name}.{timestamp}.bak"

    shutil.copy2(plans_src, plans_dst)

    logger.info("Plans file copied to %s", plans_dst)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nnU-Net planning and preprocessing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Pipeline YAML config.",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify dataset integrity.",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    config = load_config(args.config)

    log_dir = Path(config.paths.logs)
    _setup_logging(log_dir)

    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 1. Set environment
    # ------------------------------------------------------------------
    setup = NNUNetSetup(config)
    setup.set_env_vars()

    logger.info("nnU-Net environment variables set.")

    # ------------------------------------------------------------------
    # 2. Build command
    # ------------------------------------------------------------------
    dataset_id = str(config.dataset.id)
    num_processes = str(config.nnunet.num_processes)

    if args.verify_only:
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            dataset_id,
            "--verify_dataset_integrity",
        ]

        logger.info("Running in VERIFY ONLY mode.")

    else:
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            dataset_id,
            "--verify_dataset_integrity",
            "-np",
            num_processes,
        ]

    logger.info("Command: %s", " ".join(cmd))

    # ------------------------------------------------------------------
    # 3. Execute (LIVE OUTPUT)
    # ------------------------------------------------------------------
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None

        for line in process.stdout:
            logger.info("[nnUNet] %s", line.rstrip())

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                cmd,
            )

        logger.info("nnU-Net preprocessing completed successfully.")

        # ------------------------------------------------------------------
        # 4. Backup plans file
        # ------------------------------------------------------------------
        if not args.verify_only:
            _copy_plans_file(config, log_dir, logger)

        return 0

    except FileNotFoundError:
        logger.error(
            "nnUNetv2_plan_and_preprocess not found. "
            "Install nnU-Net v2 first."
        )
        return 1

    except subprocess.CalledProcessError as exc:
        logger.error(
            "nnU-Net preprocessing failed (exit code %d).",
            exc.returncode,
        )
        return 1

    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())