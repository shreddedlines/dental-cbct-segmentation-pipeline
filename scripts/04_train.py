#!/usr/bin/env python3
"""Launch nnU-Net v2 training for ToothFairy3 dental CBCT segmentation.

Sets nnU-Net environment variables and invokes ``nnUNetv2_train``.
Supports fold selection, continue-training, and dry-run modes.

Usage
-----
    # Standard training (fold 0):
    python scripts/04_train.py --config configs/pipeline_config.yaml

    # Continue interrupted training:
    python scripts/04_train.py --config configs/pipeline_config.yaml --continue-training

    # Specific fold:
    python scripts/04_train.py --config configs/pipeline_config.yaml --fold 2

    # Dry run – print the command without executing:
    python scripts/04_train.py --config configs/pipeline_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
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
    log_file = log_dir / "04_train.log"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


def _build_command(config, fold: int, continue_training: bool) -> list[str]:
    """Build the ``nnUNetv2_train`` command list.

    Parameters
    ----------
    config : PipelineConfig
        Resolved pipeline configuration.
    fold : int
        Cross-validation fold index (0-4, or ``all``).
    continue_training : bool
        If ``True``, append ``--c`` to resume from the latest checkpoint.

    Returns
    -------
    list[str]
        Command tokens ready for ``subprocess.run``.
    """
    cmd = [
        "nnUNetv2_train",
        str(config.dataset.id),           # dataset ID, e.g. 100
        config.nnunet.configuration,      # e.g. 3d_fullres
        str(fold),                        # fold index
        "--npz",                          # save softmax predictions
        "-tr", config.nnunet.trainer,     # e.g. nnUNetTrainer
        "-p", config.nnunet.plans,        # e.g. nnUNetPlans
    ]

    if continue_training:
        cmd.append("--c")

    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nnU-Net v2 on the ToothFairy3 dental CBCT dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/04_train.py --config configs/pipeline_config.yaml\n"
            "  python scripts/04_train.py --config configs/pipeline_config.yaml --fold 2\n"
            "  python scripts/04_train.py --config configs/pipeline_config.yaml --continue-training\n"
            "  python scripts/04_train.py --config configs/pipeline_config.yaml --dry-run\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML pipeline configuration file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold to train (default: 0).",
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        default=False,
        help="Continue training from the latest checkpoint (adds --c flag).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the training command without executing it.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry-point for the training script.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    args = _parse_args()
    config = load_config(args.config)
    log_dir = Path(config.paths.logs)

    _setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 1. Set nnU-Net environment variables
    # ------------------------------------------------------------------
    setup = NNUNetSetup(config)
    setup.set_env_vars()
    logger.info("nnU-Net environment variables set.")

    # ------------------------------------------------------------------
    # 2. Determine training parameters
    # ------------------------------------------------------------------
    # Honour CLI --continue-training flag, falling back to config value.
    continue_training = args.continue_training or config.training.continue_training
    fold = args.fold

    cmd = _build_command(config, fold=fold, continue_training=continue_training)
    cmd_str = " ".join(cmd)

    logger.info("=" * 60)
    logger.info("nnU-Net training command:")
    logger.info("  %s", cmd_str)
    logger.info("=" * 60)
    logger.info("  Dataset:          %s (ID %d)", config.dataset.name, config.dataset.id)
    logger.info("  Configuration:    %s", config.nnunet.configuration)
    logger.info("  Trainer:          %s", config.nnunet.trainer)
    logger.info("  Plans:            %s", config.nnunet.plans)
    logger.info("  Fold:             %d", fold)
    logger.info("  Continue:         %s", continue_training)
    logger.info("  Epochs (config):  %d", config.training.epochs)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 3. Dry run – just print the command and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("Dry-run mode – command NOT executed.")
        print(f"\n{cmd_str}\n")
        return 0

    # ------------------------------------------------------------------
    # 4. Execute training
    # ------------------------------------------------------------------
    try:
        logger.info("Starting training …  (Ctrl+C to interrupt)")

        # Training can run for days; stream output directly to the terminal
        # instead of buffering it in memory (stdout=PIPE would be impractical
        # for long-running commands).
        result = subprocess.run(
            cmd,
            check=True,
        )

        logger.info("Training completed successfully (exit code %d).", result.returncode)
        return 0

    except FileNotFoundError:
        logger.error(
            "nnUNetv2_train not found. "
            "Ensure nnU-Net v2 is installed: pip install nnunetv2"
        )
        return 1

    except subprocess.CalledProcessError as exc:
        logger.error(
            "Training failed with exit code %d.", exc.returncode
        )
        logger.error(
            "To resume, run:\n"
            "  python scripts/04_train.py --config %s --fold %d --continue-training",
            args.config,
            fold,
        )
        return 1

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("=" * 60)
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.warning("=" * 60)
        logger.warning(
            "nnU-Net saves checkpoints periodically.  To resume training, run:"
        )
        logger.warning(
            "  python scripts/04_train.py --config %s --fold %d --continue-training",
            args.config,
            fold,
        )
        logger.warning("=" * 60)
        return 1

    except Exception as exc:
        logger.exception("Unexpected error during training: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
