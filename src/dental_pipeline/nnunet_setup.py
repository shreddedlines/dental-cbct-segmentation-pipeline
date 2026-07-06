"""nnU-Net v2 environment setup for ToothFairy3 dental CBCT segmentation.

Creates the required directory structure, symlinks the read-only image data,
remaps label masks from non-contiguous FDI IDs to consecutive nnU-Net IDs,
generates a compatible dataset.json, sets environment variables, and verifies
the installation.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict

from dental_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class NNUNetSetup:
    """Prepares and verifies the nnU-Net v2 runtime environment.

    Responsibilities
    ----------------
    1. Create the three canonical nnU-Net directories
       (``nnUNet_raw``, ``nnUNet_preprocessed``, ``nnUNet_results``).
    2. Symlink the read-only ``imagesTr`` directory into the raw directory.
    3. **Remap** ``labelsTr`` from non-contiguous FDI IDs to consecutive
       nnU-Net-compatible IDs (writes real files, not symlinks).
    4. Generate a new ``dataset.json`` with consecutive labels.
    5. Expose the correct ``nnUNet_*`` environment variables.
    6. Verify that symlinks resolve and the nnU-Net CLI tools are on PATH.

    Parameters
    ----------
    config : PipelineConfig
        Resolved pipeline configuration.
    """

    # nnU-Net CLI commands that must be reachable.
    _REQUIRED_COMMANDS = (
        "nnUNetv2_plan_and_preprocess",
        "nnUNetv2_train",
        "nnUNetv2_predict",
    )

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dataset_name: str = config.dataset.name  # e.g. Dataset100_ToothFairy3
        self.dataset_src: Path = Path(config.paths.dataset_root)
        self.nnunet_raw: Path = Path(config.paths.nnunet_raw)
        self.nnunet_preprocessed: Path = Path(config.paths.nnunet_preprocessed)
        self.nnunet_results: Path = Path(config.paths.nnunet_results)
        self.dataset_dst: Path = self.nnunet_raw / self.dataset_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Orchestrate the full nnU-Net setup sequence.

        This now includes automatic label remapping from non-contiguous
        FDI label IDs to consecutive nnU-Net-compatible IDs.
        """
        logger.info("=" * 60)
        logger.info("Starting nnU-Net environment setup")
        logger.info("=" * 60)

        self.setup_directory_structure()
        self.create_image_symlinks()
        self.remap_labels()
        self.set_env_vars()
        self.verify_setup()

        logger.info("=" * 60)
        logger.info("nnU-Net environment setup completed successfully")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Directory structure
    # ------------------------------------------------------------------

    def setup_directory_structure(self) -> None:
        """Create the three canonical nnU-Net directories if they do not exist."""
        for label, directory in [
            ("nnUNet_raw", self.nnunet_raw),
            ("nnUNet_preprocessed", self.nnunet_preprocessed),
            ("nnUNet_results", self.nnunet_results),
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info("Directory ready: %s -> %s", label, directory)

    # ------------------------------------------------------------------
    # Image symlinks (read-only, no modification needed)
    # ------------------------------------------------------------------

    def create_image_symlinks(self) -> None:
        """Symlink imagesTr from the read-only dataset.

        Only ``imagesTr`` is symlinked because images do not need remapping.
        Labels and dataset.json are handled by :meth:`remap_labels`.

        Raises
        ------
        FileNotFoundError
            If the source dataset or imagesTr is missing.
        """
        if not self.dataset_src.is_dir():
            raise FileNotFoundError(
                f"Dataset source directory does not exist: {self.dataset_src}"
            )

        self.dataset_dst.mkdir(parents=True, exist_ok=True)

        src = self.dataset_src / "imagesTr"
        dst = self.dataset_dst / "imagesTr"

        if not src.exists():
            raise FileNotFoundError(f"imagesTr not found: {src}")

        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                logger.info("  Symlink already correct: imagesTr")
                return
            # Broken or wrong target — remove and re-create.
            logger.warning("  Replacing stale/broken symlink: imagesTr")
            dst.unlink()
        elif dst.exists():
            logger.warning(
                "  Real directory already exists at %s — skipping "
                "symlink creation. Remove it if you want a symlink.",
                dst,
            )
            return

        os.symlink(src, dst, target_is_directory=True)
        logger.info("  Symlinked: imagesTr -> %s", src)

    # ------------------------------------------------------------------
    # Label remapping
    # ------------------------------------------------------------------

    def remap_labels(self) -> None:
        """Remap all label masks and generate a consecutive dataset.json.

        This is the critical step that makes nnU-Net v2 happy. It:
          1. Reads the original dataset.json to discover label IDs.
          2. Builds a mapping from non-contiguous → consecutive IDs.
          3. Remaps every labelsTr/*.nii.gz file (in parallel).
          4. Writes a new dataset.json with consecutive IDs.
          5. Saves label_mapping.json for reverse mapping.
        """
        # Lazy import to avoid circular dependencies
        from dental_pipeline.label_remapping import LabelRemapper

        labels_dst = self.dataset_dst / "labelsTr"

        # Check if remapping was already completed
        mapping_file = self.dataset_dst / "label_mapping.json"
        dj_file = self.dataset_dst / "dataset.json"

        if (
            mapping_file.is_file()
            and dj_file.is_file()
            and not dj_file.is_symlink()
            and labels_dst.is_dir()
            and any(labels_dst.glob("*.nii.gz"))
        ):
            logger.info(
                "Label remapping already completed (found label_mapping.json "
                "and remapped labelsTr). Skipping."
            )
            return

        # Remove any leftover symlinks for labelsTr and dataset.json
        for item_name in ("labelsTr", "dataset.json"):
            item_path = self.dataset_dst / item_name
            if item_path.is_symlink():
                item_path.unlink()
                logger.info("  Removed stale symlink: %s", item_name)

        # Run the full remapping pipeline
        remapper = LabelRemapper(self.config)
        remapper.run()

    # ------------------------------------------------------------------
    # Environment variables
    # ------------------------------------------------------------------

    def get_env_vars(self) -> Dict[str, str]:
        """Return the environment-variable dict required by nnU-Net v2.

        Returns
        -------
        dict[str, str]
            Keys: ``nnUNet_raw``, ``nnUNet_preprocessed``, ``nnUNet_results``.
        """
        return {
            "nnUNet_raw": str(self.nnunet_raw.resolve()),
            "nnUNet_preprocessed": str(self.nnunet_preprocessed.resolve()),
            "nnUNet_results": str(self.nnunet_results.resolve()),
        }

    def set_env_vars(self) -> None:
        """Inject nnU-Net environment variables into the current process."""
        env_vars = self.get_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info("Set env: %s=%s", key, value)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_setup(self) -> None:
        """Verify that the setup is complete and nnU-Net is ready.

        Checks:
        1. imagesTr symlink resolves.
        2. labelsTr directory contains remapped files.
        3. dataset.json exists and is a real file (not a symlink).
        4. label_mapping.json exists.
        5. nnU-Net CLI commands are on PATH.
        6. Environment variables are set.

        Raises
        ------
        RuntimeError
            If any check fails.
        """
        errors: list[str] = []

        # 1. imagesTr
        images_link = self.dataset_dst / "imagesTr"
        if not images_link.exists():
            errors.append(f"imagesTr does not exist: {images_link}")
        else:
            logger.info("  ✓ imagesTr OK: %s", images_link)

        # 2. labelsTr (remapped, real files)
        labels_dir = self.dataset_dst / "labelsTr"
        if not labels_dir.is_dir():
            errors.append(f"labelsTr directory missing: {labels_dir}")
        else:
            n_files = len(list(labels_dir.glob("*.nii.gz")))
            if n_files == 0:
                errors.append(f"No .nii.gz files in {labels_dir}")
            else:
                logger.info("  ✓ labelsTr OK: %d remapped files", n_files)

        # 3. dataset.json (must be real file, not symlink)
        dj = self.dataset_dst / "dataset.json"
        if dj.is_symlink():
            errors.append(
                f"dataset.json is still a symlink — remapping did not run: {dj}"
            )
        elif not dj.is_file():
            errors.append(f"dataset.json missing: {dj}")
        else:
            logger.info("  ✓ dataset.json OK (consecutive labels)")

        # 4. label_mapping.json
        mapping = self.dataset_dst / "label_mapping.json"
        if not mapping.is_file():
            errors.append(f"label_mapping.json missing: {mapping}")
        else:
            logger.info("  ✓ label_mapping.json OK")

        # 5. CLI commands
        for cmd in self._REQUIRED_COMMANDS:
            location = shutil.which(cmd)
            if location is None:
                errors.append(
                    f"nnU-Net command not found on PATH: {cmd}. "
                    "Ensure nnU-Net v2 is installed (pip install nnunetv2)."
                )
            else:
                logger.info("  ✓ Command found: %s -> %s", cmd, location)

        # 6. Environment variables
        for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            val = os.environ.get(var)
            if val is None:
                errors.append(f"Environment variable not set: {var}")
            else:
                logger.info("  ✓ Env var set: %s=%s", var, val)

        if errors:
            for err in errors:
                logger.error("  ✗ %s", err)
            raise RuntimeError(
                "nnU-Net setup verification failed with "
                f"{len(errors)} error(s). See log for details."
            )

        logger.info("All verification checks passed.")
