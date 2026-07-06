"""
Interactive visualization for dental CBCT volumes and segmentation masks.

Provides multi-planar slice viewers (axial, coronal, sagittal) with
segmentation overlay, 3D surface rendering via marching cubes, and
static screenshot export.

Usage
-----
    from dental_pipeline.visualization import VolumeVisualizer
    from dental_pipeline.config import load_config

    config = load_config("configs/pipeline_config.yaml")
    viz = VolumeVisualizer(config)
    viz.generate_visualizations(image_dir, pred_dir, output_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure

from dental_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# ── Label categories for grouping in visualizations ──────────────────────
LABEL_CATEGORIES: Dict[str, List[int]] = {
    "Anatomical": [1, 2, 3, 4, 5, 6, 7],
    "Restorations": [8, 9, 10],
    "Teeth": [
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    ],
    "Canals": [103, 104, 105],
    "Pulps": [
        111, 112, 113, 114, 115, 116, 117, 118,
        121, 122, 123, 124, 125, 126, 127, 128,
        131, 132, 133, 134, 135, 136, 137, 138,
        141, 142, 143, 144, 145, 146, 147, 148,
    ],
}

# Structures worth rendering in 3D (large enough to produce meshes)
_3D_RENDER_LABELS: List[int] = [1, 2, 5, 6, 7, 8, 9, 10] + LABEL_CATEGORIES["Teeth"]


def _generate_colormap(n: int) -> Dict[int, str]:
    """Return *n* distinct RGBA colour strings keyed by integer index."""
    cmap = plt.cm.get_cmap("tab20", max(n, 20))
    return {
        i: f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]:.2f})"
        for i, c in enumerate(cmap(np.linspace(0, 1, n)))
    }


class VolumeVisualizer:
    """Create interactive Plotly HTML viewers for CBCT + segmentation data."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.label_names = config.get_label_names()
        self.opacity = config.visualization.opacity
        self.num_cases = config.visualization.num_cases

    # ── I/O helpers ──────────────────────────────────────────────────────

    @staticmethod
    def load_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a NIfTI volume and return (data, affine).

        Parameters
        ----------
        path:
            Path to a ``.nii.gz`` file.

        Returns
        -------
        tuple
            (data_array, affine_4x4)
        """
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float32), img.affine

    @staticmethod
    def load_segmentation(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a NIfTI segmentation mask (integer labels)."""
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.int32), img.affine

    # ── Multi-planar slice viewer ────────────────────────────────────────

    def create_slice_viewer(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        case_id: str = "case",
    ) -> go.Figure:
        """Build an interactive Plotly figure with axial/coronal/sagittal views.

        The figure uses a slider to scroll through slices along the axial
        axis and overlays the segmentation with per-class colours.

        Parameters
        ----------
        image:
            3-D image array (X, Y, Z).
        segmentation:
            3-D integer label array matching *image* shape.
        case_id:
            Identifier used in the figure title.

        Returns
        -------
        go.Figure
        """
        # Normalise image to [0, 1] for display
        img_min, img_max = float(np.min(image)), float(np.max(image))
        if img_max - img_min > 0:
            img_norm = (image - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(image)

        # Collect unique foreground labels present in this segmentation
        unique_labels = sorted(set(int(v) for v in np.unique(segmentation)) - {0})
        n_labels = len(unique_labels)
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        colors = _generate_colormap(max(n_labels, 1))

        # Build colour-mapped RGBA overlay (for the mid-slice preview)
        mid_z = image.shape[2] // 2
        mid_y = image.shape[1] // 2
        mid_x = image.shape[0] // 2

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Axial", "Coronal", "Sagittal"),
            horizontal_spacing=0.04,
        )

        # --- Axial slice (fixed mid_z, slider changes it) ---
        axial_img = img_norm[:, :, mid_z].T
        fig.add_trace(
            go.Heatmap(
                z=axial_img, colorscale="Gray", showscale=False,
                name="CBCT Axial",
            ),
            row=1, col=1,
        )

        # --- Coronal slice ---
        coronal_img = img_norm[:, mid_y, :].T
        fig.add_trace(
            go.Heatmap(
                z=coronal_img, colorscale="Gray", showscale=False,
                name="CBCT Coronal",
            ),
            row=1, col=2,
        )

        # --- Sagittal slice ---
        sagittal_img = img_norm[mid_x, :, :].T
        fig.add_trace(
            go.Heatmap(
                z=sagittal_img, colorscale="Gray", showscale=False,
                name="CBCT Sagittal",
            ),
            row=1, col=3,
        )

        # Overlay segmentation contours on the axial slice
        seg_axial = segmentation[:, :, mid_z].T
        for lbl in unique_labels:
            mask = (seg_axial == lbl).astype(np.uint8)
            if mask.sum() == 0:
                continue
            name = self.label_names.get(lbl, f"Label {lbl}")
            colour = colors.get(label_to_idx[lbl], "rgba(255,0,0,0.5)")
            # Represent as a semi-transparent heatmap layer
            fig.add_trace(
                go.Heatmap(
                    z=mask * (label_to_idx[lbl] + 1),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, colour]],
                    showscale=False,
                    opacity=self.opacity,
                    name=name,
                    visible="legendonly",
                ),
                row=1, col=1,
            )

        # Slider for axial slices
        n_slices = image.shape[2]
        steps = []
        for i in range(0, n_slices, max(1, n_slices // 50)):
            step = {
                "method": "restyle",
                "args": [{"z": [img_norm[:, :, i].T]}, [0]],
                "label": str(i),
            }
            steps.append(step)

        fig.update_layout(
            sliders=[{
                "active": len(steps) // 2,
                "currentvalue": {"prefix": "Axial Slice: "},
                "steps": steps,
                "pad": {"t": 50},
            }],
            title=f"Multi-Planar Viewer — {case_id}",
            height=600,
            width=1600,
            showlegend=True,
            legend=dict(x=1.02, y=1, font=dict(size=9)),
        )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1)

        return fig

    # ── 3-D Surface rendering ────────────────────────────────────────────

    def create_3d_surface(
        self,
        segmentation: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        labels_to_render: Optional[List[int]] = None,
    ) -> go.Figure:
        """Create a 3-D mesh surface from selected segmentation labels.

        Uses *scikit-image* marching cubes to extract isosurfaces for each
        label. Large labels (jawbones, teeth) are included by default;
        tiny structures (pulps, canals) are skipped for performance.

        Parameters
        ----------
        segmentation:
            3-D integer label array.
        spacing:
            Voxel spacing in mm (x, y, z).
        labels_to_render:
            Subset of label IDs to render; defaults to ``_3D_RENDER_LABELS``.

        Returns
        -------
        go.Figure
        """
        if labels_to_render is None:
            labels_to_render = _3D_RENDER_LABELS

        present_labels = set(int(v) for v in np.unique(segmentation)) - {0}
        render_labels = sorted(present_labels & set(labels_to_render))

        n = len(render_labels)
        colors = _generate_colormap(max(n, 1))

        fig = go.Figure()

        for idx, lbl in enumerate(render_labels):
            mask = (segmentation == lbl).astype(np.float32)

            # Skip labels with very few voxels — marching cubes will fail
            if mask.sum() < 50:
                continue

            try:
                verts, faces, normals, _ = measure.marching_cubes(
                    mask, level=0.5, spacing=spacing, step_size=2,
                )
            except Exception:
                logger.debug("Marching cubes failed for label %d — skipping", lbl)
                continue

            x, y, z = verts.T
            i, j, k = faces.T
            name = self.label_names.get(lbl, f"Label {lbl}")
            colour = colors.get(idx, "rgba(100,100,255,0.6)")

            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    name=name,
                    opacity=0.5,
                    color=colour.replace("rgba", "rgb").rsplit(",", 1)[0] + ")",
                    showlegend=True,
                    visible=True,
                )
            )

        fig.update_layout(
            title="3-D Segmentation Surface",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode="data",
            ),
            height=800,
            width=1000,
            legend=dict(x=1.02, y=1, font=dict(size=9)),
        )

        # Add buttons to toggle all / none
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.0, y=1.15,
                    buttons=[
                        dict(
                            label="Show All",
                            method="restyle",
                            args=[{"visible": True}],
                        ),
                        dict(
                            label="Hide All",
                            method="restyle",
                            args=[{"visible": "legendonly"}],
                        ),
                    ],
                )
            ]
        )

        return fig

    # ── Overlay figure (matplotlib, for report screenshots) ──────────────

    def create_overlay_screenshot(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        output_path: Path,
        slice_idx: Optional[int] = None,
    ) -> Path:
        """Save a multi-panel PNG showing CBCT, segmentation, and overlay.

        Parameters
        ----------
        image:
            3-D CBCT volume.
        segmentation:
            3-D integer label map.
        output_path:
            Destination ``.png`` file.
        slice_idx:
            Axial slice index; defaults to the volume midpoint.

        Returns
        -------
        Path
            Path to the saved PNG.
        """
        if slice_idx is None:
            slice_idx = image.shape[2] // 2

        img_slice = image[:, :, slice_idx].T
        seg_slice = segmentation[:, :, slice_idx].T

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=self.config.visualization.dpi)

        # Panel 1: CBCT
        axes[0].imshow(img_slice, cmap="gray")
        axes[0].set_title("CBCT Volume", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Segmentation
        seg_display = np.zeros((*seg_slice.shape, 3), dtype=np.float32)
        unique_labels = sorted(set(int(v) for v in np.unique(seg_slice)) - {0})
        cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
        for i, lbl in enumerate(unique_labels):
            c = cmap(i / max(len(unique_labels), 1))
            seg_display[seg_slice == lbl] = c[:3]

        axes[1].imshow(seg_display)
        axes[1].set_title("Segmentation", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Overlay
        axes[2].imshow(img_slice, cmap="gray")
        overlay = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
        for i, lbl in enumerate(unique_labels):
            c = cmap(i / max(len(unique_labels), 1))
            mask = seg_slice == lbl
            overlay[mask] = (*c[:3], 0.4)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=14, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight", dpi=self.config.visualization.dpi)
        plt.close(fig)

        logger.info("Overlay screenshot saved → %s", output_path)
        return output_path

    # ── Save helpers ─────────────────────────────────────────────────────

    @staticmethod
    def save_html(fig: go.Figure, output_path: Path) -> Path:
        """Write a Plotly figure to a self-contained HTML file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info("HTML saved → %s", output_path)
        return output_path

    @staticmethod
    def save_screenshot(fig: go.Figure, output_path: Path) -> Path:
        """Save a static PNG of a Plotly figure.

        Falls back to HTML-only if ``kaleido`` is not installed.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(output_path), width=1400, height=800, scale=2)
            logger.info("Screenshot saved → %s", output_path)
        except Exception as exc:
            logger.warning(
                "Could not save static image (%s). "
                "Install kaleido: pip install kaleido",
                exc,
            )
            # Fall back: save as HTML instead
            html_path = output_path.with_suffix(".html")
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            logger.info("Fallback HTML saved → %s", html_path)
            return html_path
        return output_path

    # ── Batch generation ─────────────────────────────────────────────────

    def generate_visualizations(
        self,
        image_dir: Path,
        pred_dir: Path,
        output_dir: Path,
        num_cases: Optional[int] = None,
        case_ids: Optional[List[str]] = None,
    ) -> List[Path]:
        """Generate interactive HTML viewers and screenshots for selected cases.

        Parameters
        ----------
        image_dir:
            Directory containing ``*_0000.nii.gz`` image files.
        pred_dir:
            Directory containing predicted segmentation ``.nii.gz`` files.
        output_dir:
            Where to write HTML + PNG outputs.
        num_cases:
            Number of cases to visualise (picked evenly across the list).
        case_ids:
            Explicit list of case identifiers to visualise.

        Returns
        -------
        list[Path]
            Paths to all generated HTML files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if num_cases is None:
            num_cases = self.num_cases

        # Discover available prediction files
        pred_files = sorted(Path(pred_dir).glob("*.nii.gz"))
        if not pred_files:
            logger.warning("No prediction files found in %s", pred_dir)
            return []

        # Filter to requested case IDs if specified
        if case_ids:
            pred_files = [
                p for p in pred_files
                if any(cid in p.name for cid in case_ids)
            ]

        # Sub-sample evenly
        if len(pred_files) > num_cases:
            indices = np.linspace(0, len(pred_files) - 1, num_cases, dtype=int)
            pred_files = [pred_files[i] for i in indices]

        generated: List[Path] = []

        for pred_path in pred_files:
            case_name = pred_path.name.replace(".nii.gz", "")
            img_name = case_name + "_0000.nii.gz"
            img_path = Path(image_dir) / img_name

            if not img_path.is_file():
                # Try without _0000 suffix
                alt_name = case_name.rsplit("_0000", 1)[0] + "_0000.nii.gz"
                img_path = Path(image_dir) / alt_name
                if not img_path.is_file():
                    logger.warning("Image not found for %s — skipping", case_name)
                    continue

            logger.info("Generating visualisation for %s", case_name)

            image, _ = self.load_volume(img_path)
            seg, seg_affine = self.load_segmentation(pred_path)

            # Extract spacing from affine
            spacing = tuple(float(abs(seg_affine[i, i])) for i in range(3))

            # 1. Multi-planar slice viewer
            slice_fig = self.create_slice_viewer(image, seg, case_id=case_name)
            html_path = output_dir / f"{case_name}_slices.html"
            self.save_html(slice_fig, html_path)
            generated.append(html_path)

            # 2. 3-D surface rendering
            surface_fig = self.create_3d_surface(seg, spacing=spacing)
            surface_html = output_dir / f"{case_name}_3d.html"
            self.save_html(surface_fig, surface_html)
            generated.append(surface_html)

            # 3. Static overlay screenshot (for report)
            screenshot_path = output_dir / f"{case_name}_overlay.png"
            self.create_overlay_screenshot(image, seg, screenshot_path)
            generated.append(screenshot_path)

        logger.info(
            "Visualisation complete — %d artefacts generated in %s",
            len(generated), output_dir,
        )
        return generated
