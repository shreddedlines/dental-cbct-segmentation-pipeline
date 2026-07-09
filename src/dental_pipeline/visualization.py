"""
Professional 3D visualization for dental CBCT segmentation results.

Generates publication-quality outputs from existing predictions:

* **Interactive HTML** — Plotly viewer with opacity controls, camera presets,
  and per-structure toggle via the legend.
* **Multi-view static PNGs** — Front, Left, Right, Top, Isometric rendered
  via PyVista/VTK offscreen (no Chrome/Chromium required).
* **Structure-specific PNGs** — Teeth-only, Teeth+Restorations,
  Teeth+Jawbone, Complete Anatomy.
* **360° rotation animation** — GIF (always) and MP4 (when ffmpeg available),
  rendered via PyVista offscreen.

Usage
-----
    from dental_pipeline.visualization import ProfessionalVisualizer
    from dental_pipeline.config import load_config

    config = load_config("configs/pipeline_config.yaml")
    viz = ProfessionalVisualizer(config)
    viz.generate_all(seg_path, output_dir)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from skimage import measure

from dental_pipeline.config import LABEL_CATEGORIES, PipelineConfig

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Colour palette — carefully tuned for dental anatomy
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: (plotly_rgb_string, default_opacity, hex_color_for_pyvista)
_STRUCTURE_STYLE: Dict[int, Tuple[str, float, str]] = {
    # ── Jawbones: warm bone tones, semi-transparent ──────────────────
    1: ("rgb(210,180,140)", 0.25, "#D2B48C"),   # Lower Jawbone
    2: ("rgb(222,195,160)", 0.25, "#DEC3A0"),   # Upper Jawbone
    # ── Canals (alveolar): orange / red ──────────────────────────────
    3: ("rgb(255,140,0)",   0.90, "#FF8C00"),   # Left Inf. Alveolar Canal
    4: ("rgb(255,100,0)",   0.90, "#FF6400"),   # Right Inf. Alveolar Canal
    # ── Sinuses: pale green, very transparent ────────────────────────
    5: ("rgb(144,238,144)", 0.12, "#90EE90"),   # Left Maxillary Sinus
    6: ("rgb(152,251,152)", 0.12, "#98FB98"),   # Right Maxillary Sinus
    # ── Pharynx: nearly invisible ────────────────────────────────────
    7: ("rgb(180,180,220)", 0.06, "#B4B4DC"),
    # ── Restorations: metallic tones ─────────────────────────────────
    8: ("rgb(255,215,0)",   0.95, "#FFD700"),   # Bridge — gold
    9: ("rgb(192,192,192)", 0.95, "#C0C0C0"),   # Crown — silver
    10: ("rgb(120,130,140)",0.95, "#78828C"),    # Implant — titanium
    # ── Canals (mandibular/lingual) ──────────────────────────────────
    103: ("rgb(255,80,80)", 0.90, "#FF5050"),
    104: ("rgb(230,60,60)", 0.90, "#E63C3C"),
    105: ("rgb(200,50,50)", 0.90, "#C83232"),
}

# Teeth: white/ivory gradient per quadrant
_TOOTH_IDS = LABEL_CATEGORIES["Teeth"]
_TOOTH_QUADRANT_COLORS = [
    # UR (11-18): cool white
    [("rgb(245,245,250)", 0.95, "#F5F5FA"), ("rgb(240,240,248)", 0.95, "#F0F0F8"),
     ("rgb(235,235,245)", 0.95, "#EBEBF5"), ("rgb(230,232,242)", 0.95, "#E6E8F2"),
     ("rgb(225,228,240)", 0.95, "#E1E4F0"), ("rgb(220,225,238)", 0.95, "#DCE1EE"),
     ("rgb(215,222,235)", 0.95, "#D7DEEB"), ("rgb(210,218,232)", 0.95, "#D2DAE8")],
    # UL (21-28): warm white
    [("rgb(250,245,235)", 0.95, "#FAF5EB"), ("rgb(248,242,230)", 0.95, "#F8F2E6"),
     ("rgb(245,240,225)", 0.95, "#F5F0E1"), ("rgb(242,237,220)", 0.95, "#F2EDDC"),
     ("rgb(240,235,215)", 0.95, "#F0EBD7"), ("rgb(238,232,210)", 0.95, "#EEE8D2"),
     ("rgb(235,230,205)", 0.95, "#EBE6CD"), ("rgb(232,228,200)", 0.95, "#E8E4C8")],
    # LL (31-38): light ivory
    [("rgb(255,250,240)", 0.95, "#FFFAF0"), ("rgb(252,248,235)", 0.95, "#FCF8EB"),
     ("rgb(250,245,230)", 0.95, "#FAF5E6"), ("rgb(248,242,225)", 0.95, "#F8F2E1"),
     ("rgb(245,240,220)", 0.95, "#F5F0DC"), ("rgb(242,238,215)", 0.95, "#F2EED7"),
     ("rgb(240,235,210)", 0.95, "#F0EBD2"), ("rgb(238,232,205)", 0.95, "#EEE8CD")],
    # LR (41-48): pearl
    [("rgb(248,248,255)", 0.95, "#F8F8FF"), ("rgb(244,244,252)", 0.95, "#F4F4FC"),
     ("rgb(240,240,250)", 0.95, "#F0F0FA"), ("rgb(236,238,248)", 0.95, "#ECEEF8"),
     ("rgb(232,235,245)", 0.95, "#E8EBF5"), ("rgb(228,232,242)", 0.95, "#E4E8F2"),
     ("rgb(224,228,240)", 0.95, "#E0E4F0"), ("rgb(220,225,238)", 0.95, "#DCE1EE")],
]
for _qi, _q_colors in enumerate(_TOOTH_QUADRANT_COLORS):
    for _ti, _tc in enumerate(_q_colors):
        _STRUCTURE_STYLE[_TOOTH_IDS[_qi * 8 + _ti]] = _tc

# Pulps: pink / magenta
for _i, _pid in enumerate(LABEL_CATEGORIES["Pulps"]):
    _r = min(220 + (_i % 4) * 8, 255)
    _g = 80 + (_i % 8) * 5
    _b = 120 + (_i % 6) * 10
    _hex = f"#{_r:02X}{_g:02X}{_b:02X}"
    _STRUCTURE_STYLE[_pid] = (f"rgb({_r},{_g},{_b})", 0.85, _hex)


def _get_style(label_id: int) -> Tuple[str, float, str]:
    """Return (plotly_rgb, opacity, hex_color) for a label ID."""
    return _STRUCTURE_STYLE.get(label_id, ("rgb(180,180,180)", 0.5, "#B4B4B4"))


# ═══════════════════════════════════════════════════════════════════════════
# Visualization groups
# ═══════════════════════════════════════════════════════════════════════════

VIZ_GROUPS: Dict[str, Dict[str, Any]] = {
    "teeth_only": {
        "title": "Teeth Only",
        "labels": LABEL_CATEGORIES["Teeth"],
        "opacity_override": {},
        "filename": "teeth_only.png",
    },
    "teeth_restorations": {
        "title": "Teeth + Restorations",
        "labels": LABEL_CATEGORIES["Teeth"] + LABEL_CATEGORIES["Restorations"],
        "opacity_override": {},
        "filename": "teeth_restorations.png",
    },
    "jaw_teeth": {
        "title": "Teeth + Jawbone",
        "labels": LABEL_CATEGORIES["Teeth"] + [1, 2],
        "opacity_override": {1: 0.20, 2: 0.20},
        "filename": "jaw_teeth.png",
    },
    "complete_anatomy": {
        "title": "Complete Anatomy",
        "labels": (
            [1, 2, 3, 4, 5, 6]
            + LABEL_CATEGORIES["Restorations"]
            + LABEL_CATEGORIES["Teeth"]
            + LABEL_CATEGORIES["Canals"]
        ),
        "opacity_override": {1: 0.18, 2: 0.18, 5: 0.10, 6: 0.10},
        "filename": "complete_anatomy.png",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Mesh extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_mesh(
    seg: np.ndarray,
    label_id: int,
    spacing: Tuple[float, float, float],
    step_size: int = 2,
    min_voxels: int = 30,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run marching cubes for a single label and return (vertices, faces)."""
    mask = (seg == label_id).astype(np.float32)
    if mask.sum() < min_voxels:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask, level=0.5, spacing=spacing, step_size=step_size,
        )
        return verts, faces
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Bounds / camera helpers
# ═══════════════════════════════════════════════════════════════════════════

def _compute_bounds(
    meshes: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute tight bounding box from all meshes.

    Returns
    -------
    tuple
        (center, extent, bounds_min, max_extent_value)
    """
    all_verts = np.concatenate([v for v, _ in meshes.values()], axis=0)
    bmin = all_verts.min(axis=0)
    bmax = all_verts.max(axis=0)
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin
    max_ext = float(extent.max())
    return center, extent, bmin, max_ext


def _plotly_camera(
    center,
    max_extent,
    direction,
    up=(0, 0, 1),
):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)

    return dict(
        eye=dict(
            x=float(d[0] * 2.2),
            y=float(d[1] * 2.2),
            z=float(d[2] * 2.2),
        ),
        center=dict(
            x=0,
            y=0,
            z=0,
        ),
        up=dict(
            x=float(up[0]),
            y=float(up[1]),
            z=float(up[2]),
        ),
    )

    



def _make_camera_presets(
    center: np.ndarray, max_extent: float,
) -> Dict[str, Dict[str, Any]]:
    """Create all camera presets relative to the mesh bounding box."""
    return {
        "front":     _plotly_camera(center, max_extent, (0, -1, 0.15)),
        "left":      _plotly_camera(center, max_extent, (-1, 0, 0.15)),
        "right":     _plotly_camera(center, max_extent, (1, 0, 0.15)),
        "top":       _plotly_camera(center, max_extent, (0, -0.15, 1), up=(0, -1, 0)),
        "isometric": _plotly_camera(center, max_extent, (0.7, -0.7, 0.5)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PyVista offscreen renderer
# ═══════════════════════════════════════════════════════════════════════════

def _init_pyvista():
    """Import and configure PyVista for headless offscreen rendering.

    Tries rendering backends in order of preference:

    1. **EGL** — GPU-accelerated, no display needed.  Works on NVIDIA
       servers with ``libEGL.so`` (ships with the driver).
    2. **OSMesa** — CPU software rendering, no display needed.
       Works anywhere ``libOSMesa.so`` is installed.
    3. **Xvfb** — Virtual X framebuffer.  Requires the ``xvfb``
       system package (``apt install xvfb``).

    The selected backend is logged so the user knows what's running.
    """
    import os

    # Ensure VTK does not try to open a real display
    has_display = bool(os.environ.get("DISPLAY"))

    # --- Attempt 1: EGL (GPU, headless) --------------------------------
    if not has_display:
        os.environ.setdefault(
            "VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow"
        )

    import pyvista as pv
    pv.OFF_SCREEN = True
    pv.global_theme.allow_empty_mesh = True

    # Quick probe: try creating a plotter to see if the backend works
    backend_name = os.environ.get("VTK_DEFAULT_OPENGL_WINDOW", "auto")
    try:
        p = pv.Plotter(off_screen=True, window_size=(64, 64))
        p.close()
        logger.info("PyVista offscreen OK — backend: %s", backend_name)
        return pv
    except Exception:
        pass

    # --- Attempt 2: OSMesa (CPU, headless) ------------------------------
    os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
    try:
        import importlib
        importlib.reload(pv)
        pv.OFF_SCREEN = True
        p = pv.Plotter(off_screen=True, window_size=(64, 64))
        p.close()
        logger.info("PyVista offscreen OK — backend: OSMesa")
        return pv
    except Exception:
        pass

    # --- Attempt 3: Xvfb (virtual X display) ----------------------------
    # Reset to default backend
    os.environ.pop("VTK_DEFAULT_OPENGL_WINDOW", None)
    try:
        pv.start_xvfb()
        p = pv.Plotter(off_screen=True, window_size=(64, 64))
        p.close()
        logger.info("PyVista offscreen OK — backend: Xvfb")
        return pv
    except Exception:
        pass

    # --- Fallback: proceed anyway, hope for the best -------------------
    logger.warning(
        "Could not verify any offscreen backend (EGL/OSMesa/Xvfb). "
        "Rendering may fail. On Ubuntu, try: sudo apt install xvfb "
        "OR ensure NVIDIA EGL libs are on LD_LIBRARY_PATH."
    )
    return pv


def _pyvista_render(
    pv,
    meshes: Dict[int, Tuple[np.ndarray, np.ndarray]],
    label_names: Dict[int, str],
    camera_position: Any,
    opacity_override: Optional[Dict[int, float]] = None,
    window_size: Tuple[int, int] = (1920, 1080),
    bg_color: str = "#0F0F19",
    title: str = "",
) -> np.ndarray:
    """Render meshes offscreen with PyVista and return the image as numpy array.

    Parameters
    ----------
    pv : module
        The pyvista module.
    meshes : dict
        ``{label_id: (verts, faces)}``.
    label_names : dict
        ``{label_id: name}``.
    camera_position : list or str
        PyVista camera position specification.
    opacity_override : dict, optional
        Per-label opacity overrides.
    window_size : tuple
        (width, height) in pixels.
    bg_color : str
        Background colour (hex).
    title : str
        Optional title text.

    Returns
    -------
    np.ndarray
        RGB image array.
    """
    if opacity_override is None:
        opacity_override = {}

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(bg_color)

    for lid in sorted(meshes.keys()):
        verts, faces_idx = meshes[lid]
        # PyVista faces format: [n_pts, v0, v1, v2, ...]
        n_faces = faces_idx.shape[0]
        pv_faces = np.column_stack([
            np.full(n_faces, 3, dtype=np.int64),
            faces_idx.astype(np.int64),
        ]).ravel()
        mesh = pv.PolyData(verts.copy(), pv_faces)

        _, default_opacity, hex_color = _get_style(lid)
        opacity = opacity_override.get(lid, default_opacity)
        name = label_names.get(lid, f"Label {lid}")

        plotter.add_mesh(
            mesh,
            color=hex_color,
            opacity=opacity,
            label=name,
            smooth_shading=True,
            show_edges=False,
            specular=0.4,
            specular_power=15,
        )

    if title:
        plotter.add_text(title, position="upper_edge", font_size=14,
                         color="white", shadow=True)

    plotter.camera_position = camera_position
    plotter.enable_anti_aliasing("ssaa")

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def _pyvista_camera_position(
    center: np.ndarray,
    max_extent: float,
    direction: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0, 0, 1),
) -> List:
    """Build a PyVista camera_position list: [eye, focal_point, up]."""
    dist = max_extent * 1.3
    d = np.array(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-9)
    eye = center + d * dist
    return [
        tuple(eye),
        tuple(center),
        up,
    ]


def _make_pv_camera_presets(
    center: np.ndarray, max_extent: float,
) -> Dict[str, List]:
    """Camera presets in PyVista format."""
    return {
        "front":     _pyvista_camera_position(center, max_extent, (0, -1, 0.15)),
        "left":      _pyvista_camera_position(center, max_extent, (-1, 0, 0.15)),
        "right":     _pyvista_camera_position(center, max_extent, (1, 0, 0.15)),
        "top":       _pyvista_camera_position(center, max_extent, (0, -0.15, 1), up=(0, -1, 0)),
        "isometric": _pyvista_camera_position(center, max_extent, (0.7, -0.7, 0.5)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main visualizer class
# ═══════════════════════════════════════════════════════════════════════════

class ProfessionalVisualizer:
    """Generate publication-quality 3D visualizations of segmentation results.

    * Interactive HTML via Plotly (no external binary needed).
    * Static PNGs and animation via PyVista/VTK offscreen rendering
      (no Chrome/Chromium required).

    Parameters
    ----------
    config : PipelineConfig
        Resolved pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.label_names = config.get_label_names()
        self.dpi = config.visualization.dpi

    # ── I/O ──────────────────────────────────────────────────────────────

    @staticmethod
    def load_segmentation(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load a NIfTI segmentation mask and return (data, affine)."""
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.int32), img.affine

    # ── Mesh extraction ──────────────────────────────────────────────────

    def _build_meshes(
        self,
        seg: np.ndarray,
        spacing: Tuple[float, float, float],
        label_subset: Optional[List[int]] = None,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Extract marching-cubes meshes for every present label."""
        present = set(int(v) for v in np.unique(seg)) - {0}
        if label_subset is not None:
            present &= set(label_subset)
        present_sorted = sorted(present)

        logger.info(
            "Extracting meshes for %d structures (spacing=%.2f×%.2f×%.2f mm)…",
            len(present_sorted), *spacing,
        )

        meshes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for lid in present_sorted:
            result = _extract_mesh(seg, lid, spacing)
            if result is not None:
                meshes[lid] = result

        logger.info("  → %d meshes extracted successfully.", len(meshes))
        return meshes

    # ── Plotly interactive HTML ───────────────────────────────────────────

    def _create_plotly_figure(
        self,
        meshes: Dict[int, Tuple[np.ndarray, np.ndarray]],
        title: str = "Dental Segmentation",
        opacity_override: Optional[Dict[int, float]] = None,
        camera: Optional[Dict[str, Any]] = None,
        bg_color: str = "rgb(15, 15, 25)",
    ) -> go.Figure:
        """Build a Plotly Figure from pre-extracted meshes."""
        if opacity_override is None:
            opacity_override = {}

        # Compute tight camera if not provided
        if camera is None:
            center, _, _, max_ext = _compute_bounds(meshes)
            camera = _plotly_camera(center, max_ext, (0.7, -0.7, 0.5))

        # Compute scene axis ranges for tight framing
        center, extent, bmin, max_ext = _compute_bounds(meshes)
        margin = max_ext * 0.15
        bmin_all = np.concatenate([v for v, _ in meshes.values()]).min(axis=0)
        bmax_all = np.concatenate([v for v, _ in meshes.values()]).max(axis=0)

        fig = go.Figure()

        # Group traces by category
        category_for_label: Dict[int, str] = {}
        for cat_name, cat_ids in LABEL_CATEGORIES.items():
            for lid in cat_ids:
                category_for_label[lid] = cat_name

        for lid in sorted(meshes.keys()):
            verts, faces = meshes[lid]
            x = verts[:, 0].astype(float).tolist()
            y = verts[:, 1].astype(float).tolist()
            z = verts[:, 2].astype(float).tolist()

            i = faces[:, 0].astype(int).tolist()
            j = faces[:, 1].astype(int).tolist()
            k = faces[:, 2].astype(int).tolist()


            color_rgb, default_opacity, _ = _get_style(lid)
            opacity = opacity_override.get(lid, default_opacity)
            name = self.label_names.get(lid, f"Label {lid}")
            cat = category_for_label.get(lid, "Other")

            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    color=color_rgb,
                    opacity=opacity,
                    name=name,
                    legendgroup=cat,
                    legendgrouptitle_text=cat,
                    showlegend=True,
                    flatshading=False,
                    lighting=dict(
                        ambient=0.35,
                        diffuse=0.65,
                        specular=0.40,
                        roughness=0.45,
                        fresnel=0.20,
                    ),
                    lightposition=dict(
                        x=1000,
                        y=-1000,
                        z=2000,
                    ),
                    hoverinfo="name",
                )
            )

        # Compute all camera presets for buttons
        cam_presets = _make_camera_presets(center, max_ext)

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color="white", family="Arial"),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor=bg_color,
                aspectmode="data",
                camera=camera,
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            legend=dict(
                font=dict(size=10, color="white"),
                bgcolor="rgba(30,30,50,0.85)",
                bordercolor="rgba(100,100,140,0.5)",
                borderwidth=1,
                groupclick="toggleitem",
                x=1.01, y=1.0,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=900,
            width=1400,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.0, y=1.08,
                    bgcolor="rgba(50,50,80,0.8)",
                    font=dict(color="white", size=11),
                    buttons=[
                        dict(
                            label=f"📷 {name.capitalize()}",
                            method="relayout",
                            args=[{"scene.camera": preset}],
                        )
                        for name, preset in cam_presets.items()
                    ],
                ),
            ],
        )
        return fig

    @staticmethod
    def _save_html(fig: go.Figure, path: Path) -> None:
        """Write a self-contained interactive HTML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=True)
        logger.info("  HTML saved → %s", path)

    # ── PyVista PNG export ───────────────────────────────────────────────

    def export_views(
        self,
        meshes: Dict[int, Tuple[np.ndarray, np.ndarray]],
        output_dir: Path,
        opacity_override: Optional[Dict[int, float]] = None,
    ) -> List[Path]:
        """Export high-resolution PNGs from all camera presets via PyVista."""
        pv = _init_pyvista()
        center, _, _, max_ext = _compute_bounds(meshes)
        pv_presets = _make_pv_camera_presets(center, max_ext)

        generated: List[Path] = []
        for view_name, cam_pos in pv_presets.items():
            logger.info("  Rendering %s view …", view_name)
            img = _pyvista_render(
                pv, meshes, self.label_names, cam_pos,
                opacity_override=opacity_override,
                title=f"{view_name.capitalize()} View",
            )
            png_path = output_dir / f"{view_name}_view.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            _save_image(img, png_path)
            generated.append(png_path)
            logger.info("    PNG saved → %s", png_path)

        return generated

    # ── Structure-specific PNGs ──────────────────────────────────────────

    def export_structure_groups(
        self,
        seg: np.ndarray,
        spacing: Tuple[float, float, float],
        output_dir: Path,
    ) -> List[Path]:
        """Export PNGs for each structure group."""
        pv = _init_pyvista()
        generated: List[Path] = []

        for group_key, group_cfg in VIZ_GROUPS.items():
            logger.info("  Building %s …", group_cfg["title"])
            group_meshes = self._build_meshes(
                seg, spacing, label_subset=group_cfg["labels"],
            )
            if not group_meshes:
                logger.warning("  No meshes for %s — skipping.", group_key)
                continue

            center, _, _, max_ext = _compute_bounds(group_meshes)
            cam_pos = _pyvista_camera_position(center, max_ext, (0.7, -0.7, 0.5))

            img = _pyvista_render(
                pv, group_meshes, self.label_names, cam_pos,
                opacity_override=group_cfg.get("opacity_override", {}),
                title=group_cfg["title"],
            )
            png_path = output_dir / group_cfg["filename"]
            png_path.parent.mkdir(parents=True, exist_ok=True)
            _save_image(img, png_path)
            generated.append(png_path)
            logger.info("    PNG saved → %s", png_path)

        return generated

    # ── 360° rotation animation ──────────────────────────────────────────

    def export_rotation(
        self,
        meshes: Dict[int, Tuple[np.ndarray, np.ndarray]],
        output_dir: Path,
        n_frames: int = 72,
        fps: int = 18,
        opacity_override: Optional[Dict[int, float]] = None,
    ) -> List[Path]:
        """Generate a 360° rotation animation via PyVista offscreen.

        Always produces a GIF. Produces MP4 when imageio-ffmpeg is available.
        """
        try:
            import imageio.v3 as iio
            _V3 = True
        except ImportError:
            try:
                import imageio as iio
                _V3 = False
            except ImportError:
                logger.warning(
                    "imageio not installed — skipping animation. "
                    "Install: pip install imageio imageio-ffmpeg"
                )
                return []

        pv = _init_pyvista()
        center, _, _, max_ext = _compute_bounds(meshes)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: List[np.ndarray] = []

        logger.info("  Rendering %d rotation frames via PyVista …", n_frames)

        dist = max_ext * 1.3
        elev_z = 0.35 * dist

        for i in range(n_frames):
            angle = 2 * math.pi * i / n_frames
            eye = (
                float(center[0] + dist * math.sin(angle)),
                float(center[1] - dist * math.cos(angle)),
                float(center[2] + elev_z),
            )
            cam_pos = [eye, tuple(center), (0, 0, 1)]
            frame = _pyvista_render(
                pv, meshes, self.label_names, cam_pos,
                opacity_override=opacity_override,
                window_size=(960, 720),
            )
            frames.append(frame)

            if (i + 1) % 12 == 0:
                logger.info("    frame %d/%d", i + 1, n_frames)

        generated: List[Path] = []

        # ── GIF ──────────────────────────────────────────────────────
        gif_path = output_dir / "rotation.gif"
        try:
            if _V3:
                iio.imwrite(str(gif_path), frames, extension=".gif",
                            loop=0, duration=int(1000 / fps))
            else:
                iio.mimwrite(str(gif_path), frames,
                             duration=1.0 / fps, loop=0)
            logger.info("  GIF saved → %s", gif_path)
            generated.append(gif_path)
        except Exception as exc:
            logger.warning("  GIF export failed: %s", exc)

        # ── MP4 ──────────────────────────────────────────────────────
        mp4_path = output_dir / "rotation.mp4"
        try:
            if _V3:
                iio.imwrite(str(mp4_path), frames, extension=".mp4",
                            fps=fps, codec="libx264")
            else:
                iio.mimwrite(str(mp4_path), frames, fps=fps)
            logger.info("  MP4 saved → %s", mp4_path)
            generated.append(mp4_path)
        except Exception as exc:
            logger.warning(
                "  MP4 export failed (%s). Install: pip install imageio-ffmpeg", exc
            )

        return generated

    # ── Top-level orchestration ──────────────────────────────────────────

    def generate_all(
        self,
        seg_path: Path,
        output_dir: Path,
        case_id: Optional[str] = None,
    ) -> List[Path]:
        """Generate every visualization artefact for a single case."""
        if case_id is None:
            case_id = seg_path.stem.replace(".nii", "")

        logger.info("═" * 60)
        logger.info("Generating visualizations for: %s", case_id)
        logger.info("═" * 60)

        seg, affine = self.load_segmentation(seg_path)
        spacing = tuple(float(abs(affine[i, i])) for i in range(3))

        all_generated: List[Path] = []

        # Decide which labels to show in the full view
        full_labels = (
            [1, 2, 3, 4, 5, 6]
            + LABEL_CATEGORIES["Restorations"]
            + LABEL_CATEGORIES["Teeth"]
            + LABEL_CATEGORIES["Canals"]
        )
        full_meshes = self._build_meshes(seg, spacing, label_subset=full_labels)

        if not full_meshes:
            logger.error("No meshes could be extracted — nothing to visualise.")
            return []

        full_opacity = {1: 0.20, 2: 0.20, 5: 0.10, 6: 0.10}

        # Compute bounds-based camera
        center, _, _, max_ext = _compute_bounds(full_meshes)
        default_camera = _plotly_camera(center, max_ext, (0.7, -0.7, 0.5))

        # ── 1. Interactive HTML (Plotly) ─────────────────────────────
        logger.info("Creating interactive HTML viewer …")
        interactive_fig = self._create_plotly_figure(
            full_meshes,
            title=f"Dental Segmentation — {case_id}",
            opacity_override=full_opacity,
            camera=default_camera,
        )
        html_path = output_dir / "interactive.html"
        self._save_html(interactive_fig, html_path)
        all_generated.append(html_path)

        # ── 2. Multi-view PNGs (PyVista) ─────────────────────────────
        logger.info("Exporting multi-view PNGs via PyVista …")
        view_pngs = self.export_views(
            full_meshes, output_dir, opacity_override=full_opacity,
        )
        all_generated.extend(view_pngs)

        # ── 3. Structure-specific PNGs (PyVista) ─────────────────────
        logger.info("Exporting structure-specific PNGs …")
        group_pngs = self.export_structure_groups(seg, spacing, output_dir)
        all_generated.extend(group_pngs)

        # ── 4. Rotation animation (PyVista) ──────────────────────────
        logger.info("Generating 360° rotation animation …")
        anim_files = self.export_rotation(
            full_meshes, output_dir, opacity_override=full_opacity,
        )
        all_generated.extend(anim_files)

        logger.info("═" * 60)
        logger.info(
            "Done — %d artefacts for %s in %s",
            len(all_generated), case_id, output_dir,
        )
        logger.info("═" * 60)

        return all_generated

    # ── Batch driver ─────────────────────────────────────────────────────

    def generate_visualizations(
        self,
        pred_dir: Path,
        output_dir: Path,
        num_cases: Optional[int] = None,
        case_ids: Optional[List[str]] = None,
        image_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Generate visualizations for multiple cases."""
        if num_cases is None:
            num_cases = self.config.visualization.num_cases

        pred_files = sorted(Path(pred_dir).glob("*.nii.gz"))
        if not pred_files:
            logger.warning("No .nii.gz files found in %s", pred_dir)
            return []

        if case_ids:
            pred_files = [
                p for p in pred_files
                if any(cid in p.name for cid in case_ids)
            ]

        if len(pred_files) > num_cases:
            indices = np.linspace(0, len(pred_files) - 1, num_cases, dtype=int)
            pred_files = [pred_files[i] for i in indices]

        all_generated: List[Path] = []
        for pred_path in pred_files:
            case_name = pred_path.name.replace(".nii.gz", "")
            case_out = Path(output_dir) / case_name
            generated = self.generate_all(pred_path, case_out, case_id=case_name)
            all_generated.extend(generated)

        return all_generated


# ═══════════════════════════════════════════════════════════════════════════
# Image saving helper (PIL-based, no Chrome needed)
# ═══════════════════════════════════════════════════════════════════════════

def _save_image(img_array: np.ndarray, path: Path) -> None:
    """Save a numpy RGB array as PNG using PIL."""
    from PIL import Image
    Image.fromarray(img_array).save(str(path))


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility alias
# ═══════════════════════════════════════════════════════════════════════════

VolumeVisualizer = ProfessionalVisualizer
