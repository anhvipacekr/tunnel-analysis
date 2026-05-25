'Shared imports, constants, and utilities.'
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("MPLBACKEND", "QtAgg")
os.environ["VTK_TK_WIDGET_PATH"] = ""
os.environ["VTK_DISABLE_TK_WIDGET"] = "1"

import numpy as np

# ------------------------------------------------------------------------------
try:
    import laspy
except ImportError:
    laspy = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    import pyvista as pv
    if hasattr(pv, "set_qt_api"):
        try:
            pv.set_qt_api("pyside6")
        except Exception:
            pass
except ImportError:
    pv = None

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from pyvistaqt import QtInteractor
except ImportError as _exc:
    raise SystemExit(
        "PySide6 / pyvistaqt required.\n"
        "pip install PySide6 pyvista pyvistaqt vtk laspy open3d scipy matplotlib"
    ) from _exc

try:
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

TUNNEL_PROFILES = ["Circle", "Box", "Box 2-cell", "U-type"]

VL_BOX_W  = 3.0   
VL_BOX_H  = 4.5   
VL_CIR_R  = 4.0   

# Light theme palette
_BG   = "#FFFFFF"
_FG   = "#111827"
_GRID = "#E2E8F0"
_ACC1 = "#1D4ED8"
_ACC2 = "#047857"
_ACC3 = "#C2410C"
_RED  = "#DC2626"
_YEL  = "#D97706"
_GRN  = "#047857"
_DIM  = "#475569"   


# ------------------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------------------

def _unit(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        if fallback is not None:
            return fallback
        raise ValueError(f"Cannot normalise near-zero vector: {v}")
    return v / n

def validate_xyz(pts: np.ndarray, name: str = "points") -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be Nx3, got {arr.shape}.")
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        raise ValueError(f"{name}: no finite points.")
    return arr

def _normalize_rgb(colors: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if colors is None: return None
    arr = np.asarray(colors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3: return None
    cmax = float(np.nanmax(arr))
    if cmax > 255.0: return np.clip(arr / 65535.0, 0.0, 1.0).astype(np.float32)
    if cmax > 1.0: return np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)

def make_vertex_cloud(
    pts: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    colors_raw: Optional[np.ndarray] = None,
) -> "pv.PolyData":
    if pv is None: raise RuntimeError("PyVista not installed.")
    xyz = np.ascontiguousarray(validate_xyz(pts)[:, :3], dtype=np.float64)
    n = len(xyz)
    cloud = pv.PolyData()
    cloud.points = xyz
    verts = np.empty(n * 2, dtype=np.int64)
    verts[0::2] = 1
    verts[1::2] = np.arange(n, dtype=np.int64)
    cloud.verts = verts
    if intensity is not None:
        vals = np.asarray(intensity, dtype=np.float64).ravel()
        if len(vals) == n: cloud["Intensity"] = vals
    rgb = _normalize_rgb(colors_raw)
    if rgb is not None and len(rgb) == n:
        cloud["RGB"] = (rgb * 255).astype(np.uint8)
    return cloud


__all__ = [
    "json", "math", "os", "sys", "warnings",
    "dataclass", "field", "Path", "Callable", "Dict", "List", "Optional", "Tuple",
    "np", "laspy", "o3d", "cKDTree", "pv", "QtCore", "QtGui", "QtWidgets", "QtInteractor",
    "matplotlib", "plt", "mpatches", "FigureCanvas", "Figure", "_MPL_OK",
    "TUNNEL_PROFILES", "VL_BOX_W", "VL_BOX_H", "VL_CIR_R",
    "_BG", "_FG", "_GRID", "_ACC1", "_ACC2", "_ACC3", "_RED", "_YEL", "_GRN", "_DIM",
    "_unit", "validate_xyz", "_normalize_rgb", "make_vertex_cloud",
]


