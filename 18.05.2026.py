"""
TunnelApp.py - Tunnel Analysis v4.0 (bugfix r1)
CBNU Smart Structure Lab
==============================
Fixes applied (r1):
  FIX-1: axis= keyword arguments in np.diff / np.linalg.norm throughout
  FIX-2: RANSAC loop: bn initialised to -1, axis=1 on norm calls
  FIX-3: Duplicate LinePlotWidget class removed

5-Layer Architecture + 2D Cross-Section Technical View
  Layer 1 (Base)      : LAS / PLY reader, raw point-cloud I/O
  Layer 2 (Pre.)      : Voxel homogenisation + distance-statistics SOR
  Layer 3 (Geo.)      : Iterative centerline (Yi 2020) + Frenet-Serret frames
                        + SegmentationLayer
  Layer 4 (Param.)    : Settlement, convergence, ovality, eccentricity,
                        polar deformation, 2D cross-section geometry
  Layer 5 (UI)        : PySide6 accordion sidebar, MatplotlibSectionWidget,
                        section navigator, Ollama llama3 hook
"""

from __future__ import annotations

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


# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------

@dataclass
class PointCloudBundle:
    points:     np.ndarray
    intensity:  Optional[np.ndarray] = None
    colors_raw: Optional[np.ndarray] = None
    path:       Optional[str]        = None
    metadata:   Dict[str, object]    = field(default_factory=dict)
    cloud:      Optional[object]     = None

@dataclass
class SectionGeometry:
    chainage:       float = 0.0
    center_3d:      Optional[np.ndarray] = None
    pts_2d:         Optional[np.ndarray] = None   
    labels:         Optional[np.ndarray] = None   
    H1:             float = float("nan")          
    H2:             float = float("nan")          
    H3:             float = float("nan")          
    W1:             float = float("nan")          
    W2:             float = float("nan")          
    C1:             float = float("nan")          
    C2:             float = float("nan")          
    C3:             float = float("nan")          
    wall_angle_L:   float = float("nan")          
    wall_angle_R:   float = float("nan")          
    radius_fit:     float = float("nan")          
    eccentricity:   float = float("nan")          
    ovality:        float = float("nan")          
    clearance_violation: bool = False
    min_clearance_dist:  float = float("nan")

@dataclass
class PipelineContext:
    scans:              List[PointCloudBundle]   = field(default_factory=list)
    active_index:       int                      = -1
    normalized_points:  Optional[np.ndarray]     = None
    registered_points:  Optional[np.ndarray]     = None
    centerline:         Optional[np.ndarray]     = None
    centerline_smooth:  Optional[np.ndarray]     = None
    frenet_frames:      List[Dict[str, np.ndarray]] = field(default_factory=list)
    parameters:         Dict[str, float]         = field(default_factory=dict)
    heatmap_scalars:    Optional[np.ndarray]     = None
    time_series_plot:   Optional[np.ndarray]     = None
    rmse_mm:            Optional[float]          = None
    polar_map:          Optional[np.ndarray]     = None
    polar_angles:       Optional[np.ndarray]     = None
    polar_centers:      Optional[np.ndarray]     = None
    sections:           List[SectionGeometry]    = field(default_factory=list)
    tunnel_profile:     str                      = "Circle"

    @property
    def active_scan(self) -> Optional[PointCloudBundle]:
        if 0 <= self.active_index < len(self.scans):
            return self.scans[self.active_index]
        return None

    @property
    def working_points(self) -> Optional[np.ndarray]:
        if self.registered_points is not None:
            return self.registered_points
        if self.normalized_points is not None:
            return self.normalized_points
        s = self.active_scan
        return None if s is None else s.points


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

PLY_DTYPES = {
    "char":"i1","int8":"i1","uchar":"u1","uint8":"u1",
    "short":"i2","int16":"i2","ushort":"u2","uint16":"u2",
    "int":"i4","int32":"i4","uint":"u4","uint32":"u4",
    "float":"f4","float32":"f4","double":"f8",
}

def _read_las(fp: str) -> PointCloudBundle:
    if laspy is None: raise RuntimeError("laspy not installed.")
    las = laspy.read(fp)
    pts = np.vstack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).T.astype(np.float64)
    intensity = np.asarray(las.intensity, dtype=np.float64) if hasattr(las, "intensity") else None
    colors = None
    if all(hasattr(las, c) for c in ("red", "green", "blue")):
        colors = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64)
    pts = validate_xyz(pts, Path(fp).name)
    cloud = make_vertex_cloud(pts, intensity=intensity, colors_raw=colors)
    return PointCloudBundle(
        points=pts, intensity=intensity, colors_raw=colors, path=fp, cloud=cloud,
        metadata={"format": Path(fp).suffix.lower(), "point_count": int(len(pts)),
                  "bounds_min": pts.min(0).tolist(), "bounds_max": pts.max(0).tolist(),
                  "has_intensity": intensity is not None, "has_colors": colors is not None},
    )

def _read_ply(fp: str) -> PointCloudBundle:
    path = Path(fp)
    with path.open("rb") as fh:
        if fh.readline().strip() != b"ply": raise ValueError(f"Not PLY: {fp}")
        fmt = None; n_v = 0; props: List[Tuple[str, str]] = []; elem = None
        while True:
            raw = fh.readline()
            if not raw: raise ValueError("PLY header truncated.")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header": break
            if not line or line.startswith("comment"): continue
            p = line.split()
            if p[0] == "format": fmt = p[1]
            elif p[0] == "element": elem = p[1]; n_v = int(p[2]) if elem == "vertex" else n_v
            elif p[0] == "property" and elem == "vertex":
                props.append((p[2], p[1]))
        pnames = [nm.lower() for nm, _ in props]
        xyz_i = [pnames.index(a) for a in ("x", "y", "z")]
        ci = None
        for cn in (("red", "green", "blue"), ("r", "g", "b")):
            if all(c in pnames for c in cn): ci = [pnames.index(c) for c in cn]; break
        if fmt == "ascii":
            pts = np.empty((n_v, 3), dtype=np.float64)
            col = np.empty((n_v, 3), dtype=np.float64) if ci else None
            for r in range(n_v):
                vs = fh.readline().decode("ascii", "replace").split()
                pts[r] = [float(vs[i]) for i in xyz_i]
                if col is not None and ci: col[r] = [float(vs[i]) for i in ci]
        else:
            endian = "<" if "little" in (fmt or "") else ">"
            dtype = np.dtype([(f"f{i}_{nm}", endian + PLY_DTYPES[k]) for i, (nm, k) in enumerate(props)])
            table = np.fromfile(fh, dtype=dtype, count=n_v)
            fn = table.dtype.names or ()
            pts = np.column_stack([table[fn[i]] for i in xyz_i]).astype(np.float64)
            col = np.column_stack([table[fn[i]] for i in ci]).astype(np.float64) if ci else None
    pts = validate_xyz(pts, path.name)
    cloud = make_vertex_cloud(pts, colors_raw=col)
    return PointCloudBundle(
        points=pts, colors_raw=col, path=fp, cloud=cloud,
        metadata={"format": ".ply", "point_count": int(len(pts)),
                  "bounds_min": pts.min(0).tolist(), "bounds_max": pts.max(0).tolist(),
                  "has_intensity": False, "has_colors": col is not None},
    )

class BaseLayer:
    def load_scan(self, fp: str) -> PointCloudBundle:
        sfx = Path(fp).suffix.lower()
        if sfx in {".las", ".laz"}: return _read_las(fp)
        if sfx == ".ply": return _read_ply(fp)
        raise ValueError(f"Unsupported format: {sfx}")


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class PreprocessingLayer:
    def voxel_downsample(
        self, context: PipelineContext, voxel_size: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        scan = context.active_scan
        if scan is None: raise RuntimeError("voxel_downsample: no active scan.")
        pts = validate_xyz(scan.points)
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            dn = np.asarray(pcd.voxel_down_sample(float(voxel_size)).points, dtype=np.float64)
        else:
            dn = self._np_voxel(pts, voxel_size)
        dn = validate_xyz(dn, "voxel")
        c  = dn.mean(0)
        return dn - c, c

    @staticmethod
    def _np_voxel(pts: np.ndarray, vs: float) -> np.ndarray:
        pm = pts.min(0)
        cell = np.floor((pts - pm) / vs).astype(np.int64)
        dims = cell.max(0) + 1
        keys = cell[:, 0] + cell[:, 1] * int(dims[0]) + cell[:, 2] * int(dims[0]) * int(dims[1])
        order = np.argsort(keys, kind="stable")
        ks = keys[order]; ps = pts[order]
        _, first, counts = np.unique(ks, return_index=True, return_counts=True)
        cum = np.vstack([np.zeros((1, 3)), np.cumsum(ps, axis=0)])
        order_ends = first + counts
        return ((cum[order_ends] - cum[first]) / counts[:, None]).astype(np.float64)

    def statistical_outlier_removal_run(
        self, context: PipelineContext
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        scan = context.active_scan
        if scan is None: raise RuntimeError("SOR: no active scan.")
        pts = validate_xyz(scan.points); colors = scan.colors_raw; N = len(pts)
        centroid = pts.mean(0); centred = pts - centroid
        ev, vecs = np.linalg.eigh(np.cov(centred.T))
        long_ax = vecs[:, np.argmax(ev)]
        proj = centred @ long_ax
        pmin, pmax = float(proj.min()), float(proj.max())
        ns = max(1, int(np.ceil((pmax - pmin) / 1.0)))
        devs = np.full(N, np.nan, dtype=np.float64)
        for s in range(ns):
            lo = pmin + s; hi = pmin + s + 1
            if s == ns - 1: hi = pmax + 1e-9
            mask = (proj >= lo) & (proj < hi)
            idx  = np.where(mask)[0]
            if len(idx) < 6: devs[idx] = 0.0; continue
            sp = pts[idx]
            ao = centroid + float(proj[idx].mean()) * long_ax
            diff = sp - ao
            ax_c = (diff @ long_ax)[:, None] * long_ax
            ri   = np.linalg.norm(diff - ax_c, axis=1)
            R    = float(np.median(ri))
            if R < 1e-4: devs[idx] = 0.0; continue
            devs[idx] = ri - R
        fin = devs[np.isfinite(devs)]
        if len(fin) < 2:
            warnings.warn("SOR: insufficient data.")
            return pts, _normalize_rgb(colors) if colors is not None else None
        mu, si = float(np.mean(fin)), float(np.std(fin, ddof=1))
        k = 2.5
        inlier = np.isfinite(devs) & (devs >= mu - k * si) & (devs <= mu + k * si)
        cleaned = validate_xyz(pts[inlier])
        cout: Optional[np.ndarray] = None
        if colors is not None:
            raw = np.asarray(colors, dtype=np.float64)
            if raw.shape[0] == N: cout = _normalize_rgb(raw[inlier])
        return cleaned, cout

    def extract_tunnel_lining(self, context: PipelineContext) -> np.ndarray:
        pts = context.working_points
        if pts is None: raise RuntimeError("No working_points.")
        pts = validate_xyz(pts)
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            try:
                _, inl = pcd.segment_plane(0.05, 3, 100)
                mask = np.ones(len(pts), dtype=bool); mask[np.asarray(inl)] = False
                pts = pts[mask]
            except Exception:
                pass
        return pts


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class RegistrationLayer:
    def anchor_translation(self, context: PipelineContext) -> np.ndarray:
        scan = context.active_scan; pts = context.working_points
        if scan is None or pts is None: raise RuntimeError("Load scan first.")
        src = validate_xyz(pts)
        if len(context.scans) < 2: return src
        tgt = validate_xyz(context.scans[0].points)
        return src + (self._anchor(tgt, context.scans[0].intensity) -
                      self._anchor(src, scan.intensity))

    def run_surface_icp(self, context: PipelineContext) -> Tuple[np.ndarray, float]:
        pts = context.working_points
        if pts is None: raise RuntimeError("Run anchor first.")
        src = validate_xyz(pts)
        if len(context.scans) < 2: return src, 0.0
        return self._icp(src, validate_xyz(context.scans[0].points))

    def calculate_rmse(self, context: PipelineContext) -> float:
        pts = context.working_points
        if pts is None or len(context.scans) < 2: return float("nan")
        return self._rmse(validate_xyz(pts), validate_xyz(context.scans[0].points))

    def _anchor(self, pts: np.ndarray, intensity: Optional[np.ndarray]) -> np.ndarray:
        pts = validate_xyz(pts)
        if intensity is not None:
            vals = np.asarray(intensity, dtype=np.float64).ravel()
            if vals.shape[0] == pts.shape[0]:
                fm = np.isfinite(vals)
                if fm.any(): return pts[int(np.argmax(np.where(fm, vals, -np.inf)))].copy()
        est = np.median(pts, axis=0)
        for _ in range(300):
            d = np.linalg.norm(pts - est, axis=1); nz = d > 1e-10
            if not nz.any(): break
            w = 1.0 / d[nz]; new = (w[:, None] * pts[nz]).sum(axis=0) / w.sum()
            if np.linalg.norm(new - est) < 1e-7: est = new; break
            est = new
        return est

    def _icp(self, src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, float]:
        if o3d is not None and len(src) >= 20 and len(tgt) >= 20:
            vs = float(np.clip(np.linalg.norm(np.ptp(tgt, axis=0)) / 600.0, 0.02, 0.12))
            def _pc(p):
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(p); return pc
            sd = _pc(src).voxel_down_sample(vs); td = _pc(tgt).voxel_down_sample(vs)
            nr = o3d.geometry.KDTreeSearchParamHybrid(radius=max(vs * 3, 0.05), max_nn=30)
            for pc in (sd, td):
                pc.estimate_normals(nr)
                pc.orient_normals_consistent_tangent_plane(k=15)
            est  = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            crit = o3d.pipelines.registration.ICPConvergenceCriteria
            r1 = o3d.pipelines.registration.registration_icp(
                sd, td, max(vs * 6, 0.15), np.eye(4), est,
                crit(max_iteration=60, relative_fitness=1e-5, relative_rmse=1e-5))
            r2 = o3d.pipelines.registration.registration_icp(
                sd, td, max(vs * 1.5, 0.004), r1.transformation, est,
                crit(max_iteration=120, relative_fitness=1e-7, relative_rmse=1e-7))
            T = np.asarray(r2.transformation, dtype=np.float64)
            ones = np.ones((src.shape[0], 1))
            reg  = (T @ np.hstack([src, ones]).T).T[:, :3]
            return reg, float(r2.inlier_rmse) * 1000.0
        return src, self._rmse(src, tgt)

    def _rmse(self, src: np.ndarray, tgt: np.ndarray) -> float:
        if cKDTree is None: return float("nan")
        step = max(1, src.shape[0] // 100_000)
        d, _ = cKDTree(tgt).query(src[::step], k=1, workers=-1)
        return float(np.sqrt(np.mean(d ** 2))) * 1000.0


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class GeometricLayer:
    def extract_centerline(
        self, context: PipelineContext, section_count: int = 80
    ) -> Tuple[np.ndarray, List[Dict]]:
        pts = context.working_points
        if pts is None: raise RuntimeError("No working_points.")
        pts = validate_xyz(pts)
        c = pts.mean(axis=0)
        ev, vecs = np.linalg.eigh(np.cov((pts - c).T))
        ax = vecs[:, np.argmax(ev)]
        proj = (pts - c) @ ax
        order = np.argsort(proj)
        chunks = np.array_split(pts[order], section_count)
        centers = [ch.mean(axis=0) for ch in chunks if len(ch) >= 30]
        if len(centers) < 4: raise RuntimeError(f"Only {len(centers)} centers (need >= 4).")
        cl = np.asarray(centers, dtype=np.float64)
        return cl, self._frenet(cl)

    def extract_centerline_iterative(
        self, context: PipelineContext, design_axis: np.ndarray,
        section_count: int = 80, mu: float = 0.03, max_iter: int = 20
    ) -> Tuple[np.ndarray, List[Dict], int]:
        from scipy.interpolate import splev, splprep
        pts = context.working_points
        if pts is None: raise RuntimeError("No working_points.")
        pts = validate_xyz(pts)
        cur = np.asarray(design_axis, dtype=np.float64)
        if cur.ndim != 2 or cur.shape[1] != 3 or len(cur) < 4:
            raise ValueError("design_axis must be (M >= 4, 3).")
        new_ax = cur.copy(); iters = 0
        for it in range(max_iter):
            iters = it + 1; frs = self._frenet(cur); c3d: List[np.ndarray] = []
            for fr in frs:
                C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
                mask = np.abs((pts - C) @ T) < 0.05; sl = pts[mask]
                if len(sl) < 10: continue
                d = sl - C; p2 = np.column_stack([d @ N, d @ B])
                try: c2d, _, _ = self._ransac_circle(p2)
                except Exception: continue
                c3d.append(C + float(c2d[0]) * N + float(c2d[1]) * B)
            if len(c3d) < 4: warnings.warn(f"Iter {iters}: only {len(c3d)} centers."); break
            ca = np.asarray(c3d, dtype=np.float64)
            # FIX-1: axis=
            ch = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ca, axis=0), axis=1))])
            tot = ch[-1]
            if tot < 1e-6: break
            u = ch / tot; _, ui = np.unique(u, return_index=True)
            if len(ui) < 4: break
            try: tck, _ = splprep(ca[ui].T, u=u[ui], s=0, k=3, quiet=True)
            except Exception as e: warnings.warn(f"splprep: {e}"); break
            uf = np.linspace(0, 1, section_count)
            new_ax = np.column_stack(splev(uf, tck)).astype(np.float64)
            # FIX-1: axis=
            chp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cur, axis=0), axis=1))])
            tp  = chp[-1]; e_val = float("inf")
            if tp > 1e-6:
                _, uip = np.unique(chp / tp, return_index=True)
                if len(uip) >= 4:
                    try:
                        tp2, _ = splprep(cur[uip].T, u=(chp / tp)[uip], s=0, k=3, quiet=True)
                        pr2   = np.column_stack(splev(uf, tp2)).astype(np.float64)
                        e_val = float(np.mean(np.linalg.norm(new_ax - pr2, axis=1) ** 2))
                    except Exception: pass
            cur = new_ax
            if e_val < mu: break
        return new_ax, self._frenet(new_ax), iters

    def smooth_bspline(self, cl: np.ndarray, sf: float = 0.5) -> np.ndarray:
        try:
            from scipy.interpolate import splev, splprep
        except ImportError: return np.asarray(cl, dtype=np.float64)
        pts = np.asarray(cl, dtype=np.float64)
        if len(pts) < 4: raise RuntimeError("Need >= 4 pts.")
        # FIX-1: axis=
        delta = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep  = np.concatenate([[True], delta > 1e-10])
        ptsc  = pts[keep]
        if len(ptsc) < 4: return pts
        # FIX-1: axis=
        ch = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ptsc, axis=0), axis=1))])
        tot = ch[-1]
        if tot < 1e-10: return pts
        try: tck, _ = splprep(ptsc.T, u=ch / tot, s=float(np.clip(sf, 0, 1)) * len(ptsc), k=3, quiet=True)
        except Exception: return pts
        return np.column_stack(splev(np.linspace(0, 1, len(ptsc) * 4), tck)).astype(np.float64)

    def generate_frenet_planes(self, fr: List[Dict]) -> List[Dict]:
        return fr

    def _frenet(self, cl: np.ndarray) -> List[Dict]:
        pts = np.asarray(cl, dtype=np.float64)
        n = len(pts)
        if n < 2: raise RuntimeError("Frenet: need >= 2 pts.")
        
        T = self._tangents(pts)
        fT = np.empty((n, 3))
        fN = np.empty((n, 3))
        fB = np.empty((n, 3))
        
        Z_global = np.array([0.0, 0.0, 1.0])
        
        for i in range(n):
            Tc = T[i]
            
            # Avoid degeneracy if the tunnel axis is nearly vertical.
            if abs(Tc[2]) > 0.9999:
                Nx = np.array([1.0, 0.0, 0.0])
            else:
                # Horizontal N axis, aligned with the ground plane and pointing to section right.
                Nx = np.cross(Tc, Z_global)
                Nx = _unit(Nx)
            
            # Vertical B axis, pointing upward toward the tunnel crown.
            Bx = np.cross(Nx, Tc)
            Bx = _unit(Bx)
            
            fT[i] = Tc
            fN[i] = Nx
            fB[i] = Bx
            
        return [{"center": pts[i], "T": fT[i], "N": fN[i], "B": fB[i]} for i in range(n)]
    @staticmethod
    def _tangents(pts: np.ndarray) -> np.ndarray:
        n = len(pts); T = np.empty_like(pts)
        T[1:-1] = pts[2:] - pts[:-2]; T[0] = pts[1] - pts[0]; T[-1] = pts[-1] - pts[-2]
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        tiny = norms.ravel() < 1e-10
        for i in np.where(tiny)[0]: nb = i - 1 if i > 0 else i + 1; T[i] = T[nb]
        norms = np.linalg.norm(T, axis=1, keepdims=True); norms = np.where(norms < 1e-10, 1.0, norms)
        return T / norms

    @staticmethod
    def _perp(t: np.ndarray) -> np.ndarray:
        cands = [np.array([0., 0., 1.]), np.array([0., 1., 0.]), np.array([1., 0., 0.])]
        seed = cands[int(np.argmin([abs(float(c @ t)) for c in cands]))]
        return _unit(seed - (seed @ t) * t)

    def _ransac_circle(
        self, pts2d: np.ndarray, n_iter: int = 200, tol: float = 0.02
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        K = len(pts2d)
        if K < 3: raise ValueError("Need >= 3 pts.")
        bc = pts2d.mean(axis=0)
        # FIX-1 & 2: axis=1 for norm, bn=-1
        br = float(np.median(np.linalg.norm(pts2d - bc, axis=1)))
        bm = np.ones(K, dtype=bool); bn = -1
        rng = np.random.default_rng(42)
        for _ in range(n_iter):
            idx = rng.choice(K, 3, replace=False)
            try: c, r = self._c3(pts2d[idx[0]], pts2d[idx[1]], pts2d[idx[2]])
            except Exception: continue
            # FIX-1: axis=1
            mask = np.abs(np.linalg.norm(pts2d - c, axis=1) - r) < tol
            ni   = int(mask.sum())
            if ni > bn:
                bn = ni; bm = mask
                inl = pts2d[mask]
                if len(inl) >= 3: bc, br = self._lsq_c(inl)
        return bc, br, bm

    @staticmethod
    def _c3(p1, p2, p3):
        ax, ay = p1; bx, by = p2; cx, cy = p3
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10: raise ValueError("Collinear.")
        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / D
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / D
        c = np.array([ux, uy]); return c, float(np.linalg.norm(p1 - c))

    @staticmethod
    def _lsq_c(pts):
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([x, y, np.ones(len(pts))]); b = x ** 2 + y ** 2
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = res[0] / 2, res[1] / 2
        return np.array([cx, cy]), float(np.sqrt(res[2] + cx ** 2 + cy ** 2))


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class SegmentationLayer:
    def segment_rings(
        self, context: PipelineContext, segment_width: float = 1.5
    ) -> List[np.ndarray]:
        pts = context.working_points
        if pts is None: raise RuntimeError("No working_points.")
        pts = validate_xyz(pts)
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        cl = context.centerline; c = cl.mean(axis=0)
        ev, vecs = np.linalg.eigh(np.cov((cl - c).T))
        ax = vecs[:, np.argmax(ev)]
        proj = (pts - c) @ ax
        pmin, pmax = float(proj.min()), float(proj.max())
        nr = max(1, int(np.ceil((pmax - pmin) / segment_width)))
        rings: List[np.ndarray] = []
        for i in range(nr):
            lo = pmin + i * segment_width; hi = pmin + (i + 1) * segment_width
            if i == nr - 1: hi = pmax + 1e-9
            mask = (proj >= lo) & (proj < hi)
            if mask.sum() >= 10: rings.append(pts[mask])
        return rings

    def detect_seam_boundaries(
        self, ring_pts: np.ndarray, center: np.ndarray,
        frenet_frame: Dict, k_clusters: int = 6
    ) -> List[np.ndarray]:
        N_vec = frenet_frame["N"]; B_vec = frenet_frame["B"]; C = np.asarray(center, dtype=np.float64)
        rp = np.asarray(ring_pts, dtype=np.float64)
        if len(rp) < max(10, k_clusters * 3): return []
        d = rp - C; pts2d = np.column_stack([d @ N_vec, d @ B_vec])
        ang = np.arctan2(pts2d[:, 1], pts2d[:, 0])
        # FIX-1: axis=1
        radii = np.linalg.norm(pts2d, axis=1); valid = radii > 1e-4
        ang_v = ang[valid]
        si = np.argsort(ang_v); ang_s = ang_v[si]
        # FIX-1: axis=1
        r_s = np.linalg.norm(pts2d[valid][si], axis=1)
        dr = np.abs(np.gradient(r_s)); thr = float(np.mean(dr) + 1.5 * np.std(dr))
        cand = ang_s[dr > thr]
        if len(cand) < k_clusters: cand = np.linspace(-np.pi, np.pi, k_clusters, endpoint=False)
        try:
            from scipy.cluster.vq import kmeans
            pkm = np.column_stack([np.cos(cand), np.sin(cand)])
            ck, _ = kmeans(pkm, min(k_clusters, len(pkm)), iter=50)
            sa = np.arctan2(ck[:, 1], ck[:, 0])
        except Exception: sa = np.linspace(-np.pi, np.pi, k_clusters, endpoint=False)
        rm = float(np.median(np.linalg.norm(pts2d[valid], axis=1)))
        bds: List[np.ndarray] = []
        for a in sa:
            d2 = np.array([math.cos(a), math.sin(a)]); tr = float(np.linalg.norm(d2 * rm * 1.05))
            if tr > rm * 0.95: bds.append(d2)
        return bds


# ------------------------------------------------------------------------------
# Layer 4 - ParameterExtractionLayer (2D flat cross-section processing)
# ------------------------------------------------------------------------------

class ParameterExtractionLayer:
    @staticmethod
    def _req(context: PipelineContext, step: str) -> np.ndarray:
        pts = context.working_points
        if pts is None: raise RuntimeError(f"{step}: no point cloud.")
        return validate_xyz(pts)

    def calc_arch_settlement(self, context: PipelineContext) -> Dict[str, float]:
        z = self._req(context, "5.1")[:, 2]
        cr = float(np.percentile(z, 99)); sp = float(np.percentile(z, 50)); inv = float(np.percentile(z, 1))
        return {"crown_settlement_mm": (cr - sp) * 1e3, "total_height_mm": (cr - inv) * 1e3,
                "crown_z_m": cr, "springline_z_m": sp, "invert_z_m": inv}

    def calc_horizontal_convergence(self, context: PipelineContext) -> Dict[str, float]:
        x = self._req(context, "5.2")[:, 0]
        lx = float(np.percentile(x, 1)); rx = float(np.percentile(x, 99)); mx = float(np.mean(x))
        return {"lateral_convergence_mm": (rx - lx) * 1e3, "lateral_centre_offset_mm": (mx - (rx + lx) / 2) * 1e3,
                "left_wall_x_m": lx, "right_wall_x_m": rx}

    def generate_heatmap(self, context: PipelineContext) -> Tuple[np.ndarray, np.ndarray]:
        pts = self._req(context, "5.3")
        return pts, (pts[:, 2] - float(np.median(pts[:, 2]))) * 1e3

    def generate_polar_deformation_map(
        self, context: PipelineContext, design_radius_m: float = 3.0, num_bins: int = 72
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = self._req(context, "5.4")
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        edges = np.linspace(-np.pi, np.pi, num_bins + 1); angles = 0.5 * (edges[:-1] + edges[1:])
        sc: List[np.ndarray] = []; dm: List[np.ndarray] = []
        for fr in context.frenet_frames:
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < 0.05; sl = pts[mask]
            if len(sl) < 10: continue
            d = sl - C; xf = d @ N; yf = d @ B
            r = np.hypot(xf, yf); theta = np.arctan2(yf, xf)
            bidx = np.clip(np.digitize(theta, edges) - 1, 0, num_bins - 1)
            dr = np.full(num_bins, np.nan, dtype=np.float64)
            for b in range(num_bins):
                bm = bidx == b
                if bm.any(): dr[b] = (float(np.nanmedian(r[bm])) - design_radius_m) * 1e3
            sc.append(C.copy()); dm.append(dr)
        if not sc: raise RuntimeError("No valid sections for polar map.")
        return np.asarray(sc, dtype=np.float64), angles, np.asarray(dm, dtype=np.float64)

    def calc_ovality(self, context: PipelineContext) -> Dict[str, float]:
        pts = self._req(context, "5.5")
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        ov: List[float] = []
        for fr in context.frenet_frames:
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < 0.05; sl = pts[mask]
            if len(sl) < 10: continue
            d = sl - C; xf = d @ N; yf = d @ B
            M = np.array([[float(np.mean(xf ** 2)), float(np.mean(xf * yf))],
                          [float(np.mean(xf * yf)), float(np.mean(yf ** 2))]])
            ev = np.linalg.eigvalsh(M)
            a = float(np.sqrt(max(ev.max(), 1e-9))); b = float(np.sqrt(max(ev.min(), 1e-9)))
            if a > 1e-6: ov.append((a - b) / a * 100.0)
        if not ov: return {"ovality_mean_pct": float("nan"), "ovality_max_pct": float("nan")}
        return {"ovality_mean_pct": float(np.mean(ov)), "ovality_max_pct": float(np.max(ov))}

    def calc_eccentricity(self, context: PipelineContext) -> Dict[str, float]:
        pts = self._req(context, "5.6")
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        ec: List[float] = []
        for fr in context.frenet_frames:
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < 0.05; sl = pts[mask]
            if len(sl) < 10: continue
            d = sl - C; xf = d @ N; yf = d @ B
            cm = C + float(np.mean(xf)) * N + float(np.mean(yf)) * B
            ec.append(float(np.linalg.norm(cm - C)) * 1e3)
        if not ec: return {"eccentricity_mean_mm": float("nan"), "eccentricity_max_mm": float("nan")}
        return {"eccentricity_mean_mm": float(np.mean(ec)), "eccentricity_max_mm": float(np.max(ec))}

    @staticmethod
    def _classify_pts_2d(pts2d: np.ndarray, profile: str = "Circle") -> np.ndarray:
        K = len(pts2d)
        labels = np.zeros(K, dtype=np.int32)
        z = pts2d[:, 1]
        z_lo = float(np.percentile(z, 12))
        z_hi = float(np.percentile(z, 88))
        z_range = z_hi - z_lo if z_hi > z_lo else 1.0
        for i in range(K):
            frac = (z[i] - z_lo) / z_range
            if profile == "U-type":
                if frac < 0.15: labels[i] = 2 
                else: labels[i] = 0           
            else:
                if frac > 0.72: labels[i] = 1   
                elif frac < 0.15: labels[i] = 2 
                else: labels[i] = 0             
        return labels

    @staticmethod
    def _wall_angle(pts2d: np.ndarray, side: str = "left") -> float:
        x = pts2d[:, 0]; z = pts2d[:, 1]
        if side == "left":
            mask = x < float(np.percentile(x, 25))
        else:
            mask = x > float(np.percentile(x, 75))
        wp = pts2d[mask]
        if len(wp) < 4: return float("nan")
        try:
            coeffs = np.polyfit(wp[:, 1], wp[:, 0], 1) 
            return math.degrees(math.atan(abs(float(coeffs[0]))))
        except Exception:
            return float("nan")

    def _extract_section_geometry(
        self, pts2d: np.ndarray, labels: np.ndarray, profile: str,
        vl_box_w: float, vl_box_h: float, vl_cir_r: float
    ) -> Dict[str, float]:
        x = pts2d[:, 0]; z = pts2d[:, 1]
        x_min = float(np.percentile(x, 1)); x_max = float(np.percentile(x, 99))
        z_min = float(np.percentile(z, 1)); z_max = float(np.percentile(z, 99))
        z_mid = float(np.percentile(z, 50))
        W1 = x_max - x_min
        H1 = z_max - z_min
        H2 = z_max - z_mid
        H3 = z_mid - z_min
        z_band = (z >= z_mid - H1 * 0.04) & (z <= z_mid + H1 * 0.04)
        W2 = (float(np.percentile(x[z_band], 99)) - float(np.percentile(x[z_band], 1))
              if z_band.sum() > 4 else W1)
        if profile == "Circle":
            radii_eval = np.hypot(x - (x_min + x_max)/2.0, z - z_mid)
            C1 = max(0.0, float(np.percentile(radii_eval, 5)) - vl_cir_r)
            C2 = C1
            C3 = max(0.0, z_max - (z_mid + vl_cir_r))
        else:
            C1 = max(0.0, abs(x_min) - vl_box_w)
            C2 = max(0.0, x_max - vl_box_w)
            C3 = max(0.0, z_max - (z_min + vl_box_h))

        if profile == "Circle":
            signed_clearance = np.hypot(x, z) - vl_cir_r
            min_clearance_dist = float(np.nanmin(signed_clearance)) if signed_clearance.size else float("nan")
            clearance_violation = bool(np.any(signed_clearance < 0.0))
        else:
            inside_x = (x >= -vl_box_w) & (x <= vl_box_w)
            inside_z = (z >= 0.0) & (z <= vl_box_h)
            inside = inside_x & inside_z
            dx_out = np.maximum(np.abs(x) - vl_box_w, 0.0)
            dz_out = np.maximum.reduce((np.zeros_like(z), -z, z - vl_box_h))
            signed_clearance = np.hypot(dx_out, dz_out)
            if inside.any():
                signed_clearance = signed_clearance.copy()
                inside_margin = np.minimum.reduce((vl_box_w - np.abs(x), z, vl_box_h - z))
                signed_clearance[inside] = -inside_margin[inside]
            min_clearance_dist = float(np.nanmin(signed_clearance)) if signed_clearance.size else float("nan")
            clearance_violation = bool(inside.any())
        # UI display disabled ? logic active for PDF 3.6 data pipeline
        wal = self._wall_angle(pts2d, "left")
        war = self._wall_angle(pts2d, "right")
        r_fit = float("nan")
        if profile == "Circle":
            try:
                from scipy.optimize import least_squares
                cx0 = float(np.clip(np.mean(x), -2.0, 2.0))
                cz0 = float(np.clip(np.mean(z), -2.0, 2.0))
                r0 = float(np.clip((W1 + H1) / 4.0, 2.0, 15.0))
                def res(p): return np.sqrt((x - p[0])**2 + (z - p[1])**2) - p[2]
                sol = least_squares(
                    res, [cx0, cz0, r0], loss="soft_l1", max_nfev=50,
                    bounds=([-2.0, -2.0, 2.0], [2.0, 2.0, 15.0])
                )
                if sol.success and np.isfinite(sol.x[2]):
                    r_fit = float(sol.x[2])
            except Exception:
                r_fit = float("nan")

        cx = float(np.mean(x)); cz = float(np.mean(z))
        M = np.array([[float(np.mean((x - cx)**2)), float(np.mean((x - cx) * (z - cz)))],
                      [float(np.mean((x - cx) * (z - cz))), float(np.mean((z - cz)**2))]])
        ev = np.linalg.eigvalsh(M)
        a = float(np.sqrt(max(ev.max(), 1e-9))); b = float(np.sqrt(max(ev.min(), 1e-9)))
        ovality = (a - b) / a * 100.0 if a > 1e-6 else float("nan")
        ecc = float(np.sqrt((cx - (x_min + x_max)/2.0)**2 + (cz - (z_min + z_max)/2.0)**2)) * 1e3
        return dict(H1=H1, H2=H2, H3=H3, W1=W1, W2=W2, C1=C1, C2=C2, C3=C3,
                    wall_angle_L=wal, wall_angle_R=war, radius_fit=r_fit, ovality=ovality,
                    eccentricity=ecc, clearance_violation=clearance_violation,
                    min_clearance_dist=min_clearance_dist)

    def compute_all_sections(
        self, context: PipelineContext, vl_box_w: float, vl_box_h: float, vl_cir_r: float, epsilon: float = 0.05
    ) -> List[SectionGeometry]:
        pts = self._req(context, "5.7")
        if not context.frenet_frames: raise RuntimeError("Centerline frames missing.")
        profile = context.tunnel_profile
        sections: List[SectionGeometry] = []
        cl = context.centerline
        if cl is not None and len(cl) == len(context.frenet_frames):
            # FIX-1: explicit axis=
            chain_diffs = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
            chainages = chain_diffs.tolist()
        else:
            chainages = list(range(len(context.frenet_frames)))

        for idx, fr in enumerate(context.frenet_frames):
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < epsilon
            sl = pts[mask]
            if len(sl) < 8:
                sections.append(SectionGeometry(chainage=chainages[idx], center_3d=C))
                continue
            d = sl - C
            xf = (d @ N).reshape(-1, 1) 
            zf = (d @ B).reshape(-1, 1) 
            pts2d = np.hstack([xf, zf])
            dist_2d = np.hypot(pts2d[:, 0], pts2d[:, 1])
            valid_mask = dist_2d < 15.0
            pts2d = pts2d[valid_mask]
            if len(pts2d) < 8:
                sections.append(SectionGeometry(chainage=chainages[idx], center_3d=C, pts_2d=pts2d))
                continue
            labels = self._classify_pts_2d(pts2d, profile)
            geom = self._extract_section_geometry(pts2d, labels, profile, vl_box_w, vl_box_h, vl_cir_r)
            sg = SectionGeometry(
                chainage=chainages[idx], center_3d=C, pts_2d=pts2d, labels=labels,
                H1=geom["H1"], H2=geom["H2"], H3=geom["H3"], W1=geom["W1"], W2=geom["W2"],
                C1=geom["C1"], C2=geom["C2"], C3=geom["C3"],
                wall_angle_L=geom["wall_angle_L"], wall_angle_R=geom["wall_angle_R"],
                radius_fit=geom["radius_fit"], ovality=geom["ovality"], eccentricity=geom["eccentricity"],
                clearance_violation=geom["clearance_violation"], min_clearance_dist=geom["min_clearance_dist"]
            )
            sections.append(sg)
        return self._smooth_series(sections)

    @staticmethod
    def _smooth_series(sections: List[SectionGeometry]) -> List[SectionGeometry]:
        if len(sections) < 6: return sections
        fields = ["H1", "H2", "H3", "W1", "W2", "radius_fit", "ovality", "eccentricity"]
        chain = np.array([s.chainage for s in sections], dtype=np.float64)
        for fld in fields:
            vals = np.array([getattr(s, fld) for s in sections], dtype=np.float64)
            finite = np.isfinite(vals)
            if finite.sum() < 4: continue
            try:
                coeffs = np.polyfit(chain[finite], vals[finite], 2)
                smoothed = np.polyval(coeffs, chain)
                for i, s in enumerate(sections):
                    if not np.isnan(getattr(s, fld)):
                        val = float(smoothed[i])
                        if fld == "radius_fit": val = float(np.clip(val, 2.0, 15.0))
                        setattr(s, fld, val)
            except Exception: pass
        return sections


# ------------------------------------------------------------------------------
# TimeSeriesLayer
# ------------------------------------------------------------------------------

class TimeSeriesLayer:
    def load_epochs(self, p0: str, pn: str) -> Tuple[PointCloudBundle, PointCloudBundle]:
        bl = BaseLayer(); return bl.load_scan(p0), bl.load_scan(pn)

    def plot_deformation(self, context: PipelineContext) -> np.ndarray:
        pts = context.working_points
        if pts is None: raise RuntimeError("Load epochs first.")
        pts = validate_xyz(pts)
        sc  = (pts[:,2] - np.median(pts[:,2]))*1e3
        ord_= np.argsort(pts[:,0])
        return np.array([float(np.nanmean(c)) for c in np.array_split(sc[ord_],120) if len(c)>0])


# ------------------------------------------------------------------------------
# DigitalTwinAILayer
# ------------------------------------------------------------------------------

class DigitalTwinAILayer:
    OLLAMA_URL   = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3"
    _TIMEOUT     = (5.0, 120.0)

    def export_ifc(self, context: PipelineContext) -> Dict[str, object]:
        scan = context.active_scan
        return {"ifc_schema":"IFC4","status":"ready_for_ifcopenshell_hook",
                "scan_path": scan.path if scan else None,
                "point_count": int(len(scan.points)) if scan else 0,
                "centerline_points": 0 if context.centerline is None else int(len(context.centerline)),
                "frenet_frames": int(len(context.frenet_frames)),
                "parameters": context.parameters, "local_llm_status":"ready_for_ollama_hook"}

    def query_local_ai(self, prompt: str, context: PipelineContext) -> str:
        try: import requests
        except ImportError: return "[ERROR] pip install requests"
        sys_p = self._sys(context)
        payload = {"model":self.OLLAMA_MODEL,
                   "prompt":f"{sys_p}\n\nEngineer query: {prompt}",
                   "stream":False,"options":{"temperature":0.2,"num_predict":1024}}
        try:
            r = requests.post(self.OLLAMA_URL, json=payload, timeout=self._TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            return (f"[CONNECTION ERROR] {e}\n\n"
                    f"Start Ollama: ollama serve\nPull: ollama pull {self.OLLAMA_MODEL}")
        try: data = r.json()
        except ValueError: return "[PARSE ERROR]\n" + r.text[:600]
        text = data.get("response","").strip()
        if not text: return "[EMPTY]\n" + json.dumps(data,indent=2)[:400]
        m=data.get("model", "unknown"); n=data.get("eval_count", "unknown")
        es=data.get("eval_duration",0)/1e9; ls=data.get("load_duration",0)/1e9
        return f"{text}\n\n{'-'*52}\nModel:{m}|Tokens:{n}|Eval:{es:.1f}s|Load:{ls:.1f}s"

    def _sys(self, context: PipelineContext) -> str:
        p = context.parameters
        lines = ("\n".join(f"  - {k}: {v:.3f}" for k,v in p.items()
                           if isinstance(v,(int,float))) if p else "  (none)")
        scan = context.active_scan
        info = (f"Scan: {scan.path}\nPoints: {len(scan.points):,}" if scan else "not loaded")
        return (
            "You are a licensed structural engineer specialising in tunnel SHM.\n"
            "Thresholds: Crown settlement >10mm=caution|>25mm=critical; "
            "Lateral convergence >15mm=caution|>30mm=critical; "
            "Ovality >0.5%=caution|>1.0%=critical.\n"
            f"--- Scan ---\n{info}\n--- Parameters ---\n{lines}\n--------------------------------"
        )


# ------------------------------------------------------------------------------
# Qt worker
# ------------------------------------------------------------------------------

class PipelineWorker(QtCore.QObject):
    finished = QtCore.Signal(str, object)
    failed   = QtCore.Signal(str, str)

    def __init__(self, key: str, cb: Callable[[], object]) -> None:
        super().__init__(); self.task_key=key; self.callback=cb

    @QtCore.Slot()
    def run(self) -> None:
        try: self.finished.emit(self.task_key, self.callback())
        except Exception as e: self.failed.emit(self.task_key, str(e))


# ------------------------------------------------------------------------------
# Collapsible sidebar section
# ------------------------------------------------------------------------------

class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title: str, step: int, tag: str, parent=None):
        super().__init__(parent)
        self._btn = QtWidgets.QToolButton()
        self._btn.setCheckable(True); self._btn.setChecked(False)
        self._btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(QtCore.Qt.RightArrow)
        self._btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._btn.setMinimumHeight(44); self._btn.setObjectName("SectionToggle")
        self._btn.setText(f"  Step {step}: {title}  [{tag}]")
        self._btn.toggled.connect(self._toggle)
        self._body = QtWidgets.QWidget(); self._body.setObjectName("SectionContent")
        self._blay = QtWidgets.QVBoxLayout(self._body)
        self._blay.setContentsMargins(12,4,4,8); self._blay.setSpacing(4)
        self._body.setVisible(False)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._btn); root.addWidget(self._body)

    def _toggle(self, checked: bool) -> None:
        self._btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._body.setVisible(checked)

    def add_sub_button(self, label: str, slot: Callable) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(f"  - {label}")
        b.setObjectName("SubButton"); b.setMinimumHeight(32)
        b.setCursor(QtCore.Qt.PointingHandCursor); b.clicked.connect(slot)
        self._blay.addWidget(b); return b

    def all_sub_buttons(self) -> List[QtWidgets.QPushButton]:
        return self._body.findChildren(QtWidgets.QPushButton)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class MatplotlibSectionWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sections: List[SectionGeometry] = []
        self._idx: int = 0
        self._profile: str = "Circle"
        self._vl_box_w  = VL_BOX_W
        self._vl_box_h  = VL_BOX_H
        self._vl_cir_r  = VL_CIR_R

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)

        nav = QtWidgets.QHBoxLayout()
        self._btn_prev = QtWidgets.QPushButton("\u25C0 Prev")
        self._btn_next = QtWidgets.QPushButton("Next \u25B6")
        self._lbl_ch   = QtWidgets.QLabel("Current chainage: --")
        self._lbl_ch.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_ch.setStyleSheet("color:#111827; font-weight:bold; font-size:10.5pt;")
        self._btn_prev.clicked.connect(self._prev); self._btn_next.clicked.connect(self._next)
        nav.addWidget(self._btn_prev); nav.addWidget(self._lbl_ch, 1); nav.addWidget(self._btn_next)
        lay.addLayout(nav)

        if _MPL_OK:
            self._fig = Figure(figsize=(7.5, 6.5), facecolor=_BG)
            self._ax  = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
            lay.addWidget(self._canvas, 1)
        else:
            lay.addWidget(QtWidgets.QLabel("Matplotlib is required for 2D cross-section plotting."))
        self._draw_empty()

    def _prev(self) -> None:
        if not self._sections: return
        self._idx = (self._idx - 1) % len(self._sections); self._refresh()

    def _next(self) -> None:
        if not self._sections: return
        self._idx = (self._idx + 1) % len(self._sections); self._refresh()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Left: self._prev()
        elif event.key() == QtCore.Qt.Key_Right: self._next()
        super().keyPressEvent(event)

    def set_sections(self, sections: List[SectionGeometry], profile: str, vl_box_w: float, vl_box_h: float, vl_cir_r: float) -> None:
        self._sections = sections; self._idx = 0; self._profile = profile
        self._vl_box_w = vl_box_w; self._vl_box_h = vl_box_h; self._vl_cir_r = vl_cir_r
        self._refresh()

    def _draw_empty(self) -> None:
        if not _MPL_OK: return
        ax = self._ax; ax.clear(); ax.set_facecolor(_BG)
        ax.text(0.5, 0.5, "Run Step 5.7: Plot 2D Technical Section\nto display tunnel cross-sections and engineering dimensions.",
                ha="center", va="center", color=_FG, fontsize=11, transform=ax.transAxes)
        for s in ax.spines.values(): s.set_color(_GRID)
        ax.tick_params(colors=_FG); self._canvas.draw_idle()

    def _refresh(self) -> None:
        if not _MPL_OK or not self._sections: self._draw_empty(); return
        sg = self._sections[self._idx]
        self._lbl_ch.setText(f"Chainage: {sg.chainage:.2f} m  ({self._idx + 1}/{len(self._sections)})")
        if sg.pts_2d is None or len(sg.pts_2d) < 4: self._draw_empty(); return
        self._draw_section(sg)

    def _draw_section(self, sg: SectionGeometry) -> None:
        ax = self._ax; ax.clear(); ax.set_facecolor(_BG)
        pts2d = sg.pts_2d
        if pts2d is None or len(pts2d) < 4:
            self._draw_empty(); return
        labels = sg.labels if sg.labels is not None and len(sg.labels) == len(pts2d) else np.zeros(len(pts2d), dtype=np.int32)
        finite = np.isfinite(pts2d[:, 0]) & np.isfinite(pts2d[:, 1])
        pts2d = pts2d[finite]; labels = labels[finite]
        if len(pts2d) < 4:
            self._draw_empty(); return
        x = pts2d[:, 0]; z = pts2d[:, 1]

        colors_map = np.where(labels == 1, _ACC3, np.where(labels == 2, _ACC2, _ACC1))
        ax.scatter(x, z, c=colors_map, s=2.0, alpha=0.62, linewidths=0, rasterized=True)

        xmn, xmx = float(np.percentile(x, 1)), float(np.percentile(x, 99))
        zmn, zmx = float(np.percentile(z, 1)), float(np.percentile(z, 99))
        zmid = (zmn + zmx) / 2.0
        x_span = max(xmx - xmn, 1.0)
        z_span = max(zmx - zmn, 1.0)
        dim_gap = max(0.35, 0.08 * max(x_span, z_span))
        label_box = dict(facecolor="#FFFFFFEE", edgecolor=_GRID, boxstyle="round,pad=0.22")

        def _hdim(x0, x1, y, text):
            ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="<->", color=_DIM, lw=1.2))
            ax.text((x0 + x1)/2.0, y + dim_gap*0.18, text, color=_DIM, fontsize=8,
                    ha="center", va="bottom", bbox=label_box)

        def _vdim(y0, y1, x_pos, text):
            ax.annotate("", xy=(x_pos, y1), xytext=(x_pos, y0), arrowprops=dict(arrowstyle="<->", color=_DIM, lw=1.2))
            ax.text(x_pos + dim_gap*0.18, (y0 + y1)/2.0, text, color=_DIM, fontsize=8,
                    ha="left", va="center", bbox=label_box)

        dim_y_top = zmx + dim_gap
        dim_y_bottom = zmn - dim_gap
        dim_x_right = xmx + dim_gap
        dim_x_left = xmn - dim_gap
        if np.isfinite(sg.W1): _hdim(xmn, xmx, dim_y_top, f"W1={sg.W1:.3f}m")
        if np.isfinite(sg.W2): _hdim(xmn, xmx, dim_y_bottom, f"W2={sg.W2:.3f}m")
        if np.isfinite(sg.H1): _vdim(zmn, zmx, dim_x_right, f"H1={sg.H1:.3f}m")
        if np.isfinite(sg.H2): _vdim(zmid, zmx, dim_x_left, f"H2={sg.H2:.3f}m")
        if np.isfinite(sg.H3): _vdim(zmn, zmid, dim_x_left - dim_gap*0.55, f"H3={sg.H3:.3f}m")

        def _draw_angle_arc(angle: float, cx: float, cz: float, side: str):
            if not np.isfinite(angle): return
            r_arc = min(x_span, z_span) * 0.14
            start_ang = 90.0
            ext_ang = angle if side == "left" else -angle
            arc_patch = mpatches.Arc((cx, cz), 2*r_arc, 2*r_arc, angle=0,
                                     theta1=min(start_ang, start_ang+ext_ang),
                                     theta2=max(start_ang, start_ang+ext_ang), color=_YEL, lw=1.5)
            ax.add_patch(arc_patch)
            m_rad = math.radians(start_ang + ext_ang/2.0)
            ax.text(cx + r_arc*1.4*math.cos(m_rad), cz + r_arc*1.4*math.sin(m_rad),
                    f"{angle:.1f} deg", color=_YEL, fontsize=7.5, fontweight="bold",
                    ha="center", va="center", bbox=label_box)

        _draw_angle_arc(sg.wall_angle_L, xmn + x_span*0.10, zmid, "left")
        _draw_angle_arc(sg.wall_angle_R, xmx - x_span*0.10, zmid, "right")

        # DISABLED: vehicle clearance envelope drawing temporarily disabled
        # vl_color = _DIM
        # vl_lw = 1.6
        # vl_ls = "-."
        # if self._profile == "Circle":
        #     vl_circ = plt.Circle((0.0, 0.0), self._vl_cir_r, fill=False, edgecolor=vl_color, lw=vl_lw, ls=vl_ls, alpha=0.95)
        #     ax.add_patch(vl_circ)
        #     ax.text(0.0, self._vl_cir_r + dim_gap*0.35, f"VL R={self._vl_cir_r:.1f}m", color=vl_color,
        #             fontsize=7.5, ha="center", bbox=label_box)
        #     if np.isfinite(sg.radius_fit):
        #         fit_c = plt.Circle((0.0, 0.0), sg.radius_fit, fill=False, edgecolor=_GRN, lw=1.2, ls="--", alpha=0.85)
        #         ax.add_patch(fit_c)
        # else:
        #     vl_box = mpatches.Rectangle((-self._vl_box_w, 0.0), 2*self._vl_box_w, self._vl_box_h,
        #                                 fill=False, edgecolor=vl_color, lw=vl_lw, ls=vl_ls, alpha=0.95)
        #     ax.add_patch(vl_box)
        #     ax.text(0.0, self._vl_box_h + dim_gap*0.35, f"VL {2*self._vl_box_w:.1f}x{self._vl_box_h:.1f}m", color=vl_color,
        #             fontsize=7.5, ha="center", bbox=label_box)
        #     if self._profile == "Box 2-cell":
        #         ax.plot([0.0, 0.0], [0.0, self._vl_box_h], color=vl_color, lw=1.0, ls=":")
        # DISABLED: collision violation banner temporarily disabled
        # if sg.clearance_violation:
        #     ax.text(0.5, 0.94, "\u26A0\uFE0F CLEARANCE VIOLATION DETECTED!", transform=ax.transAxes,
        #             ha="center", va="top", color=_RED, fontsize=10.5, fontweight="bold",
        #             bbox=dict(facecolor="#FFFFFFEE", edgecolor=_RED, boxstyle="round,pad=0.35"))
        inf = [f"Chainage: {sg.chainage:.2f} m", f"Clear width W1: {sg.W1:.3f} m", f"Clear height H1: {sg.H1:.3f} m",
               f"Ovality epsilon: {sg.ovality:.2f} %", f"Eccentricity e: {sg.eccentricity:.1f} mm"]
        if np.isfinite(sg.min_clearance_dist): inf.append(f"Min clearance: {sg.min_clearance_dist:.3f} m")
        # DISABLED: if sg.clearance_violation: inf.append("Status: CLEARANCE VIOLATION")
        if self._profile == "Circle" and np.isfinite(sg.radius_fit): inf.append(f"R_Fit: {sg.radius_fit:.3f} m")
        ax.text(0.02, 0.98, "\n".join(inf), transform=ax.transAxes, fontsize=7.5, color=_FG,
                va="top", ha="left", family="monospace", bbox=dict(facecolor="#FFFFFFEE", edgecolor=_GRID, pad=4))

        if self._profile == "Circle":
            vl_x0, vl_x1 = -self._vl_cir_r, self._vl_cir_r
            vl_z0, vl_z1 = -self._vl_cir_r, self._vl_cir_r
        else:
            vl_x0, vl_x1 = -self._vl_box_w, self._vl_box_w
            vl_z0, vl_z1 = 0.0, self._vl_box_h
        plot_x0 = min(float(np.min(x)), vl_x0, dim_x_left - dim_gap)
        plot_x1 = max(float(np.max(x)), vl_x1, dim_x_right + dim_gap)
        plot_z0 = min(float(np.min(z)), vl_z0, dim_y_bottom - dim_gap)
        plot_z1 = max(float(np.max(z)), vl_z1, dim_y_top + dim_gap)
        pad = max(0.4, 0.04 * max(plot_x1 - plot_x0, plot_z1 - plot_z0))
        view_cap = 16.0
        plot_x0 = max(plot_x0 - pad, -view_cap); plot_x1 = min(plot_x1 + pad, view_cap)
        plot_z0 = max(plot_z0 - pad, -view_cap); plot_z1 = min(plot_z1 + pad, view_cap)
        if plot_x1 - plot_x0 < 1.0:
            mid = (plot_x0 + plot_x1) / 2.0; plot_x0, plot_x1 = mid - 0.5, mid + 0.5
        if plot_z1 - plot_z0 < 1.0:
            mid = (plot_z0 + plot_z1) / 2.0; plot_z0, plot_z1 = mid - 0.5, mid + 0.5

        ax.set_xlabel("Flat 2D horizontal axis X_2D (normal vector N, m)", color=_FG, fontsize=8.5)
        ax.set_ylabel("Flat 2D vertical axis Z_2D (binormal vector B, m)", color=_FG, fontsize=8.5)
        ax.set_title(f"2D technical tunnel cross-section analysis  |  Profile: {self._profile}", color=_FG, fontsize=10, fontweight="bold")
        ax.set_xlim(plot_x0, plot_x1); ax.set_ylim(plot_z0, plot_z1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color=_GRID, lw=0.5, alpha=0.4)
        for spine in ax.spines.values(): spine.set_color(_GRID)
        ax.tick_params(colors=_FG, labelsize=7.5)

        lbl_p = [mpatches.Patch(color=_ACC1, label="Wall"), mpatches.Patch(color=_ACC3, label="Crown"), mpatches.Patch(color=_ACC2, label="Floor")]
        ax.legend(handles=lbl_p, fontsize=7.5, facecolor=_BG, edgecolor=_GRID, labelcolor=_FG, loc="lower right")

        self._fig.tight_layout(); self._canvas.draw_idle()


# ------------------------------------------------------------------------------
# PolarDeformationPlotWidget
# ------------------------------------------------------------------------------

class PolarDeformationPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._angles: Optional[np.ndarray] = None; self._dmap: Optional[np.ndarray] = None
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        if _MPL_OK:
            self._fig, self._ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(4, 4))
            self._fig.patch.set_facecolor(_BG); self._canvas = FigureCanvas(self._fig); lay.addWidget(self._canvas)
        else: lay.addWidget(QtWidgets.QLabel("Matplotlib missing."))

    def update_data(self, angles: np.ndarray, dmap: np.ndarray) -> None:
        if not _MPL_OK: return
        self._angles = angles; self._dmap = dmap; self._redraw()

    def _redraw(self) -> None:
        if not _MPL_OK or self._angles is None: return
        ax = self._ax; ax.clear()
        mean_dr = np.nanmean(self._dmap, axis=0); ang = self._angles
        for i in range(len(ang)-1):
            if np.isnan(mean_dr[i]): continue
            av = abs(float(mean_dr[i]))
            col = _GRN if av < 1.0 else (_YEL if av < 3.0 else _RED)
            ax.bar(ang[i], av, width=(ang[1] - ang[0]), color=col, alpha=0.85, edgecolor="none")
        ax.set_title("Polar radial deformation dr [mm]", color=_FG, fontsize=9, pad=8)
        ax.set_facecolor(_BG); ax.tick_params(colors=_FG, labelsize=7); ax.grid(True, color=_GRID, lw=0.6, alpha=0.75)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        self._fig.tight_layout(); self._canvas.draw_idle()


# ------------------------------------------------------------------------------
# LinePlotWidget
# ------------------------------------------------------------------------------

class LinePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: Optional[np.ndarray] = None; self.title = "Time-series"; self.setMinimumHeight(220)

    def set_values(self, values: Optional[np.ndarray], title: str = "") -> None:
        self.values = None if values is None else np.asarray(values, dtype=np.float64).ravel()
        self.title = title or "Time-series"; self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        rc = self.rect().adjusted(14, 14, -14, -14)
        p.fillRect(self.rect(), QtGui.QColor("#FFFFFF"))
        p.setPen(QtGui.QPen(QtGui.QColor("#CBD5E1"), 1)); p.drawRoundedRect(rc, 6, 6)
        p.setPen(QtGui.QColor("#111827")); p.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        p.drawText(rc.adjusted(10, 6, -10, -6), QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft, self.title)
        if self.values is None or len(self.values) < 2:
            p.setFont(QtGui.QFont("Segoe UI", 9)); p.setPen(QtGui.QColor("#64748B"))
            p.drawText(rc, QtCore.Qt.AlignCenter, "Run Step 6.2 to generate chart."); return
        pr = rc.adjusted(42, 42, -18, -34)
        p.setPen(QtGui.QPen(QtGui.QColor("#E2E8F0"), 1))
        for i in range(5):
            y = pr.top() + i * pr.height() / 4.0; p.drawLine(pr.left(), int(y), pr.right(), int(y))
        vals = self.values[np.isfinite(self.values)]
        if len(vals) < 2: return
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if math.isclose(vmin, vmax): vmax = vmin + 1.0
        pts = []
        for i, v in enumerate(self.values):
            x = pr.left() + i / max(1, len(self.values) - 1) * pr.width()
            y = pr.bottom() - (float(v) - vmin) / (vmax - vmin) * pr.height()
            pts.append(QtCore.QPointF(x, y))
        p.setPen(QtGui.QPen(QtGui.QColor("#2563EB"), 2))
        for a, b in zip(pts[:-1], pts[1:]): p.drawLine(a, b)
        p.setPen(QtGui.QColor("#475569")); p.setFont(QtGui.QFont("Segoe UI", 8))
        p.drawText(pr.left(), rc.bottom() - 8, f"min {vmin:.2f}mm")
        p.drawText(pr.right() - 110, rc.bottom() - 8, f"max {vmax:.2f}mm")


# ------------------------------------------------------------------------------
# Main Window & PySide6 UI
# ------------------------------------------------------------------------------

class TunnelAnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tunnel Analysis v4.0 (r1) - CBNU Smart Structure Lab")
        self.resize(1720, 1000)

        self.context   = PipelineContext()
        self.base_mod  = BaseLayer()
        self.pre_mod   = PreprocessingLayer()
        self.reg_mod   = RegistrationLayer()
        self.geo_mod   = GeometricLayer()
        self.seg_mod   = SegmentationLayer()
        self.par_mod   = ParameterExtractionLayer()
        self.ts_mod    = TimeSeriesLayer()
        self.dt_mod    = DigitalTwinAILayer()

        self.plotter:        Optional[QtInteractor]   = None
        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker:        Optional[PipelineWorker] = None
        self._all_sub_btns: List[QtWidgets.QPushButton] = []
        self._ai_tab_idx:   int = 5
        self._section_tab_idx: int = 3

        self._build_ui()
        self._apply_theme()
        self._init_pyvista()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        self.setCentralWidget(central)
        root.addWidget(self._build_sidebar())

        right = QtWidgets.QWidget(); rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(14, 12, 14, 12); rlay.setSpacing(10)
        root.addWidget(right, 1)

        self.header = QtWidgets.QFrame(); self.header.setObjectName("Header")
        hlay = QtWidgets.QVBoxLayout(self.header); hlay.setContentsMargins(14, 10, 14, 10)
        self.task_title = QtWidgets.QLabel("Tunnel Analysis v4.0")
        self.task_title.setObjectName("TaskTitle")
        self.task_desc  = QtWidgets.QLabel("Select a structural analysis workflow from the sidebar.")
        self.task_desc.setWordWrap(True); self.task_desc.setObjectName("TaskDescription")
        hlay.addWidget(self.task_title); hlay.addWidget(self.task_desc)
        rlay.addWidget(self.header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal); rlay.addWidget(splitter, 1)

        self.vp_frame  = QtWidgets.QFrame(); self.vp_frame.setObjectName("ViewportFrame")
        self.vp_layout = QtWidgets.QVBoxLayout(self.vp_frame)
        self.vp_layout.setContentsMargins(0, 0, 0, 0); self.vp_layout.setSpacing(0)
        splitter.addWidget(self.vp_frame)

        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.setMinimumWidth(460); splitter.addWidget(self.right_tabs)
        splitter.setSizes([1100, 620])

        self.results_text = QtWidgets.QPlainTextEdit(); self.results_text.setReadOnly(True)
        self.right_tabs.addTab(self.results_text, "Results Log")

        self.meta_table = QtWidgets.QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.meta_table.horizontalHeader().setStretchLastSection(True)
        self.right_tabs.addTab(self.meta_table, "Scan Database")

        self.ts_plot = LinePlotWidget()
        self.right_tabs.addTab(self.ts_plot, "Time-Series Plot")

        self.section_widget = MatplotlibSectionWidget()
        self.right_tabs.addTab(self.section_widget, "2D Cross-Section")

        self.polar_plot = PolarDeformationPlotWidget()
        self.right_tabs.addTab(self.polar_plot, "Polar Deformation")

        ai_panel = QtWidgets.QWidget(); ai_lay = QtWidgets.QVBoxLayout(ai_panel)
        ai_lay.setContentsMargins(8, 8, 8, 8); ai_lay.setSpacing(6)
        self.ai_prompt = QtWidgets.QPlainTextEdit()
        self.ai_prompt.setPlaceholderText("Enter a structural engineering question for the local AI assistant (Llama 3)...")
        self.ai_prompt.setMaximumHeight(90)
        self.ai_send = QtWidgets.QPushButton("Query AI Assistant")
        self.ai_send.clicked.connect(self._slot_7_2_query_ai)
        self.ai_resp = QtWidgets.QPlainTextEdit(); self.ai_resp.setReadOnly(True)
        ai_lay.addWidget(QtWidgets.QLabel("Engineering query:")); ai_lay.addWidget(self.ai_prompt)
        ai_lay.addWidget(self.ai_send)
        ai_lay.addWidget(QtWidgets.QLabel("AI analysis report:")); ai_lay.addWidget(self.ai_resp, 1)
        self.right_tabs.addTab(ai_panel, "AI Engineering Assistant")

        self.sb_pts  = QtWidgets.QLabel("Points: --")
        self.sb_rmse = QtWidgets.QLabel("RMSE: --")
        self.sb_msg  = QtWidgets.QLabel("Ready")
        self.sb_prog = QtWidgets.QProgressBar(); self.sb_prog.setRange(0, 100)
        self.statusBar().addWidget(self.sb_pts)
        self.statusBar().addWidget(self.sb_rmse)
        self.statusBar().addWidget(self.sb_msg, 1)
        self.statusBar().addPermanentWidget(self.sb_prog)

    def _build_sidebar(self) -> QtWidgets.QFrame:
        sb = QtWidgets.QFrame(); sb.setObjectName("Sidebar"); sb.setFixedWidth(375)
        out = QtWidgets.QVBoxLayout(sb); out.setContentsMargins(10, 14, 10, 14); out.setSpacing(6)

        t1 = QtWidgets.QLabel("TUNNEL ANALYSIS"); t1.setObjectName("ProductTitle")
        t2 = QtWidgets.QLabel("v4.0 r1 - CBNU Smart Structure Lab"); t2.setObjectName("LabSubtitle")
        out.addWidget(t1); out.addWidget(t2)
        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setObjectName("Separator"); out.addWidget(sep)

        pf_frame = QtWidgets.QGroupBox("Tunnel Profile Type")
        pf_frame.setStyleSheet("QGroupBox{color:#334155;border:1px solid #CBD5E1;border-radius:5px;margin-top:6px;padding:4px;}")
        pf_lay = QtWidgets.QHBoxLayout(pf_frame)
        self._profile_combo = QtWidgets.QComboBox()
        self._profile_combo.addItems(TUNNEL_PROFILES)
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)
        pf_lay.addWidget(self._profile_combo); out.addWidget(pf_frame)

        vl_frame = QtWidgets.QGroupBox("Vehicle Clearance Limit (m)")
        vl_frame.setStyleSheet("QGroupBox{color:#334155;border:1px solid #CBD5E1;border-radius:5px;margin-top:6px;padding:4px;}")
        vl_lay = QtWidgets.QFormLayout(vl_frame)
        self._sp_vl_w = QtWidgets.QDoubleSpinBox(); self._sp_vl_w.setValue(VL_BOX_W)
        self._sp_vl_h = QtWidgets.QDoubleSpinBox(); self._sp_vl_h.setValue(VL_BOX_H)
        self._sp_vl_r = QtWidgets.QDoubleSpinBox(); self._sp_vl_r.setValue(VL_CIR_R)
        vl_lay.addRow("Half clear width W:", self._sp_vl_w)
        vl_lay.addRow("Clear height H:", self._sp_vl_h)
        vl_lay.addRow("Circular clearance radius R:", self._sp_vl_r)
        out.addWidget(vl_frame)

        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        out.addWidget(scroll, 1)
        sc = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(sc)
        sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(4); scroll.setWidget(sc)

        SECTIONS = [
            (1, "LiDAR data acquisition", "Base", [
                ("1.1  Import LAS / PLY data", self._slot_1_1_import),
                ("1.2  Initialize 3D viewport", self._slot_1_2_viewport),
            ]),
            (2, "Preprocessing and noise filtering", "Pre.", [
                ("2.1  Voxel downsampling", self._slot_2_1_voxel),
                ("2.2  Statistical outlier removal", self._slot_2_2_sor),
                ("2.3  Extract tunnel lining shell", self._slot_2_3_lining),
            ]),
            (3, "Registration and synchronization", "Reg.", [
                ("3.1  Anchor translation", self._slot_3_1_anchor),
                ("3.2  Fine surface ICP", self._slot_3_2_icp),
                ("3.3  Calculate RMSE", self._slot_3_3_rmse),
            ]),
            (4, "Geometric coordinate system", "Geo.", [
                ("4.1  Extract PCA centerline", self._slot_4_1_centerline),
                ("4.2  Iterative centerline refinement", self._slot_4_2_iterative),
                ("4.3  Smooth B-Spline centerline", self._slot_4_3_bspline),
                ("4.4  Generate gravity-aligned N-B sections", self._slot_4_4_frenet),
                ("4.5  Detect ring seams", self._slot_4_5_seams),
            ]),
            (5, "Parameter extraction", "Param.", [
                ("5.1  Crown settlement dv", self._slot_5_1_settlement),
                ("5.2  Horizontal convergence dh", self._slot_5_2_convergence),
                ("5.3  3D deformation heatmap", self._slot_5_3_heatmap),
                ("5.4  Polar radial deformation dr", self._slot_5_4_polar),
                ("5.5  Ovality epsilon", self._slot_5_5_ovality),
                ("5.6  Section eccentricity e", self._slot_5_6_eccentricity),
                ("5.7  Plot 2D Technical Section", self._slot_5_7_sections),
            ]),
            (6, "Time-series analysis", "T-S", [
                ("6.1  Load T0 and Tn epochs", self._slot_6_1_epochs),
                ("6.2  Plot deformation trend", self._slot_6_2_plot),
            ]),
            (7, "BIM and AI", "BIM/AI", [
                ("7.1  Export IFC package", self._slot_7_1_ifc),
                ("7.2  Query structural AI assistant", self._slot_7_2_query_ai),
            ]),
        ]
        for step, title_s, tag, buttons in SECTIONS:
            sec = CollapsibleSection(title_s, step, tag)
            for label, slot in buttons:
                btn = sec.add_sub_button(label, slot); self._all_sub_btns.append(btn)
            sl.addWidget(sec)

        sl.addStretch()
        self.pt_label   = QtWidgets.QLabel("Points: --")
        self.rmse_label = QtWidgets.QLabel("RMSE: --")
        out.addWidget(self.pt_label); out.addWidget(self.rmse_label)
        return sb

    def _on_profile_changed(self, text: str) -> None:
        self.context.tunnel_profile = text

    def _init_pyvista(self) -> None:
        while self.vp_layout.count():
            item = self.vp_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if pv is None: self._vp_msg("PyVista is not installed."); return
        try:
            self.plotter = QtInteractor(self.vp_frame); self.plotter.set_background("#F8FAFC")
            self.vp_layout.addWidget(self.plotter, 1); self.plotter.add_axes(color="#111827")
            self.plotter.show_bounds(color="#94A3B8", grid="front", location="outer", font_size=8)
            self.plotter.render()
        except Exception as exc: self.plotter = None; self._vp_msg(f"Failed to initialize PyVista: {exc}")

    def _vp_msg(self, msg: str) -> None:
        lbl = QtWidgets.QLabel(msg); lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setWordWrap(True); lbl.setObjectName("ViewportMessage")
        self.vp_layout.addWidget(lbl, 1)

    def _start_worker(self, key: str, cb: Callable[[], object]) -> None:
        if self.worker_thread is not None: self._log("A workflow task is already running."); return
        self._btns_enabled(False); self.sb_prog.setValue(10); self.sb_msg.setText(f"Running task: {key} ...")
        self.worker_thread = QtCore.QThread(self)
        self.worker = PipelineWorker(key, cb); self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished); self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker_thread.quit); self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater); self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker); self.worker_thread.start()

    @QtCore.Slot()
    def _clear_worker(self) -> None:
        self.worker_thread = None; self.worker = None; self._btns_enabled(True); self.sb_prog.setValue(0)

    def _btns_enabled(self, en: bool) -> None:
        for b in self._all_sub_btns: b.setEnabled(en)

    @QtCore.Slot(str, object)
    def _on_finished(self, key: str, result: object) -> None:
        self.sb_prog.setValue(100); self.sb_msg.setText(f"Task completed: {key}"); self._dispatch(key, result)

    @QtCore.Slot(str, str)
    def _on_failed(self, key: str, msg: str) -> None:
        self.sb_prog.setValue(0); self.sb_msg.setText(f"Task failed: {key}")
        self._log(f"[SYSTEM ERROR] {key}: {msg}")
        QtWidgets.QMessageBox.critical(self, f"Task error: {key}", msg)

    def _dispatch(self, key: str, result: object) -> None:
        if key == "1.1_import":
            b: PointCloudBundle = result
            self.context.scans.append(b); self.context.active_index = len(self.context.scans) - 1
            self._render_bundle(b, "1.1 Data Acquisition"); self._update_meta(b)
            n = len(b.points); self.pt_label.setText(f"Points: {n:,}"); self.sb_pts.setText(f"Points: {n:,}")
            self._log(f"Loaded point cloud successfully from: {b.path}")

        elif key == "2.1_voxel":
            pts, _ = result; self.context.normalized_points = pts
            self._render_pts(pts, "2.1 Voxel Grid Filter", "#3B82F6"); self._log(f"Voxel downsampling complete: {len(pts):,} points retained.")

        elif key == "2.2_sor":
            pts, col = result; self.context.normalized_points = pts
            if self.context.active_scan: self.context.active_scan.colors_raw = col
            self._render_pts(pts, "2.2 Statistical Outlier Removal", "#0EA5E9"); self._log(f"Statistical outlier removal complete: {len(pts):,} points retained.")

        elif key == "2.3_lining":
            pts = np.asarray(result, dtype=np.float64); self.context.normalized_points = pts
            self._render_pts(pts, "2.3 Isolated Tunnel Lining", "#6366F1"); self._log(f"Tunnel lining extraction complete: {len(pts):,} points retained.")

        elif key == "3.1_anchor":
            pts = np.asarray(result, dtype=np.float64); self.context.registered_points = pts
            self._render_pts(pts, "3.1 Target Anchor Matrix Applied", "#10B981"); self._log("Target anchor translation matrix applied.")

        elif key == "3.2_icp":
            pts, rmse = result; self.context.registered_points = np.asarray(pts, dtype=np.float64)
            self.context.rmse_mm = rmse; self._render_pts(self.context.registered_points, "3.2 Fine ICP Iterations", "#059669")
            rt = f"{rmse:.3f} mm" if np.isfinite(rmse) else "N/A"
            self.rmse_label.setText(f"RMSE: {rt}"); self.sb_rmse.setText(f"RMSE: {rt}")
            self._log(f"Surface ICP registration complete. Relative RMSE: {rt}")

        elif key == "3.3_rmse":
            rmse = float(result); self.context.rmse_mm = rmse
            rt = f"{rmse:.3f} mm" if np.isfinite(rmse) else "N/A"
            self.rmse_label.setText(f"RMSE: {rt}"); self.sb_rmse.setText(f"RMSE: {rt}")
            self._log(f"Surface model RMSE computed: {rt}")

        elif key == "4.1_centerline":
            cl, fr = result; self.context.centerline = cl; self.context.frenet_frames = fr
            self._render_cl(cl, fr); self._log(f"PCA centerline extracted: {len(cl)} chainage control points.")

        elif key == "4.2_iterative":
            cl, fr, iters = result; self.context.centerline = cl; self.context.frenet_frames = fr
            self._render_cl(cl, fr); self._log(f"Yi (2020) iterative centerline refinement completed after {iters} section-fitting iterations.")

        elif key == "4.3_bspline":
            sm = np.asarray(result, dtype=np.float64); self.context.centerline_smooth = sm
            if self.plotter:
                self.plotter.add_lines(sm, color="#F59E0B", width=4, connected=True, name="cl_sm")
                self.plotter.render()
            self._log(f"B-Spline centerline smoothing complete: {len(sm)} points.")

        elif key == "4.4_frenet":
            self.context.frenet_frames = result; self._log(f"Gravity-aligned section frames generated successfully: {len(result)} N-B frames.")

        elif key == "4.5_seams":
            d: Dict = result; self._log(f"Ring seam detection complete: {d['ring_count']} lining rings segmented, {d['total_seams']} seam boundaries identified.")

        elif key in ("5.1_settlement", "5.2_convergence", "5.5_ovality", "5.6_eccentricity"):
            self.context.parameters.update(result); self._show_params(result)

        elif key == "5.3_heatmap":
            pts, sc = result; self.context.heatmap_scalars = sc; self._render_heatmap(np.asarray(pts, dtype=np.float64), sc)

        elif key == "5.4_polar":
            centers, angles, dmap = result
            self.context.polar_centers = centers; self.context.polar_angles = angles; self.context.polar_map = dmap
            finite = dmap[np.isfinite(dmap)]
            mx = float(np.nanmax(finite)) if finite.size else float("nan")
            mn = float(np.nanmin(finite)) if finite.size else float("nan")
            self.context.parameters.update({"polar_max_outward_mm": mx, "polar_max_inward_mm": mn})
            self.polar_plot.update_data(angles, dmap); self.right_tabs.setCurrentIndex(4)
            self._log(f"Polar radial deformation map generated: max outward={mx:+.2f} mm, max inward={mn:+.2f} mm")

        elif key == "5.7_sections":
            sections: List[SectionGeometry] = result; self.context.sections = sections
            self.section_widget.set_sections(sections, profile=self.context.tunnel_profile, vl_box_w=self._sp_vl_w.value(), vl_box_h=self._sp_vl_h.value(), vl_cir_r=self._sp_vl_r.value())
            self.right_tabs.setCurrentIndex(self._section_tab_idx)
            valid = [s for s in sections if s.pts_2d is not None]
            self._log("--- 2D technical cross-section analysis ---")
            self._log(f"  Total section slices analyzed along the alignment: {len(sections)}")
            if valid:
                w1s = [s.W1 for s in valid if np.isfinite(s.W1)]
                h1s = [s.H1 for s in valid if np.isfinite(s.H1)]
                if w1s: self._log(f"  Average clear section width W1: {np.mean(w1s):.3f} m")
                if h1s: self._log(f"  Average clear section height H1: {np.mean(h1s):.3f} m")
            self._log("------------------------------------------------")

        elif key == "6.1_epochs":
            t0, tn = result; self.context.scans = [t0, tn]; self.context.active_index = 1
            self._log("Time-series point-cloud epochs loaded successfully.")

        elif key == "6.2_plot":
            series = np.asarray(result, dtype=np.float64); self.context.time_series_plot = series
            self.ts_plot.set_values(series, "Deformation Trend Chart Across Chainage Line (mm)")
            self.right_tabs.setCurrentIndex(2)

        elif key == "7.1_ifc":
            self.ai_resp.setPlainText(json.dumps(result, indent=2)); self.right_tabs.setCurrentIndex(self._ai_tab_idx)

        elif key == "7.2_ai":
            self.ai_resp.setPlainText(str(result)); self.right_tabs.setCurrentIndex(self._ai_tab_idx)

    def _slot_1_1_import(self) -> None:
        self._hdr("LiDAR Data Acquisition", "Load LAS/LAZ/PLY point-cloud data into the project database.")
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load tunnel point-cloud data", "", "Point Clouds (*.las *.laz *.ply);;All Files (*.*)")
        if not fp: return
        self._start_worker("1.1_import", lambda: self.base_mod.load_scan(fp))

    def _slot_1_2_viewport(self) -> None:
        self._hdr("Initialize 3D Viewport", "Prepare the PyVista inspection viewport with a light technical theme.")
        if self.plotter:
            self.plotter.clear(); self.plotter.set_background("#F8FAFC"); self.plotter.add_axes(color="#111827")
            self.plotter.show_bounds(color="#94A3B8", grid="front", location="outer", font_size=8); self.plotter.render()
        self._log("3D viewport initialized and refreshed.")

    def _slot_2_1_voxel(self) -> None:
        self._hdr("Voxel Downsampling", "Homogenize point density using a voxel grid while preserving tunnel geometry.")
        self._start_worker("2.1_voxel", lambda: self.pre_mod.voxel_downsample(self.context))

    def _slot_2_2_sor(self) -> None:
        self._hdr("Statistical Outlier Removal", "Remove environmental noise using distance-statistics filtering.")
        self._start_worker("2.2_sor", lambda: self.pre_mod.statistical_outlier_removal_run(self.context))

    def _slot_2_3_lining(self) -> None:
        self._hdr("Tunnel Lining Extraction", "Isolate the structural tunnel lining surface for downstream analysis.")
        self._start_worker("2.3_lining", lambda: self.pre_mod.extract_tunnel_lining(self.context))

    def _slot_3_1_anchor(self) -> None:
        self._hdr("Target Anchor Translation", "Apply the initial target-based translation alignment.")
        self._start_worker("3.1_anchor", lambda: self.reg_mod.anchor_translation(self.context))

    def _slot_3_2_icp(self) -> None:
        self._hdr("Surface ICP Registration", "Refine station alignment with surface-based ICP and report RMSE.")
        self._start_worker("3.2_icp", lambda: self.reg_mod.run_surface_icp(self.context))

    def _slot_3_3_rmse(self) -> None:
        self._hdr("Registration RMSE Check", "Evaluate registration quality using nearest-surface residuals.")
        self._start_worker("3.3_rmse", lambda: self.reg_mod.calculate_rmse(self.context))

    def _slot_4_1_centerline(self) -> None:
        self._hdr("PCA Centerline Extraction", "Extract initial tunnel centerline control points from the working cloud.")
        self._start_worker("4.1_centerline", lambda: self.geo_mod.extract_centerline(self.context))

    def _slot_4_2_iterative(self) -> None:
        self._hdr("Iterative Centerline Refinement", "Refine the tunnel axis using orthogonal section fitting.")
        if self.context.centerline is None: self._log("Run Step 4.1 first."); return
        cl = self.context.centerline
        self._start_worker("4.2_iterative", lambda: self.geo_mod.extract_centerline_iterative(self.context, design_axis=cl, section_count=80, mu=0.03, max_iter=20))

    def _slot_4_3_bspline(self) -> None:
        self._hdr("B-Spline Centerline Smoothing", "Generate a smooth differentiable tunnel axis for sectioning.")
        if self.context.centerline is None: self._log("Run Step 4.1 first."); return
        cl = self.context.centerline; self._start_worker("4.3_bspline", lambda: self.geo_mod.smooth_bspline(cl))

    def _slot_4_4_frenet(self) -> None:
        self._hdr("Gravity-Aligned Section Frames", "Generate Frenet N-B section frames for orthogonal cross-sections.")
        if not self.context.frenet_frames: self._log("Run Step 4.1 first."); return
        fr = self.context.frenet_frames; self._start_worker("4.4_frenet", lambda: self.geo_mod.generate_frenet_planes(fr))

    def _slot_4_5_seams(self) -> None:
        self._hdr("Ring Seam Detection", "Segment tunnel rings and identify seam transition locations.")
        if not self.context.frenet_frames: self._log("Run Step 4.1 first."); return
        def _task():
            rings = self.seg_mod.segment_rings(self.context); cl = self.context.centerline; frs = self.context.frenet_frames
            n = min(len(rings), len(cl) if cl is not None else 0, len(frs))
            total = sum(len(self.seg_mod.detect_seam_boundaries(rings[i], cl[i], frs[i], k_clusters=6)) for i in range(n))
            return {"ring_count": len(rings), "total_seams": total}
        self._start_worker("4.5_seams", _task)

    def _slot_5_1_settlement(self) -> None:
        self._hdr("Crown Settlement", "Extract vertical displacement indicators at the tunnel crown.")
        self._start_worker("5.1_settlement", lambda: self.par_mod.calc_arch_settlement(self.context))

    def _slot_5_2_convergence(self) -> None:
        self._hdr("Horizontal Convergence", "Estimate lateral wall convergence across each tunnel section.")
        self._start_worker("5.2_convergence", lambda: self.par_mod.calc_horizontal_convergence(self.context))

    def _slot_5_3_heatmap(self) -> None:
        self._hdr("3D Deformation Heatmap", "Visualize deformation magnitudes on the tunnel point cloud.")
        self._start_worker("5.3_heatmap", lambda: self.par_mod.generate_heatmap(self.context))

    def _slot_5_4_polar(self) -> None:
        self._hdr("Polar Radial Deformation", "Map radial deformation by angle around each section.")
        if not self.context.frenet_frames or self.context.working_points is None: self._log("Complete Steps 2 and 4 before running this analysis."); return
        self._start_worker("5.4_polar", lambda: self.par_mod.generate_polar_deformation_map(self.context, design_radius_m=3.0, num_bins=72))

    def _slot_5_5_ovality(self) -> None:
        self._hdr("Section Ovality", "Calculate ovality as a geometric distortion indicator.")
        self._start_worker("5.5_ovality", lambda: self.par_mod.calc_ovality(self.context))

    def _slot_5_6_eccentricity(self) -> None:
        self._hdr("Section Eccentricity", "Calculate measured center offset relative to the design center.")
        self._start_worker("5.6_eccentricity", lambda: self.par_mod.calc_eccentricity(self.context))

    def _slot_5_7_sections(self) -> None:
        self._hdr("Plot 2D Technical Section", "Display flat 2D engineering cross-sections with vehicle clearance limits.")
        if not self.context.frenet_frames or self.context.working_points is None: self._log("Complete Steps 2 and 4 before running this analysis."); return
        self.context.tunnel_profile = self._profile_combo.currentText()
        self._start_worker("5.7_sections", lambda: self.par_mod.compute_all_sections(self.context, vl_box_w=self._sp_vl_w.value(), vl_box_h=self._sp_vl_h.value(), vl_cir_r=self._sp_vl_r.value()))

    def _slot_6_1_epochs(self) -> None:
        self._hdr("Load Time-Series Epochs", "Load reference and monitoring point-cloud epochs for deformation comparison.")
        fp0, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load reference epoch T0", "", "Point Clouds (*.las *.laz *.ply);;All Files (*.*)")
        if not fp0: return
        fpn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load monitoring epoch", "", "Point Clouds (*.las *.laz *.ply);;All Files (*.*)")
        if not fpn: return
        self._start_worker("6.1_epochs", lambda: self.ts_mod.load_epochs(fp0, fpn))

    def _slot_6_2_plot(self) -> None:
        self._hdr("Deformation Trend Chart", "Plot deformation trend metrics along the chainage line.")
        self._start_worker("6.2_plot", lambda: self.ts_mod.plot_deformation(self.context))

    def _slot_7_1_ifc(self) -> None:
        self._hdr("IFC/BIM Export Package", "Prepare geometry and parameters for IFC/BIM integration.")
        self._start_worker("7.1_ifc", lambda: self.dt_mod.export_ifc(self.context))

    def _slot_7_2_query_ai(self) -> None:
        self._hdr("AI Engineering Assistant", "Ask the local LLM to summarize tunnel condition and maintenance risks.")
        prompt = self.ai_prompt.toPlainText().strip() or "Summarize the tunnel inspection results and identify locations that require engineering attention."
        self.right_tabs.setCurrentIndex(self._ai_tab_idx)
        self._start_worker("7.2_ai", lambda: self.dt_mod.query_local_ai(prompt, self.context))

    def _render_bundle(self, b: PointCloudBundle, title: str) -> None:
        mesh = b.cloud or make_vertex_cloud(b.points, b.intensity, b.colors_raw); self._render_mesh(mesh, title)

    def _render_pts(self, pts: np.ndarray, title: str, color: str = "#2563EB") -> None:
        self._render_mesh(make_vertex_cloud(pts), title, color=color)

    def _render_mesh(self, mesh: "pv.PolyData", title: str, color: str = None) -> None:
        if self.plotter is None: return
        rgb = mesh.get_array("RGB") if "RGB" in mesh.array_names else None
        clean = make_vertex_cloud(np.asarray(mesh.points, dtype=np.float64), intensity=mesh.get_array("Intensity") if "Intensity" in mesh.array_names else None, colors_raw=rgb.astype(np.float64)/255.0 if rgb is not None else None)
        self.plotter.clear(); self.plotter.set_background("#F8FAFC")
        kw = dict(style="points", point_size=2.4, render_points_as_spheres=False, reset_camera=True)
        if "RGB" in clean.array_names and color is None: self.plotter.add_mesh(clean, scalars="RGB", rgb=True, **kw)
        elif "Intensity" in clean.array_names and color is None: self.plotter.add_mesh(clean, scalars="Intensity", cmap="viridis", **kw)
        else: self.plotter.add_mesh(clean, color=color or "#1D4ED8", **kw)
        self.plotter.add_text(title, position="upper_left", font_size=11, color="#111827", name="ttl")
        self.plotter.add_axes(color="#111827"); self.plotter.show_bounds(color="#94A3B8", grid="front", location="outer", font_size=8)
        self.plotter.camera.parallel_projection = True; self.plotter.reset_camera(); self.plotter.render()

    def _render_cl(self, cl: np.ndarray, fr: List[Dict]) -> None:
        pts = self.context.working_points
        if pts is not None: self._render_pts(pts, "4.x Centerline Frame Calibration", "#CBD5E1")
        if self.plotter is None: return
        self.plotter.add_lines(cl, color="#E11D48", width=5, connected=True, name="cl")
        skip = max(1, len(fr) // 18)
        for i, frame in enumerate(fr[::skip]):
            c = frame["center"]
            for k, col in (("T", "#2563EB"), ("N", "#16A34A"), ("B", "#EA580C")):
                ln = np.vstack([c, c + frame[k] * 0.6]); self.plotter.add_lines(ln, color=col, width=2, connected=True, name=f"f{k}{i}")
        self.plotter.render()

    def _render_heatmap(self, pts: np.ndarray, sc: np.ndarray) -> None:
        mesh = make_vertex_cloud(pts)
        if len(sc) == mesh.n_points: mesh["Delta_mm"] = sc
        if self.plotter is None: return
        self.plotter.clear(); self.plotter.set_background("#F8FAFC")
        self.plotter.add_mesh(mesh, scalars="Delta_mm", cmap="turbo", style="points", point_size=2.8, render_points_as_spheres=False, reset_camera=True, scalar_bar_args={"title": "Delta (mm)"})
        self.plotter.add_text("Heatmap - Vertical Displacement (Z-Axis Deviation)", position="upper_left", font_size=11, color="#111827", name="ttl")
        self.plotter.add_axes(color="#111827"); self.plotter.reset_camera(); self.plotter.render()

    def _hdr(self, title: str, desc: str) -> None:
        self.task_title.setText(title); self.task_desc.setText(desc)

    def _show_params(self, params: Dict[str, float]) -> None:
        self.results_text.appendPlainText("--- Parameters Extracted ---")
        for k, v in params.items(): self.results_text.appendPlainText(f"  {k}: {v:.4f}")
        self.results_text.appendPlainText("----------------------------"); self.right_tabs.setCurrentIndex(0)

    def _update_meta(self, b: PointCloudBundle) -> None:
        rows = list(b.metadata.items()); self.meta_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.meta_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(k))); self.meta_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(v)))
        self.right_tabs.setCurrentIndex(1)

    def _log(self, msg: str) -> None:
        self.results_text.appendPlainText(str(msg))

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #F1F5F9; color: #111827; font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }
            #Sidebar { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
            #ProductTitle { color: #0F4C81; font-size: 15pt; font-weight: 800; letter-spacing: 0.5px; }
            #LabSubtitle  { color: #64748B; font-size: 9pt; padding-bottom: 4px; }
            #Separator    { color: #E2E8F0; margin: 4px 0; }
            QScrollArea   { background: transparent; border: none; }
            QToolButton#SectionToggle { background: #EEF4FA; border: 1px solid #D1DCEB; border-radius: 6px; padding: 6px 10px; font-weight: 600; color: #1E3A5F; text-align: left; }
            QToolButton#SectionToggle:hover   { background: #DBEAFE; border-color: #3B82F6; }
            QToolButton#SectionToggle:checked { background: #BFDBFE; border-color: #1D4ED8; }
            QWidget#SectionContent { background: #F8FAFC; border-left: 2px solid #BFDBFE; margin-left: 10px; }
            QPushButton#SubButton { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 5px; padding: 6px 10px; text-align: left; color: #334155; font-size: 9.5pt; }
            QPushButton#SubButton:hover    { background: #EFF6FF; border-color: #3B82F6; color: #1D4ED8; }
            QPushButton#SubButton:disabled { background: #F1F5F9; color: #94A3B8; border-color: #E2E8F0; }
            QPushButton { background: #EEF4FA; border: 1px solid #CBD6E2; border-radius: 6px; padding: 8px 12px; font-weight: 600; }
            QPushButton:hover { background: #DBEAFE; border-color: #2563EB; }
            #Header, #ViewportFrame { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; }
            #TaskTitle       { color: #0F172A; font-size: 14pt; font-weight: 700; }
            #TaskDescription { color: #475569; }
            QTabWidget::pane, QPlainTextEdit, QTableWidget { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; }
            QHeaderView::section { background: #EEF4FA; border: 1px solid #E2E8F0; padding: 5px; }
            QProgressBar { background: #EEF4FA; border: 1px solid #CBD5E1; border-radius: 4px; text-align: center; min-width: 140px; }
            QProgressBar::chunk { background: #2563EB; border-radius: 4px; }
            QDoubleSpinBox, QComboBox { background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px; color: #111827; }
        """)

def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Tunnel Analysis v4.0")
    win = TunnelAnalysisWindow(); win.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())





