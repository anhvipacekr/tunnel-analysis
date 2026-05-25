"""
target_detector.py - Improved target detection (Faro SCENE style).
Key improvements:
  - Fast DBSCAN using scipy.spatial (vs pure Python loop)
  - Point spacing estimation for adaptive parameters
  - Checkerboard: FFT-based grid pattern detection
  - Sphere: multi-scale RANSAC with standard size matching
  - Confidence scoring per Faro SCENE methodology
"""
from .common import *
from .models import PointCloudBundle
from dataclasses import dataclass, field
import uuid


# ── Standard target sizes (Faro/Leica) ────────────────────────────────────
SPHERE_RADII_STD   = [0.0725, 0.100, 0.145, 0.200]   # m
CHECKER_SIZES_STD  = [0.100, 0.150, 0.200, 0.300]     # m (full board side)


@dataclass
class Target:
    id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:        str   = ""
    type:        str   = "unknown"
    center:      Optional[np.ndarray] = None
    normal:      Optional[np.ndarray] = None
    radius:      float = float("nan")
    intensity:   float = float("nan")
    confidence:  float = 0.0
    n_points:    int   = 0
    residual_mm: float = float("nan")
    scan_idx:    int   = -1
    matched_id:  str   = ""

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items()}


class TargetDetector:
    SPHERE_RADII = SPHERE_RADII_STD

    # ── Public API ─────────────────────────────────────────────────────────
    def detect_all(
        self, bundle: PointCloudBundle, scan_idx: int = 0,
        detect_sphere: bool = True, detect_flat: bool = True,
        detect_intensity: bool = True,
        sphere_radius_range: Tuple[float, float] = (0.05, 0.25),
        intensity_percentile: float = 95.0,
        min_cluster_pts: int = 10,
        cell_size_range: Tuple[float, float] = (0.05, 0.30),
        min_contrast_ratio: float = 2.0,
    ) -> List[Target]:
        pts = validate_xyz(bundle.points)
        intensity = (np.asarray(bundle.intensity, dtype=np.float64).ravel()
                     if bundle.intensity is not None else None)

        # Estimate point spacing for adaptive eps
        spacing = self._estimate_spacing(pts)
        targets: List[Target] = []

        if detect_intensity and intensity is not None:
            targets.extend(self.detect_intensity_targets(
                pts, intensity, scan_idx=scan_idx,
                percentile=intensity_percentile,
                min_cluster_pts=min_cluster_pts,
                eps=spacing * 8))

        if detect_sphere:
            targets.extend(self.detect_sphere_targets(
                pts, scan_idx=scan_idx,
                radius_range=sphere_radius_range,
                min_cluster_pts=min_cluster_pts,
                eps=max(spacing * 6, sphere_radius_range[1] * 2.5)))

        if detect_flat:
            targets.extend(self.detect_checkerboard_targets(
                pts, intensity=intensity, scan_idx=scan_idx,
                min_cluster_pts=min_cluster_pts,
                cell_size_range=cell_size_range,
                min_contrast_ratio=min_contrast_ratio,
                eps=max(spacing * 8, 0.05)))

        # Auto-name
        counts = {}
        for t in targets:
            k = t.type[0].upper()
            counts[k] = counts.get(k, 0) + 1
            if not t.name:
                t.name = f"{k}{counts[k]:02d}"
        return targets

    # ── 1. Sphere detection ────────────────────────────────────────────────
    def detect_sphere_targets(
        self, pts: np.ndarray, scan_idx: int = 0,
        radius_range: Tuple[float, float] = (0.05, 0.25),
        min_cluster_pts: int = 10,
        n_ransac: int = 300, tol: float = 0.008,
        eps: float = 0.3,
    ) -> List[Target]:
        pts = validate_xyz(pts)
        clusters = self._fast_dbscan(pts, eps=eps, min_pts=min_cluster_pts)
        targets: List[Target] = []
        for cp in clusters:
            if len(cp) < min_cluster_pts: continue
            result = self._ransac_sphere(cp, radius_range, n_ransac, tol)
            if result is None: continue
            center, radius, inliers, residual = result
            n_in = int(inliers.sum())
            conf = min(1.0, n_in / max(len(cp), 1))
            std_match = any(abs(radius - r) < 0.015 for r in self.SPHERE_RADII)
            conf = min(1.0, conf + (0.2 if std_match else 0.0))
            if conf < 0.25: continue
            targets.append(Target(
                type="sphere", center=center, radius=radius,
                confidence=conf, n_points=n_in,
                residual_mm=residual * 1e3, scan_idx=scan_idx))
        return targets

    # ── 2. Checkerboard detection (FFT grid analysis) ──────────────────────
    def detect_checkerboard_targets(
        self, pts: np.ndarray, intensity: Optional[np.ndarray] = None,
        scan_idx: int = 0, min_cluster_pts: int = 30,
        max_cluster_pts: int = 10000,
        cell_size_range: Tuple[float, float] = (0.05, 0.30),
        min_contrast_ratio: float = 2.0, eps: float = 0.1,
    ) -> List[Target]:
        pts = validate_xyz(pts)
        targets: List[Target] = []

        # Use intensity-weighted clustering if available
        if intensity is not None and len(intensity) == len(pts):
            int_arr = np.asarray(intensity, dtype=np.float64)
            thr = float(np.percentile(int_arr, 40))
            mask = int_arr >= thr
            if mask.sum() >= min_cluster_pts:
                pts_use = pts[mask]
                int_use = int_arr[mask]
            else:
                pts_use = pts; int_use = int_arr
        else:
            pts_use = pts; int_use = None

        clusters = self._fast_dbscan(pts_use, eps=eps, min_pts=min_cluster_pts)

        for cp in clusters:
            if len(cp) < min_cluster_pts: continue
            if len(cp) > max_cluster_pts: continue

            # Fit plane
            result = self._fit_plane(cp, tol=0.012)
            if result is None: continue
            normal, centroid, thickness, inliers = result
            if thickness > 0.030: continue

            inlier_pts = cp[inliers]
            if len(inlier_pts) < min_cluster_pts: continue

            # Project onto plane
            u = _unit(np.cross(normal, np.array([0,0,1])
                               if abs(normal[2]) < 0.9 else np.array([1,0,0])))
            v = np.cross(normal, u)
            pu = inlier_pts @ u
            pv = inlier_pts @ v
            width  = float(pu.max() - pu.min())
            height = float(pv.max() - pv.min())

            # Size check
            min_sz = cell_size_range[0] * 2
            max_sz = cell_size_range[1] * 12
            if not (min_sz <= width <= max_sz and min_sz <= height <= max_sz):
                continue

            # Get intensity for inlier points
            if int_use is not None:
                if cKDTree is not None:
                    tree = cKDTree(pts_use)
                    _, idx = tree.query(inlier_pts, k=1, workers=-1)
                    int_inliers = int_use[idx]
                else:
                    int_inliers = np.ones(len(inlier_pts))
            else:
                int_inliers = np.ones(len(inlier_pts))

            # FFT-based grid pattern detection
            grid_score, cell_size = self._fft_grid_score(
                pu, pv, int_inliers, cell_size_range)

            # Contrast check
            p75 = float(np.percentile(int_inliers, 75))
            p25 = float(np.percentile(int_inliers, 25))
            contrast = p75 / max(p25, 1e-6)

            if contrast < min_contrast_ratio and grid_score < 0.3:
                continue

            conf = min(1.0, 0.4 * min(contrast / 3.0, 1.0) +
                            0.4 * grid_score +
                            0.2 * (1.0 - thickness / 0.030))

            # Check if size matches standard
            std_match = any(abs(max(width, height) - s) < 0.03
                            for s in CHECKER_SIZES_STD)
            if std_match: conf = min(1.0, conf + 0.15)

            if conf < 0.20: continue
            targets.append(Target(
                type="checkerboard", center=centroid, normal=normal,
                confidence=conf, n_points=len(inliers),
                residual_mm=thickness * 1e3, scan_idx=scan_idx))

        return targets

    # ── 3. Intensity-based detection ───────────────────────────────────────
    def detect_intensity_targets(
        self, pts: np.ndarray, intensity: np.ndarray,
        scan_idx: int = 0, percentile: float = 97.0,
        min_cluster_pts: int = 20, max_cluster_pts: int = 3000,
        eps: float = 0.15,
    ) -> List[Target]:
        pts = validate_xyz(pts)
        intensity = np.asarray(intensity, dtype=np.float64)
        if len(intensity) != len(pts): return []
        thr = float(np.percentile(intensity, percentile))
        mask = intensity >= thr
        if mask.sum() < min_cluster_pts: return []
        hi_pts = pts[mask]; hi_int = intensity[mask]
        clusters = self._fast_dbscan(hi_pts, eps=eps, min_pts=min_cluster_pts)
        targets: List[Target] = []
        for cp in clusters:
            n = len(cp)
            if n < min_cluster_pts or n > max_cluster_pts: continue
            center = cp.mean(axis=0)
            dists = np.linalg.norm(hi_pts - center, axis=1)
            nearby = dists < eps
            peak = float(hi_int[nearby].max()) if nearby.any() else thr
            conf = min(1.0, (peak - thr) / max(intensity.max() - thr, 1e-9))
            targets.append(Target(
                type="intensity", center=center, intensity=peak,
                confidence=conf, n_points=n, scan_idx=scan_idx))
        return targets

    # ── 4. Manual target ───────────────────────────────────────────────────
    def add_manual_target(
        self, position: np.ndarray, scan_idx: int = 0,
        name: str = "", refine_radius: float = 0.1,
        pts: Optional[np.ndarray] = None,
    ) -> Target:
        center = np.asarray(position, dtype=np.float64)
        n_pts = 1
        if pts is not None and cKDTree is not None:
            tree = cKDTree(pts)
            idx = tree.query_ball_point(center, refine_radius)
            if len(idx) >= 3:
                center = pts[idx].mean(axis=0); n_pts = len(idx)
        return Target(type="manual", center=center, confidence=1.0,
                      n_points=n_pts, scan_idx=scan_idx, name=name or "M01")

    # ── Registration ───────────────────────────────────────────────────────
    def register_by_targets(
        self, src_targets: List[Target], tgt_targets: List[Target],
    ) -> Tuple[np.ndarray, float, List[Tuple[str, str, float]]]:
        pairs = [(s, t) for s in src_targets
                 for t in tgt_targets if s.matched_id == t.id
                 and s.center is not None and t.center is not None]
        if len(pairs) < 3:
            raise RuntimeError(f"Need >= 3 matched pairs, got {len(pairs)}.")
        src_pts = np.array([p[0].center for p in pairs])
        tgt_pts = np.array([p[1].center for p in pairs])
        T, rmse = self._horn_svd(src_pts, tgt_pts)
        ones = np.ones((len(src_pts), 1))
        src_reg = (T @ np.hstack([src_pts, ones]).T).T[:, :3]
        residuals = [(p[0].id, p[1].id,
                      float(np.linalg.norm(src_reg[i] - tgt_pts[i])) * 1e3)
                     for i, p in enumerate(pairs)]
        return T, rmse, residuals

    def match_targets(
        self, src_targets: List[Target], tgt_targets: List[Target],
        max_dist: float = 2.0, same_type: bool = True,
    ) -> List[Tuple[Target, Target, float]]:
        matches = []; used = set()
        for st in src_targets:
            if st.center is None: continue
            best_d, best_tt = max_dist, None
            for tt in tgt_targets:
                if tt.id in used or tt.center is None: continue
                if same_type and st.type != tt.type: continue
                d = float(np.linalg.norm(st.center - tt.center))
                if d < best_d: best_d = d; best_tt = tt
            if best_tt:
                st.matched_id = best_tt.id; best_tt.matched_id = st.id
                used.add(best_tt.id)
                matches.append((st, best_tt, best_d))
        return matches

    def apply_transform(self, pts: np.ndarray, T: np.ndarray) -> np.ndarray:
        pts = validate_xyz(pts)
        ones = np.ones((len(pts), 1))
        return (T @ np.hstack([pts, ones]).T).T[:, :3]

    # ── Private helpers ────────────────────────────────────────────────────
    @staticmethod
    def _estimate_spacing(pts: np.ndarray, n_sample: int = 500) -> float:
        """Estimate average point spacing using kNN."""
        if cKDTree is None or len(pts) < 10: return 0.05
        step = max(1, len(pts) // n_sample)
        sample = pts[::step]
        tree = cKDTree(pts)
        d, _ = tree.query(sample, k=2, workers=-1)
        return float(np.median(d[:, 1]))

    @staticmethod
    def _fast_dbscan(pts: np.ndarray, eps: float, min_pts: int) -> List[np.ndarray]:
        """Fast DBSCAN using sklearn (memory efficient)."""
        if len(pts) < min_pts: return []
        try:
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_pts, algorithm="ball_tree",
                        n_jobs=-1).fit(pts)
            labels = db.labels_
            unique = set(labels) - {-1}
            return [pts[labels == c] for c in unique
                    if (labels == c).sum() >= min_pts]
        except ImportError:
            pass
        # Fallback: simple grid-based clustering
        if cKDTree is None: return []
        tree = cKDTree(pts)
        n = len(pts)
        labels = np.full(n, -1, dtype=np.int32)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)
        for i in range(n):
            if visited[i]: continue
            nb = tree.query_ball_point(pts[i], eps)
            if len(nb) < min_pts: continue
            visited[i] = True
            labels[i] = cluster_id
            stack = [j for j in nb if j != i]
            while stack:
                j = stack.pop()
                if visited[j]: continue
                visited[j] = True
                labels[j] = cluster_id
                nb2 = tree.query_ball_point(pts[j], eps)
                if len(nb2) >= min_pts:
                    stack.extend([k for k in nb2 if not visited[k]])
            cluster_id += 1
        return [pts[labels == c] for c in range(cluster_id)
                if (labels == c).sum() >= min_pts]

    @staticmethod
    def _fft_grid_score(
        pu: np.ndarray, pv: np.ndarray, intensity: np.ndarray,
        cell_size_range: Tuple[float, float], grid_res: int = 64,
    ) -> Tuple[float, float]:
        """FFT-based checkerboard grid pattern score.
        Returns (score 0-1, estimated cell size m).
        """
        try:
            # Rasterize intensity onto 2D grid
            u_min, u_max = pu.min(), pu.max()
            v_min, v_max = pv.min(), pv.max()
            if u_max - u_min < 1e-6 or v_max - v_min < 1e-6:
                return 0.0, 0.0
            ui = np.clip(((pu - u_min) / (u_max - u_min) * (grid_res - 1))
                         .astype(int), 0, grid_res - 1)
            vi = np.clip(((pv - v_min) / (v_max - v_min) * (grid_res - 1))
                         .astype(int), 0, grid_res - 1)
            grid = np.zeros((grid_res, grid_res), dtype=np.float64)
            count = np.zeros((grid_res, grid_res), dtype=np.int32)
            for k in range(len(pu)):
                grid[vi[k], ui[k]] += intensity[k]
                count[vi[k], ui[k]] += 1
            mask = count > 0
            grid[mask] /= count[mask]
            # Fill empty cells with mean
            grid[~mask] = grid[mask].mean() if mask.any() else 0.0
            # Normalize
            g_min, g_max = grid.min(), grid.max()
            if g_max - g_min < 1e-6: return 0.0, 0.0
            grid_norm = (grid - g_min) / (g_max - g_min)
            # 2D FFT
            fft = np.abs(np.fft.fft2(grid_norm - 0.5))
            fft[0, 0] = 0  # remove DC
            # Find dominant frequency
            fft_shift = np.fft.fftshift(fft)
            cy, cx = grid_res // 2, grid_res // 2
            # Look for peaks in frequency domain
            peak_val = float(fft_shift.max())
            total_energy = float(fft_shift.sum()) + 1e-9
            score = min(1.0, peak_val / total_energy * grid_res)
            # Estimate cell size from dominant frequency
            peak_idx = np.unravel_index(fft_shift.argmax(), fft_shift.shape)
            freq_u = abs(peak_idx[1] - cx) / grid_res
            freq_v = abs(peak_idx[0] - cy) / grid_res
            freq = max(freq_u, freq_v, 1e-6)
            width = u_max - u_min; height = v_max - v_min
            cell_u = (width * freq) if freq_u > freq_v else (height * freq)
            cell_size = float(np.clip(cell_u, cell_size_range[0], cell_size_range[1]))
            return score, cell_size
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _ransac_sphere(
        pts: np.ndarray, radius_range: Tuple[float, float],
        n_iter: int = 300, tol: float = 0.008,
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray, float]]:
        if len(pts) < 4: return None
        rng = np.random.default_rng(42)
        best_n, best_c = 0, pts.mean(axis=0)
        best_r = float(np.median(np.linalg.norm(pts - best_c, axis=1)))
        best_mask = np.zeros(len(pts), dtype=bool)
        for _ in range(n_iter):
            idx = rng.choice(len(pts), 4, replace=False)
            try: c, r = TargetDetector._fit_sphere_4pts(pts[idx])
            except Exception: continue
            if not (radius_range[0] <= r <= radius_range[1]): continue
            mask = np.abs(np.linalg.norm(pts - c, axis=1) - r) < tol
            n = int(mask.sum())
            if n > best_n:
                best_n = n; best_mask = mask
                if n >= 4:
                    c2, r2 = TargetDetector._lsq_sphere(pts[mask])
                    if radius_range[0] <= r2 <= radius_range[1]:
                        best_c = c2; best_r = r2
        if best_n < 4: return None
        res = float(np.sqrt(np.mean(
            np.abs(np.linalg.norm(pts[best_mask] - best_c, axis=1) - best_r) ** 2)))
        return best_c, best_r, best_mask, res

    @staticmethod
    def _fit_sphere_4pts(pts):
        A = np.column_stack([2*pts, np.ones(4)])
        b = (pts**2).sum(axis=1)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        c = x[:3]; r = float(np.sqrt(max(x[3] + np.dot(c, c), 1e-9)))
        return c, r

    @staticmethod
    def _lsq_sphere(pts):
        A = np.column_stack([2*pts, np.ones(len(pts))])
        b = (pts**2).sum(axis=1)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        c = x[:3]; r = float(np.sqrt(max(x[3] + np.dot(c, c), 1e-9)))
        return c, r

    @staticmethod
    def _fit_plane(pts, n_iter=200, tol=0.008):
        if len(pts) < 3: return None
        rng = np.random.default_rng(42)
        best_inliers = np.array([], dtype=int)
        best_n = np.array([0,0,1], dtype=np.float64)
        best_c = pts.mean(axis=0)
        for _ in range(n_iter):
            idx = rng.choice(len(pts), 3, replace=False)
            p0,p1,p2 = pts[idx]
            n = np.cross(p1-p0, p2-p0)
            nn = float(np.linalg.norm(n))
            if nn < 1e-10: continue
            n /= nn
            d = np.abs((pts - p0) @ n)
            inliers = np.where(d < tol)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers; best_n = n
                best_c = pts[inliers].mean(axis=0)
        if len(best_inliers) < 3: return None
        thick = float(np.abs((pts[best_inliers] - best_c) @ best_n).max())
        return best_n, best_c, thick, best_inliers

    @staticmethod
    def _horn_svd(src, tgt):
        sc = src.mean(0); tc = tgt.mean(0)
        H = (src - sc).T @ (tgt - tc)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        t = tc - R @ sc
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
        ones = np.ones((len(src),1))
        reg = (T @ np.hstack([src,ones]).T).T[:,:3]
        rmse = float(np.sqrt(np.mean(np.linalg.norm(reg-tgt,axis=1)**2)))*1e3
        return T, rmse
