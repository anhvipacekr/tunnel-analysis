from .common import *
from .models import PipelineContext
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
        self, context: PipelineContext, k_sigma: float = 2.5, section_len: float = 0.5
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Improved SOR per PDF 3.2:
        1. Partition tunnel into sections along dominant axis
        2. Per section: compute radial distance from axis
        3. Use ROBUST median + MAD (not mean+std) to handle bimodal distribution
        4. Remove points outside [R_med - k*MAD, R_med + k*MAD]
        5. Also remove points with R < R_med * 0.4 (interior objects)
        """
        scan = context.active_scan
        if scan is None: raise RuntimeError("SOR: no active scan.")
        pts = validate_xyz(scan.points); colors = scan.colors_raw; N = len(pts)

        # Dominant axis
        centroid = pts.mean(0); centred = pts - centroid
        ev, vecs = np.linalg.eigh(np.cov(centred.T))
        long_ax = vecs[:, np.argmax(ev)]
        proj = centred @ long_ax
        pmin, pmax = float(proj.min()), float(proj.max())
        ns = max(1, int(np.ceil((pmax - pmin) / section_len)))

        inlier = np.zeros(N, dtype=bool)
        for s in range(ns):
            lo = pmin + s * section_len
            hi = pmin + (s + 1) * section_len
            if s == ns - 1: hi = pmax + 1e-9
            mask = (proj >= lo) & (proj < hi)
            idx = np.where(mask)[0]
            if len(idx) < 6:
                inlier[idx] = True; continue

            sp = pts[idx]
            ao = centroid + float(proj[idx].mean()) * long_ax
            diff = sp - ao
            ax_c = (diff @ long_ax)[:, None] * long_ax
            ri = np.linalg.norm(diff - ax_c, axis=1)

            # Robust: use median + MAD
            R_med = float(np.median(ri))
            if R_med < 1e-4:
                inlier[idx] = True; continue
            mad = float(np.median(np.abs(ri - R_med))) + 1e-9
            # 1.4826 converts MAD to sigma-equivalent
            thr = k_sigma * 1.4826 * mad

            # Keep: within band AND not interior object (R > R_med * 0.4)
            band = (np.abs(ri - R_med) <= thr) & (ri >= R_med * 0.40)
            inlier[idx[band]] = True

        cleaned = validate_xyz(pts[inlier])
        cout: Optional[np.ndarray] = None
        if colors is not None:
            raw = np.asarray(colors, dtype=np.float64)
            if raw.shape[0] == N: cout = _normalize_rgb(raw[inlier])
        n_removed = N - int(inlier.sum())
        return cleaned, cout, {
            "n_raw": N, "n_clean": int(inlier.sum()),
            "n_removed": n_removed,
            "outlier_pts": pts[~inlier].copy()
        }

    def extract_tunnel_lining(self, context: PipelineContext) -> np.ndarray:
        """Extract tunnel lining surface per PDF 3.2.

        Multi-pass strategy:
        Pass 1 - Axis estimation: PCA dominant axis
        Pass 2 - Coarse radius band: keep [R_med*0.5, R_med*1.5] per section
        Pass 3 - Fine statistical filter: keep within mu +/- 2.5*sigma of radial deviation
        Pass 4 - Intensity-based filter: if intensity available, remove low-intensity
                 interior objects (cables/lights have different reflectance)
        """
        pts = context.working_points
        if pts is None: raise RuntimeError("No working_points.")
        pts = validate_xyz(pts)
        scan = context.active_scan
        intensity = None
        if scan is not None and scan.intensity is not None:
            raw_int = np.asarray(scan.intensity, dtype=np.float64).ravel()
            # align intensity to working_points if sizes match
            if len(raw_int) == len(pts):
                intensity = raw_int

        # Pass 1: dominant axis
        c = pts.mean(axis=0)
        ev, vecs = np.linalg.eigh(np.cov((pts - c).T))
        ax = vecs[:, np.argmax(ev)]
        proj = (pts - c) @ ax
        pmin, pmax = float(proj.min()), float(proj.max())
        section_len = max(0.3, (pmax - pmin) / 60.0)
        ns = max(1, int(np.ceil((pmax - pmin) / section_len)))

        keep = np.zeros(len(pts), dtype=bool)
        for s in range(ns):
            lo = pmin + s * section_len
            hi = pmin + (s + 1) * section_len
            if s == ns - 1: hi = pmax + 1e-9
            mask = (proj >= lo) & (proj < hi)
            idx = np.where(mask)[0]
            if len(idx) < 6:
                keep[idx] = True; continue

            sp = pts[idx]
            ao = c + float(proj[idx].mean()) * ax
            diff = sp - ao
            ax_c = (diff @ ax)[:, None] * ax
            ri = np.linalg.norm(diff - ax_c, axis=1)
            R_med = float(np.median(ri))
            R_std = float(np.std(ri)) + 1e-6

            # Pass 2: coarse band
            coarse = (ri >= R_med * 0.50) & (ri <= R_med * 1.50)
            if coarse.sum() < 6:
                keep[idx] = True; continue

            # Pass 3: fine statistical filter on coarse subset
            ri_c = ri[coarse]
            mu_c = float(np.mean(ri_c)); si_c = float(np.std(ri_c)) + 1e-6
            fine = coarse & (ri >= mu_c - 2.5 * si_c) & (ri <= mu_c + 2.5 * si_c)

            # Pass 4: intensity filter (remove very low intensity = interior objects)
            if intensity is not None:
                int_sec = intensity[idx]
                int_med = float(np.median(int_sec[fine])) if fine.sum() > 0 else float(np.median(int_sec))
                int_thr = int_med * 0.35
                int_ok  = int_sec >= int_thr
                final   = fine & int_ok
                if final.sum() >= 6:
                    keep[idx[final]] = True
                    continue
            keep[idx[fine]] = True

        result = validate_xyz(pts[keep])

        # Global pass: remove points far outside expected tunnel radius
        if len(result) >= 10:
            c2 = result.mean(axis=0)
            ev2, vecs2 = np.linalg.eigh(np.cov((result - c2).T))
            ax2 = vecs2[:, np.argmax(ev2)]
            diff2 = result - c2
            ax_c2 = (diff2 @ ax2)[:, None] * ax2
            ri2 = np.linalg.norm(diff2 - ax_c2, axis=1)
            R_global = float(np.median(ri2))
            mad_global = float(np.median(np.abs(ri2 - R_global))) + 1e-9
            thr_global = 3.0 * 1.4826 * mad_global
            global_keep = (np.abs(ri2 - R_global) <= thr_global) & (ri2 >= R_global * 0.35)
            result = validate_xyz(result[global_keep])

        return result


