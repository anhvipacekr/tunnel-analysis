from .common import *
from .models import PipelineContext
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


    def detect_ring_seams_by_intensity(
        self,
        context: PipelineContext,
        bin_width: float = 0.10,
        min_prominence: float = 2.0,
    ) -> Dict[str, np.ndarray]:
        """Detect tunnel ring seams from axial intensity derivative.

        PDF section 3.3 describes intensity drops at concrete ring joints.
        This method bins intensity along the tunnel axis, smooths the signal,
        computes the first derivative, and returns valley/drop locations.
        """
        scan = context.active_scan
        pts = context.working_points
        if scan is None or pts is None:
            raise RuntimeError("No active scan/working points.")
        pts = validate_xyz(pts)
        if scan.intensity is None:
            raise RuntimeError("Intensity data is required for seam detection.")
        intensity = np.asarray(scan.intensity, dtype=np.float64).ravel()
        if len(intensity) != len(scan.points):
            raise RuntimeError("Intensity length does not match point count.")

        # Align intensity to working_points if they are exactly active scan points.
        if len(intensity) != len(pts):
            raise RuntimeError("Intensity-based seams require unfiltered active scan points.")

        if context.centerline is not None and len(context.centerline) >= 2:
            cl = validate_xyz(context.centerline, "centerline")
            center = cl.mean(axis=0)
            ev, vecs = np.linalg.eigh(np.cov((cl - center).T))
            axis = vecs[:, np.argmax(ev)]
        else:
            center = pts.mean(axis=0)
            ev, vecs = np.linalg.eigh(np.cov((pts - center).T))
            axis = vecs[:, np.argmax(ev)]

        proj = (pts - center) @ axis
        pmin, pmax = float(np.nanmin(proj)), float(np.nanmax(proj))
        if pmax - pmin < bin_width:
            raise RuntimeError("Scan length too short for seam detection.")

        edges = np.arange(pmin, pmax + bin_width, bin_width)
        if len(edges) < 4:
            raise RuntimeError("Too few bins for seam detection.")
        centers = 0.5 * (edges[:-1] + edges[1:])
        bidx = np.clip(np.digitize(proj, edges) - 1, 0, len(centers) - 1)
        signal = np.full(len(centers), np.nan, dtype=np.float64)
        counts = np.zeros(len(centers), dtype=np.int64)
        for i in range(len(centers)):
            mask = bidx == i
            counts[i] = int(mask.sum())
            if counts[i] >= 5:
                signal[i] = float(np.nanmedian(intensity[mask]))

        finite = np.isfinite(signal)
        if finite.sum() < 5:
            raise RuntimeError("Insufficient valid intensity bins.")
        signal = np.interp(centers, centers[finite], signal[finite])

        # Robust moving average smoothing.
        win = max(3, int(round(0.5 / max(bin_width, 1e-6))))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float64) / win
        smooth = np.convolve(signal, kernel, mode="same")
        derivative = np.gradient(smooth, centers)

        # Large negative derivative = intensity drop valley/seam.
        med = float(np.nanmedian(derivative))
        mad = float(np.nanmedian(np.abs(derivative - med))) + 1e-9
        threshold = med - min_prominence * 1.4826 * mad
        candidates = np.where(derivative < threshold)[0]

        # Non-maximum suppression along axis.
        seams: List[int] = []
        min_gap_bins = max(1, int(round(0.7 / max(bin_width, 1e-6))))
        for idx in candidates:
            if not seams or idx - seams[-1] >= min_gap_bins:
                seams.append(int(idx))
            elif derivative[idx] < derivative[seams[-1]]:
                seams[-1] = int(idx)

        seam_chainage = centers[seams] - centers[0] if seams else np.array([], dtype=np.float64)
        seam_projection = centers[seams] if seams else np.array([], dtype=np.float64)
        return {
            "chainage_m": seam_chainage.astype(np.float64),
            "projection_m": seam_projection.astype(np.float64),
            "bin_centers_m": centers.astype(np.float64),
            "intensity_smooth": smooth.astype(np.float64),
            "derivative": derivative.astype(np.float64),
            "threshold": np.array([threshold], dtype=np.float64),
        }

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


