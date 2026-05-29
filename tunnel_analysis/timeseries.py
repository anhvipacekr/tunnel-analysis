"""
timeseries.py - Time-series deformation analysis per PDF section 3.5.
T0/Tn cloud-to-cloud displacement using kd-tree nearest neighbour + ICP alignment.
"""
from .common import *
from .models import PointCloudBundle, PipelineContext
from .io_layer import BaseLayer
from .registration import RegistrationLayer


class TimeSeriesLayer:

    def load_epochs(self, p0: str, pn: str) -> Tuple[PointCloudBundle, PointCloudBundle]:
        bl = BaseLayer()
        return bl.load_scan(p0), bl.load_scan(pn)

    def compute_cloud_to_cloud(
        self,
        context: PipelineContext,
        n_sections: int = 120,
        max_dist_mm: float = 200.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Cloud-to-cloud displacement T0 -> Tn per PDF 3.5.

        Steps:
        1. ICP-align Tn onto T0 coordinate system
        2. For each point in Tn: find nearest neighbour in T0 (kd-tree)
        3. Signed displacement = distance in mm (positive = outward/expansion)
        4. Bin along tunnel axis -> per-section mean displacement curve
        5. Return: (Tn_points, displacement_mm_per_point, stats_dict)
        """
        if len(context.scans) < 2:
            raise RuntimeError("Load T0 and Tn epochs first (Step 6.1).")

        pts_t0 = validate_xyz(context.scans[0].points)
        pts_tn = validate_xyz(context.scans[1].points)

        # Step 1: ICP align Tn -> T0
        reg = RegistrationLayer()
        pts_tn_aligned, rmse = reg._icp(pts_tn, pts_t0)

        # Step 2: kd-tree nearest neighbour
        if cKDTree is None:
            raise RuntimeError("scipy required for cloud-to-cloud analysis.")
        tree_t0 = cKDTree(pts_t0)
        dist_m, _ = tree_t0.query(pts_tn_aligned, k=1, workers=-1)
        dist_mm = dist_m * 1000.0

        # Clamp outliers
        dist_mm = np.clip(dist_mm, 0.0, max_dist_mm)

        # Step 3: stats
        stats = {
            "c2c_mean_mm":   float(np.nanmean(dist_mm)),
            "c2c_median_mm": float(np.nanmedian(dist_mm)),
            "c2c_max_mm":    float(np.nanmax(dist_mm)),
            "c2c_p95_mm":    float(np.nanpercentile(dist_mm, 95)),
            "icp_rmse_mm":   float(rmse),
            "n_points_tn":   int(len(pts_tn_aligned)),
            "n_points_t0":   int(len(pts_t0)),
        }

        return pts_tn_aligned, dist_mm, stats

    def plot_deformation(
        self,
        context: PipelineContext,
        n_sections: int = 120,
    ) -> np.ndarray:
        """Per-section mean displacement curve along tunnel axis.

        Returns array of shape (n_sections,) with mean displacement in mm.
        Used by LinePlotWidget for the deformation trend chart.
        """
        pts_tn_aligned, dist_mm, stats = self.compute_cloud_to_cloud(
            context, n_sections=n_sections)

        # Bin along dominant axis
        if context.centerline is not None and len(context.centerline) >= 2:
            cl = validate_xyz(context.centerline)
            center = cl.mean(axis=0)
            ev, vecs = np.linalg.eigh(np.cov((cl - center).T))
            axis = vecs[:, np.argmax(ev)]
        else:
            center = pts_tn_aligned.mean(axis=0)
            ev, vecs = np.linalg.eigh(np.cov((pts_tn_aligned - center).T))
            axis = vecs[:, np.argmax(ev)]

        proj = (pts_tn_aligned - center) @ axis
        order = np.argsort(proj)
        dist_sorted = dist_mm[order]

        chunks = np.array_split(dist_sorted, n_sections)
        series = np.array(
            [float(np.nanmean(c)) for c in chunks if len(c) > 0],
            dtype=np.float64,
        )
        return series

    def compute_per_section_stats(
        self,
        context: PipelineContext,
    ) -> List[Dict[str, float]]:
        """Per Frenet-section displacement stats for detailed reporting."""
        if not context.frenet_frames:
            raise RuntimeError("Run centerline first (Step 4.x).")

        pts_tn_aligned, dist_mm, _ = self.compute_cloud_to_cloud(context)

        results = []
        eps = 0.10
        for fr in context.frenet_frames:
            C, T = fr["center"], fr["T"]
            mask = np.abs((pts_tn_aligned - C) @ T) < eps
            if mask.sum() < 5:
                continue
            d_sec = dist_mm[mask]
            results.append({
                "chainage_m":  float(np.linalg.norm(C - context.frenet_frames[0]["center"])),
                "mean_mm":     float(np.nanmean(d_sec)),
                "max_mm":      float(np.nanmax(d_sec)),
                "p95_mm":      float(np.nanpercentile(d_sec, 95)),
                "n_points":    int(mask.sum()),
            })
        return results
