from .common import *
from .models import PipelineContext
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


    def merge_scans(self, context: PipelineContext) -> Tuple[np.ndarray, List[float]]:
        """Merge all loaded scan stations into one point cloud.

        Steps per PDF 3.3:
        1. Use scan[0] as reference (anchor station)
        2. For each subsequent scan: anchor translation -> ICP -> merge
        3. Return merged cloud + per-scan RMSE list
        """
        if len(context.scans) < 2:
            pts = context.working_points
            if pts is None: raise RuntimeError("No scans loaded.")
            return validate_xyz(pts), [0.0]

        # Reference = scan[0]
        tgt = validate_xyz(context.scans[0].points)
        merged = [tgt]
        rmse_list = [0.0]

        for i in range(1, len(context.scans)):
            src_pts = validate_xyz(context.scans[i].points)
            # Step 1: anchor translation
            t_offset = (self._anchor(tgt, context.scans[0].intensity) -
                        self._anchor(src_pts, context.scans[i].intensity))
            src_shifted = src_pts + t_offset
            # Step 2: ICP fine registration
            src_reg, rmse = self._icp(src_shifted, tgt)
            merged.append(src_reg)
            rmse_list.append(rmse)

        merged_cloud = np.vstack(merged)
        return validate_xyz(merged_cloud), rmse_list

    def register_and_merge(self, context: PipelineContext) -> Tuple[np.ndarray, List[float]]:
        """Register all scans to reference and merge — main entry point."""
        return self.merge_scans(context)

    def apply_manual_transform(self, pts: np.ndarray, offset: Tuple[float, float, float], 
                                               rotation: Tuple[float, float, float]) -> np.ndarray:
        """Apply manual translation and rotation (degrees) to point cloud."""
        if pts is None: return None
        pts = validate_xyz(pts)
        
        # Translation
        pts = pts + np.asarray(offset)
        
        # Rotation
        rx, ry, rz = np.radians(rotation)
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        return pts @ R.T


    def register_and_merge_chain(self, context: PipelineContext) -> Tuple[np.ndarray, List[float]]:
        """Chain registration: S1 -> S2 -> S3... per professional software."""
        if len(context.scans) < 2:
            pts = context.working_points
            if pts is None: raise RuntimeError("No scans loaded.")
            return validate_xyz(pts), [0.0]

        # Reference is Scan 0
        merged = [validate_xyz(context.scans[0].points)]
        rmse_list = [0.0]
        
        current_ref = merged[0]

        for i in range(1, len(context.scans)):
            src_pts = validate_xyz(context.scans[i].points)
            
            # 1. Anchor translation (between current src and current_ref)
            t_offset = (self._anchor(current_ref, context.scans[0].intensity) - 
                        self._anchor(src_pts, context.scans[i].intensity))
            src_shifted = src_pts + t_offset
            
            # 2. ICP fine registration
            src_reg, rmse = self._icp(src_shifted, current_ref)
            
            merged.append(src_reg)
            rmse_list.append(rmse)
            # Update reference for the next station (Chain)
            current_ref = src_reg

        merged_cloud = np.vstack(merged)
        return validate_xyz(merged_cloud), rmse_list


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


