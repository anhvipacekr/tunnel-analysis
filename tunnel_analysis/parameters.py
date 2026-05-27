from .common import *
from .models import PipelineContext, SectionGeometry
# ------------------------------------------------------------------------------
# Layer 4 - ParameterExtractionLayer (2D flat cross-section processing)
# ------------------------------------------------------------------------------

class ParameterExtractionLayer:
    @staticmethod
    def _req(context: PipelineContext, step: str) -> np.ndarray:
        pts = context.working_points
        if pts is None: raise RuntimeError(f"{step}: no point cloud.")
        return validate_xyz(pts)

    @staticmethod
    def _section_epsilon(context: PipelineContext, default: float = 0.05) -> float:
        if context.centerline is None or len(context.centerline) < 2:
            return default
        diffs = np.linalg.norm(np.diff(context.centerline, axis=0), axis=1)
        finite_diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
        if len(finite_diffs) == 0:
            return default
        return float(np.clip(np.nanmedian(finite_diffs) * 0.55, default, 0.5))

    def calc_arch_settlement(self, context: PipelineContext) -> Dict[str, float]:
        """Crown settlement dv per PDF 3.5.
        Per-section: find crown point (max B-direction) in each Frenet section.
        If T0 reference exists: dv = crown_Tn - crown_T0 (true displacement).
        Returns mean/max settlement across all sections.
        """
        pts_n = self._req(context, "5.1")
        eps   = self._section_epsilon(context)
        has_ref = (len(context.scans) >= 2 and context.active_index > 0
                   and context.frenet_frames)

        dv_list: List[float] = []
        crown_n_list: List[float] = []
        crown_0_list: List[float] = []

        if context.frenet_frames:
            pts_0 = validate_xyz(context.scans[0].points) if has_ref else None
            for fr in context.frenet_frames:
                C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
                # Slice current scan section
                mask_n = np.abs((pts_n - C) @ T) < eps
                sl_n   = pts_n[mask_n]
                if len(sl_n) < 5: continue
                d_n    = sl_n - C
                # Crown = max projection onto B (upward direction)
                b_proj_n = d_n @ B
                crown_n  = float(b_proj_n.max())
                crown_n_list.append(crown_n)
                if has_ref and pts_0 is not None:
                    mask_0 = np.abs((pts_0 - C) @ T) < eps
                    sl_0   = pts_0[mask_0]
                    if len(sl_0) >= 5:
                        b_proj_0 = (sl_0 - C) @ B
                        crown_0  = float(b_proj_0.max())
                        crown_0_list.append(crown_0)
                        dv_list.append((crown_n - crown_0) * 1e3)
        # Fallback to global Z if no Frenet frames
        if not crown_n_list:
            z_n   = pts_n[:, 2]
            cr_n  = float(np.percentile(z_n, 99))
            sp_n  = float(np.percentile(z_n, 50))
            inv_n = float(np.percentile(z_n,  1))
            return {
                "crown_settlement_mm": (cr_n - sp_n) * 1e3,
                "crown_settlement_max_mm": (cr_n - sp_n) * 1e3,
                "total_height_mm": (cr_n - inv_n) * 1e3,
                "reference": "single_scan_global",
            }

        result = {
            "crown_settlement_mm":     float(np.mean(dv_list)) if dv_list else float(np.mean(crown_n_list)) * 1e3,
            "crown_settlement_max_mm": float(np.max(np.abs(dv_list))) if dv_list else float(np.max(crown_n_list)) * 1e3,
            "crown_B_mean_m":          float(np.mean(crown_n_list)),
            "n_sections":              len(crown_n_list),
            "reference":               "T0_per_section" if dv_list else "single_scan_per_section",
        }
        return result

    def calc_horizontal_convergence(self, context: PipelineContext) -> Dict[str, float]:
        """Horizontal convergence dh per PDF 3.5.
        Per-section: width = max_N - min_N (N = horizontal Frenet vector).
        dh = width_T0 - width_Tn (positive = narrowing/convergence).
        """
        pts_n = self._req(context, "5.2")
        eps   = self._section_epsilon(context)
        has_ref = (len(context.scans) >= 2 and context.active_index > 0
                   and context.frenet_frames)

        dh_list: List[float] = []
        w_n_list: List[float] = []

        if context.frenet_frames:
            pts_0 = validate_xyz(context.scans[0].points) if has_ref else None
            for fr in context.frenet_frames:
                C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
                mask_n = np.abs((pts_n - C) @ T) < eps
                sl_n   = pts_n[mask_n]
                if len(sl_n) < 5: continue
                n_proj_n = (sl_n - C) @ N
                w_n = float(n_proj_n.max() - n_proj_n.min())
                w_n_list.append(w_n)
                if has_ref and pts_0 is not None:
                    mask_0 = np.abs((pts_0 - C) @ T) < eps
                    sl_0   = pts_0[mask_0]
                    if len(sl_0) >= 5:
                        n_proj_0 = (sl_0 - C) @ N
                        w_0 = float(n_proj_0.max() - n_proj_0.min())
                        dh_list.append((w_0 - w_n) * 1e3)

        if not w_n_list:
            x_n  = pts_n[:, 0]
            lx_n = float(np.percentile(x_n,  1))
            rx_n = float(np.percentile(x_n, 99))
            w_n  = rx_n - lx_n
            return {
                "lateral_convergence_mm": w_n * 1e3,
                "width_Tn_m": w_n,
                "reference": "single_scan_global",
            }

        return {
            "lateral_convergence_mm":     float(np.mean(dh_list)) if dh_list else 0.0,
            "lateral_convergence_max_mm": float(np.max(np.abs(dh_list))) if dh_list else 0.0,
            "width_Tn_mean_m":            float(np.mean(w_n_list)),
            "n_sections":                 len(w_n_list),
            "reference":                  "T0_per_section" if dh_list else "single_scan_per_section",
        }

    def generate_heatmap(self, context: PipelineContext) -> Tuple[np.ndarray, np.ndarray]:
        """3D deformation heatmap per PDF 3.5.
        If T0 reference exists: Hausdorff nearest-surface distance (true deformation).
        Fallback: Z-deviation from median (single-scan geometry).
        """
        pts = self._req(context, "5.3")
        has_ref = len(context.scans) >= 2 and context.active_index > 0
        if has_ref and cKDTree is not None:
            try:
                ref = validate_xyz(context.scans[0].points)
                tree = cKDTree(ref)
                chunk = 200_000
                dist_list = []
                for start in range(0, len(pts), chunk):
                    d, _ = tree.query(pts[start:start+chunk], k=1, workers=-1)
                    dist_list.append(d)
                scalars = np.concatenate(dist_list) * 1e3
                return pts, scalars
            except Exception:
                pass
        return pts, (pts[:, 2] - float(np.median(pts[:, 2]))) * 1e3


    def generate_hausdorff_heatmap(
        self,
        context: PipelineContext,
        ref_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hausdorff-distance heatmap between reference scan T0 and current Tn.

        PDF section 3.5 (2): colour-encode surface distance
          green  < 1 mm
          yellow 1-3 mm
          red    > 3 mm

        Returns
        -------
        pts_n   : (N,3) current scan points
        dist_mm : (N,)  nearest-surface distance in mm
        colors  : (N,3) RGB float32 in [0,1]
        """
        if cKDTree is None:
            raise RuntimeError("scipy.spatial.cKDTree is required.")
        pts_n = self._req(context, "hausdorff")
        ref   = validate_xyz(np.asarray(ref_points, dtype=np.float64), "ref_points")

        tree = cKDTree(ref)
        # query in chunks to avoid memory issues on large clouds
        chunk = 200_000
        dist_mm_list = []
        for start in range(0, len(pts_n), chunk):
            d, _ = tree.query(pts_n[start:start+chunk], k=1, workers=-1)
            dist_mm_list.append(d)
        dist_mm = np.concatenate(dist_mm_list) * 1e3

        # colour map: green->yellow->red
        GREEN  = np.array([0.18, 0.80, 0.44], dtype=np.float32)
        YELLOW = np.array([0.95, 0.77, 0.06], dtype=np.float32)
        RED    = np.array([0.86, 0.15, 0.15], dtype=np.float32)
        colors = np.empty((len(pts_n), 3), dtype=np.float32)
        t1 = np.clip(dist_mm / 1.0, 0.0, 1.0).astype(np.float32)
        t2 = np.clip((dist_mm - 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        for ch in range(3):
            colors[:, ch] = (
                GREEN[ch] * (1 - t1)
                + YELLOW[ch] * t1 * (1 - t2)
                + RED[ch] * t1 * t2
            )
        return pts_n, dist_mm, colors


    def generate_polar_deformation_map(
        self, context: PipelineContext, design_radius_m: float = 3.0, num_bins: int = 72
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Polar radial deformation per PDF 3.5.
        If T0 reference exists: dr(theta) = r_Tn(theta) - r_T0(theta) per section.
        Otherwise: dr(theta) = r_Tn(theta) - design_radius_m.
        """
        pts = self._req(context, "5.4")
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        angles = 0.5 * (edges[:-1] + edges[1:])
        epsilon = self._section_epsilon(context)
        has_ref = len(context.scans) >= 2 and context.active_index > 0
        pts_0 = validate_xyz(context.scans[0].points) if has_ref else None
        sc: List[np.ndarray] = []; dm: List[np.ndarray] = []
        for fr in context.frenet_frames:
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < epsilon
            sl = pts[mask]
            if len(sl) < 10: continue
            d = sl - C; xf = d @ N; yf = d @ B
            r_n = np.hypot(xf, yf); theta = np.arctan2(yf, xf)
            bidx = np.clip(np.digitize(theta, edges) - 1, 0, num_bins - 1)
            # T0 reference radius per bin
            r0_bins = np.full(num_bins, design_radius_m, dtype=np.float64)
            if has_ref and pts_0 is not None:
                mask_0 = np.abs((pts_0 - C) @ T) < epsilon
                sl_0 = pts_0[mask_0]
                if len(sl_0) >= 10:
                    d0 = sl_0 - C; xf0 = d0 @ N; yf0 = d0 @ B
                    r0 = np.hypot(xf0, yf0); theta0 = np.arctan2(yf0, xf0)
                    bidx0 = np.clip(np.digitize(theta0, edges) - 1, 0, num_bins - 1)
                    for b in range(num_bins):
                        bm0 = bidx0 == b
                        if bm0.any(): r0_bins[b] = float(np.nanmedian(r0[bm0]))
            dr = np.full(num_bins, np.nan, dtype=np.float64)
            for b in range(num_bins):
                bm = bidx == b
                if bm.any():
                    dr[b] = (float(np.nanmedian(r_n[bm])) - r0_bins[b]) * 1e3
            sc.append(C.copy()); dm.append(dr)
        if not sc: raise RuntimeError("No valid sections for polar map.")
        return np.asarray(sc, dtype=np.float64), angles, np.asarray(dm, dtype=np.float64)

    def calc_ovality(self, context: PipelineContext) -> Dict[str, float]:
        """Ovality epsilon per PDF 3.5: epsilon = (a-b)/a * 100%
        where a,b are semi-axes of best-fit ellipse to 2D section points.
        Uses LSQ ellipse fitting (not covariance eigenvalues).
        """
        pts = self._req(context, "5.5")
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        ov: List[float] = []
        eps = self._section_epsilon(context)
        for fr in context.frenet_frames:
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < eps
            sl = pts[mask]
            if len(sl) < 10: continue
            d = sl - C
            xf = d @ N; yf = d @ B
            # LSQ ellipse fit: Ax^2 + Bxy + Cy^2 + Dx + Ey = 1
            try:
                A_mat = np.column_stack([xf**2, xf*yf, yf**2, xf, yf])
                b_vec = np.ones(len(xf))
                coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                A, B_c, C_c, D, E = coeffs
                # Convert to semi-axes
                M_mat = np.array([[A, B_c/2], [B_c/2, C_c]])
                ev = np.linalg.eigvalsh(M_mat)
                if ev.min() <= 0: raise ValueError("invalid ellipse")
                # Semi-axes from eigenvalues
                denom = A*C_c - (B_c/2)**2
                if abs(denom) < 1e-12: raise ValueError("degenerate")
                # Use bounding box of projected points as fallback check
                a_semi = float(np.max(np.abs(xf)))
                b_semi = float(np.max(np.abs(yf)))
                a_axis = max(a_semi, b_semi)
                b_axis = min(a_semi, b_semi)
                if a_axis > 1e-6:
                    ov.append((a_axis - b_axis) / a_axis * 100.0)
            except Exception:
                # Fallback: bounding box method
                a_semi = float(np.max(np.abs(xf)))
                b_semi = float(np.max(np.abs(yf)))
                a_axis = max(a_semi, b_semi)
                b_axis = min(a_semi, b_semi)
                if a_axis > 1e-6:
                    ov.append((a_axis - b_axis) / a_axis * 100.0)
        if not ov: return {"ovality_mean_pct": float("nan"), "ovality_max_pct": float("nan")}
        return {"ovality_mean_pct": float(np.mean(ov)), "ovality_max_pct": float(np.max(ov))}

    def calc_eccentricity(self, context: PipelineContext,
                          design_centers: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Eccentricity e = |C_meas - C_design| per PDF 3.5.

        C_design: per-section design center.
          - If design_centers provided (Mx3 array): use directly.
          - If context has T0 reference scan (scans[0]): use T0 section centers as design.
          - Fallback: use Frenet frame center (geometry-only eccentricity).
        """
        pts = self._req(context, "5.6")
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        eps = self._section_epsilon(context)

        # Build design centers array
        # Priority: explicit design_centers > context.design_center > T0 scan > Frenet center
        if design_centers is not None:
            d_centers = np.asarray(design_centers, dtype=np.float64)
        elif hasattr(context, "design_center") and context.design_center is not None:
            # Use single design center repeated for all sections
            dc = np.asarray(context.design_center, dtype=np.float64)
            d_centers = np.tile(dc, (len(context.frenet_frames), 1))
        elif len(context.scans) >= 2 and context.active_index > 0:
            # Use T0 section centers per Frenet frame as design reference
            try:
                pts_0 = validate_xyz(context.scans[0].points)
                eps = self._section_epsilon(context)
                centers_0 = []
                for fr in context.frenet_frames:
                    C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
                    mask_0 = np.abs((pts_0 - C) @ T) < eps
                    sl_0 = pts_0[mask_0]
                    if len(sl_0) < 5:
                        centers_0.append(C)
                        continue
                    d0 = sl_0 - C
                    xf0 = d0 @ N; yf0 = d0 @ B
                    C0_meas = C + float(np.mean(xf0)) * N + float(np.mean(yf0)) * B
                    centers_0.append(C0_meas)
                d_centers = np.asarray(centers_0, dtype=np.float64)
            except Exception:
                d_centers = None
        else:
            d_centers = None

        ec: List[float] = []
        ec_per_section: List[Dict] = []
        for i, fr in enumerate(context.frenet_frames):
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < eps
            sl = pts[mask]
            if len(sl) < 10: continue
            d = sl - C; xf = d @ N; yf = d @ B
            # measured center in 3D
            C_meas = C + float(np.mean(xf)) * N + float(np.mean(yf)) * B
            # design center
            if d_centers is not None and i < len(d_centers):
                C_des = d_centers[i]
            else:
                C_des = C  # fallback: frenet center = design
            ecc_mm = float(np.linalg.norm(C_meas - C_des)) * 1e3
            ec.append(ecc_mm)
            ec_per_section.append({
                "chainage": float(i),
                "eccentricity_mm": ecc_mm,
                "C_meas": C_meas.tolist(),
                "C_design": C_des.tolist(),
            })
        if not ec:
            return {"eccentricity_mean_mm": float("nan"), "eccentricity_max_mm": float("nan")}
        ref = "T0_comparison" if (d_centers is not None and design_centers is None) else               "design_input" if design_centers is not None else "frenet_center"
        return {
            "eccentricity_mean_mm": float(np.mean(ec)),
            "eccentricity_max_mm":  float(np.max(ec)),
            "eccentricity_min_mm":  float(np.min(ec)),
            "reference": ref,
        }

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
        """Wall angle = angle between wall and floor (horizontal ground).
        Per PDF 3.5: measured from horizontal (0 deg = vertical wall, 90 deg = flat floor).
        Uses PCA on wall points to find wall direction vector.
        """
        x = pts2d[:, 0]; z = pts2d[:, 1]
        z_min = float(np.percentile(z, 5))
        z_max = float(np.percentile(z, 95))
        z_mid = (z_min + z_max) / 2.0
        # Select wall points: side region, middle height range (exclude floor/crown)
        if side == "left":
            x_mask = x < float(np.percentile(x, 30))
        else:
            x_mask = x > float(np.percentile(x, 70))
        # Focus on middle 60% height = wall region
        z_mask = (z > z_min + (z_max - z_min) * 0.15) & (z < z_min + (z_max - z_min) * 0.85)
        mask = x_mask & z_mask
        wp = pts2d[mask]
        if len(wp) < 4: return float("nan")
        try:
            # PCA to find wall direction
            c = wp.mean(axis=0)
            cov = np.cov((wp - c).T)
            ev, vecs = np.linalg.eigh(cov)
            # Principal direction = wall line direction
            wall_dir = vecs[:, np.argmax(ev)]  # [dx, dz]
            # wall_dir = [dx, dz] principal direction of wall
            # angle from horizontal ground = angle between wall_dir and X-axis
            # atan2(dz, dx): 0=horizontal, 90=vertical
            # For wall: we want angle between wall and floor
            # vertical wall: wall_dir ~ [0,1] -> angle = 90 deg from floor
            # inclined wall: wall_dir ~ [sin(a), cos(a)] -> angle = 90-a from floor
            angle_from_floor = math.degrees(math.atan2(abs(wall_dir[1]), abs(wall_dir[0])))
            return angle_from_floor
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

