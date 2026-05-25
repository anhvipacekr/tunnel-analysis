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
        If T0 reference scan exists (scans[0] when active_index > 0),
        compute delta Z_max = crown_Tn - crown_T0 (true displacement).
        Otherwise fall back to single-scan geometry.
        """
        pts_n = self._req(context, "5.1")
        z_n   = pts_n[:, 2]
        cr_n  = float(np.percentile(z_n, 99))
        sp_n  = float(np.percentile(z_n, 50))
        inv_n = float(np.percentile(z_n,  1))

        # T0 reference
        has_ref = len(context.scans) >= 2 and context.active_index > 0
        if has_ref:
            try:
                z_0  = validate_xyz(context.scans[0].points)[:, 2]
                cr_0 = float(np.percentile(z_0, 99))
                sp_0 = float(np.percentile(z_0, 50))
                dv   = (cr_n - cr_0) * 1e3   # positive = heave, negative = settlement
                return {
                    "crown_settlement_mm":   dv,
                    "crown_z_Tn_m":          cr_n,
                    "crown_z_T0_m":          cr_0,
                    "springline_z_m":        sp_n,
                    "invert_z_m":            inv_n,
                    "total_height_mm":       (cr_n - inv_n) * 1e3,
                    "reference": "T0_comparison",
                }
            except Exception:
                pass
        # single-scan fallback
        return {
            "crown_settlement_mm": (cr_n - sp_n) * 1e3,
            "total_height_mm":     (cr_n - inv_n) * 1e3,
            "crown_z_Tn_m":        cr_n,
            "springline_z_m":      sp_n,
            "invert_z_m":          inv_n,
            "reference": "single_scan",
        }

    def calc_horizontal_convergence(self, context: PipelineContext) -> Dict[str, float]:
        """Horizontal convergence dh per PDF 3.5.
        If T0 reference exists: dh = (width_T0 - width_Tn) = convergence (positive = narrowing).
        Otherwise: single-scan width.
        """
        pts_n = self._req(context, "5.2")
        x_n   = pts_n[:, 0]
        lx_n  = float(np.percentile(x_n,  1))
        rx_n  = float(np.percentile(x_n, 99))
        mx_n  = float(np.mean(x_n))
        w_n   = rx_n - lx_n

        has_ref = len(context.scans) >= 2 and context.active_index > 0
        if has_ref:
            try:
                x_0  = validate_xyz(context.scans[0].points)[:, 0]
                lx_0 = float(np.percentile(x_0,  1))
                rx_0 = float(np.percentile(x_0, 99))
                w_0  = rx_0 - lx_0
                dh   = (w_0 - w_n) * 1e3   # positive = convergence (narrowing)
                return {
                    "lateral_convergence_mm":    dh,
                    "width_Tn_m":                w_n,
                    "width_T0_m":                w_0,
                    "left_wall_x_m":             lx_n,
                    "right_wall_x_m":            rx_n,
                    "lateral_centre_offset_mm":  (mx_n - (rx_n + lx_n) / 2) * 1e3,
                    "reference": "T0_comparison",
                }
            except Exception:
                pass
        return {
            "lateral_convergence_mm":   w_n * 1e3,
            "width_Tn_m":               w_n,
            "left_wall_x_m":            lx_n,
            "right_wall_x_m":           rx_n,
            "lateral_centre_offset_mm": (mx_n - (rx_n + lx_n) / 2) * 1e3,
            "reference": "single_scan",
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
        pts = self._req(context, "5.4")
        if not context.frenet_frames: raise RuntimeError("Run centerline first.")
        edges = np.linspace(-np.pi, np.pi, num_bins + 1); angles = 0.5 * (edges[:-1] + edges[1:])
        epsilon = self._section_epsilon(context)
        sc: List[np.ndarray] = []; dm: List[np.ndarray] = []
        for fr in context.frenet_frames:
            C, T, N, B = fr["center"], fr["T"], fr["N"], fr["B"]
            mask = np.abs((pts - C) @ T) < epsilon; sl = pts[mask]
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
            mask = np.abs((pts - C) @ T) < self._section_epsilon(context); sl = pts[mask]
            if len(sl) < 10: continue
            d = sl - C; xf = d @ N; yf = d @ B
            M = np.array([[float(np.mean(xf ** 2)), float(np.mean(xf * yf))],
                          [float(np.mean(xf * yf)), float(np.mean(yf ** 2))]])
            ev = np.linalg.eigvalsh(M)
            a = float(np.sqrt(max(ev.max(), 1e-9))); b = float(np.sqrt(max(ev.min(), 1e-9)))
            if a > 1e-6: ov.append((a - b) / a * 100.0)
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
        if design_centers is not None:
            d_centers = np.asarray(design_centers, dtype=np.float64)
        elif len(context.scans) >= 2 and context.active_index > 0:
            # Use T0 section centers as design reference
            try:
                pts_0 = validate_xyz(context.scans[0].points)
                c0 = pts_0.mean(axis=0)
                ev0, vecs0 = np.linalg.eigh(np.cov((pts_0 - c0).T))
                ax0 = vecs0[:, np.argmax(ev0)]
                proj0 = (pts_0 - c0) @ ax0
                n_fr = len(context.frenet_frames)
                chunks = np.array_split(pts_0[np.argsort(proj0)], n_fr)
                d_centers = np.asarray([ch.mean(axis=0) for ch in chunks if len(ch) >= 3],
                                        dtype=np.float64)
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

