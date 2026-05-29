from __future__ import annotations
from ..common import *
from ..models import PointCloudBundle, PipelineContext, SectionGeometry
from ..worker import PipelineWorker
from ..target_detector import Target
from .widgets import CollapsibleSection, MatplotlibSectionWidget, PolarDeformationPlotWidget, LinePlotWidget
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .main_window import TunnelAnalysisWindow

class DispatchMixin:
    _TAB_MAP = {
        "1.1_import": 1, "1.2_viewport": 0, "1.3_add": 2, "1.4_merge": 0,
        "1.5_rough": 0, "1.6_chain": 0, "1.7_reg_error": 0,
        "2.1_voxel": 0, "2.2_sor": 0, "2.3_lining": 0, "2.4_semantic": 0,
        "3.1_anchor": 0, "3.2_icp": 0, "3.3_rmse": 0,
        "4.1_centerline": 0, "4.2_iterative": 0, "4.3_bspline": 0,
        "4.3b_bspline": 0, "4.4_frenet": 0, "4.5_seams": 0,
        "4.5b_intensity_seams": 0,
        "5.1_settlement": 0, "5.2_convergence": 0, "5.3_heatmap": 0,
        "5.3b_hausdorff": 0, "5.4_polar": 6, "5.5_ovality": 0,
        "5.6_eccentricity": 0, "5.7_sections": 5, "5.8_clearance_3d": 0,
        "6.1_epochs": 1, "6.2_plot": 4,
        "7.1_ifc": 7, "7.2_ai": 7,
        "8.1_csv": 0, "8.2_excel": 0, "8.3_pdf": 0, "8.4_web": 0,
    }

    def _auto_switch_tab(self, key: str) -> None:
        idx = self._TAB_MAP.get(key)
        if idx is not None:
            self.right_tabs.setCurrentIndex(idx)

    @QtCore.Slot(str, object)
    def _on_finished(self, key: str, result: object) -> None:
        self.sb_prog.setValue(100); self.sb_msg.setText(f"Task completed: {key}"); self._auto_switch_tab(key); self._dispatch(key, result); self._check_auto_pipeline(key)

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
            self._refresh_station_list()
            self._render_station_markers()

        elif key == "1.3_add_scan":
            b: PointCloudBundle = result
            self.context.scans.append(b)
            self._log(f"Station {len(self.context.scans)} loaded: {b.path} ({len(b.points):,} pts)")
            self._update_meta(b)
            self._refresh_station_list()
            self._render_station_markers()
        elif key == "target_detect":
            new_targets: List[Target] = result
            self._targets.extend(new_targets)
            self._refresh_target_table()
            self._render_target_markers()
            n_sph = sum(1 for t in new_targets if t.type == "sphere")
            n_flt = sum(1 for t in new_targets if t.type == "flat")
            n_chk = sum(1 for t in new_targets if t.type == "checkerboard")
            n_int = sum(1 for t in new_targets if t.type == "intensity")
            n_man = sum(1 for t in new_targets if t.type == "manual")
            # Switch to Targets tab
            for i in range(self.right_tabs.count()):
                if self.right_tabs.tabText(i) == "Targets":
                    self.right_tabs.setCurrentIndex(i); break
            # Show result dialog
            if len(new_targets) == 0:
                QtWidgets.QMessageBox.warning(self, "Target Detection",
                    "No targets found." + chr(10) + chr(10) +
                    "Try adjusting parameters:" + chr(10) +
                    "- Lower intensity percentile (e.g. 90%)" + chr(10) +
                    "- Lower min cluster points (e.g. 10)" + chr(10) +
                    "- Lower min contrast ratio (e.g. 1.2)" + chr(10) +
                    "- Check if file has intensity/color data")
                self._log("Target detection: 0 targets found.")
            else:
                lines = [f"Found {len(new_targets)} target(s):"]
                if n_sph: lines.append(f"  Sphere:       {n_sph}")
                if n_chk: lines.append(f"  Checkerboard: {n_chk}")
                if n_int: lines.append(f"  Intensity:    {n_int}")
                if n_man: lines.append(f"  Manual:       {n_man}")
                lines.append("")
                lines.append("Targets are shown in the table and marked on 3D viewport.")
                QtWidgets.QMessageBox.information(self, "Target Detection Complete",
                    chr(10).join(lines))
                self._log(f"Target detection: {len(new_targets)} found "
                          f"(sphere={n_sph}, checkerboard={n_chk}, intensity={n_int})")
                for t in new_targets:
                    self._log(f"  [{t.type}] {t.name} conf={t.confidence:.2f} n={t.n_points}")
        elif key == "target_register":
            T, rmse, residuals, reg_pts = result
            if reg_pts is not None:
                self.context.registered_points = reg_pts
                self._render_pts(reg_pts, "Target-based Registration", "#10B981")
                self.pt_label.setText(f"Points: {len(reg_pts):,}")
            self._log(f"Target-based registration: RMSE = {rmse:.3f} mm")
            for sid, tid, res in residuals:
                status = "OK" if res < 2.0 else "CAUTION" if res < 5.0 else "POOR"
                self._log(f"  {sid} <-> {tid}: {res:.3f} mm [{status}]")
        elif key == "1.6_chain":
            pts, rmse_list = result
            self.context.registered_points = pts
            self._render_pts(pts, f"Chain Registered: {len(self.context.scans)} stations", "#10B981")
            self.sb_pts.setText(f"Points: {len(pts):,}")
            self.pt_label.setText(f"Points: {len(pts):,}")
            self._log(f"Chain registration complete: {len(pts):,} total points")
            for i, rmse in enumerate(rmse_list):
                status = "OK" if rmse < 2.0 else "CAUTION" if rmse < 5.0 else "POOR"
                self._log(f"  Station {i+1}: RMSE = {rmse:.3f} mm [{status}]")
        elif key == "1.7_reg_error":
            pts, dist_mm, colors = result
            mesh = make_vertex_cloud(pts)
            if len(dist_mm) == mesh.n_points:
                mesh["RegError_mm"] = dist_mm
            if self.plotter is not None:
                self.plotter.clear(); self.plotter.set_background("#F8FAFC")
                self.plotter.add_mesh(mesh, scalars="RegError_mm", cmap="RdYlGn_r",
                    style="points", point_size=2.8, render_points_as_spheres=False,
                    reset_camera=True, clim=[0, 5],
                    scalar_bar_args={"title": "Reg Error (mm)"})
                self.plotter.add_text("Registration Error Heatmap",
                    position="upper_left", font_size=11, color="#111827", name="ttl")
                self.plotter.add_axes(color="#111827")
                self.plotter.reset_camera(); self.plotter.render()
            self._log(f"Registration error: median={float(__import__('numpy').median(dist_mm)):.2f}mm max={float(__import__('numpy').max(dist_mm)):.2f}mm")
        elif key == "1.4_merge":
            pts, rmse_list = result
            self.context.registered_points = pts
            self._render_pts(pts, f"Merged {len(self.context.scans)} scan stations", "#10B981")
            self.sb_pts.setText(f"Points: {len(pts):,}")
            self.pt_label.setText(f"Points: {len(pts):,}")
            self._log(f"Merged {len(self.context.scans)} stations: {len(pts):,} total points")
            for i, rmse in enumerate(rmse_list):
                self._log(f"  Station {i+1}: RMSE = {rmse:.3f} mm")
        elif key == "2.1_voxel":
            pts, centroid = result; self.context.normalized_points = pts
            raw_n = len(self.context.active_scan.points) if self.context.active_scan else len(pts)
            self._render_pts(pts, "2.1 Voxel Grid Filter", "#3B82F6")
            self.pt_label.setText(f"Points: {len(pts):,}"); self.sb_pts.setText(f"Points: {len(pts):,}")
            self._log(f"Voxel downsampling complete: {len(pts):,}/{raw_n:,} points retained; centroid shifted to local origin {np.round(centroid, 3).tolist()}.")

        elif key == "2.2_sor":
            if isinstance(result, tuple) and len(result) == 3:
                pts, col, stats = result
            else:
                pts, col = result; stats = {"n_raw": len(pts), "n_clean": len(pts), "n_removed": 0, "outlier_pts": np.empty((0, 3))}
            self._kept_pts  = np.asarray(pts, dtype=np.float64)
            self._noise_pts = np.asarray(stats.get("outlier_pts", np.empty((0, 3))), dtype=np.float64)
            if self.context.active_scan and col is not None: self.context.active_scan.colors_raw = col
            self._render_filter_result(self._kept_pts, self._noise_pts, "2.2 SOR — Review noise (red) before confirming")
            self.pt_label.setText(f"Points: {len(pts):,}"); self.sb_pts.setText(f"Points: {len(pts):,}")
            n_raw = stats.get('n_raw', len(pts)); n_rem = stats.get('n_removed', 0)
            self._log(f"SOR proposal: {n_raw:,} raw -> {len(pts):,} kept, {n_rem:,} noise detected (red).")
            self._log("Review noise in 3D viewport, then use the noise panel to confirm or adjust.")
            self.sb_msg.setText(f"SOR: {n_rem:,} noise points detected (red) | {len(pts):,} kept (blue)")
            self._show_noise_panel()

        elif key == "2.4_semantic":
            pts, stats = result
            self.context.normalized_points = pts
            noise_pts = np.asarray(stats.get("noise_pts", np.empty((0,3))), dtype=np.float64)
            self._render_filter_result(np.asarray(pts, dtype=np.float64), noise_pts,
                "2.4 Semantic Noise Removal | kept=blue, removed=red")
            self.pt_label.setText(f"Points: {len(pts):,}")
            self.sb_pts.setText(f"Points: {len(pts):,}")
            self._log(f"Semantic removal: {stats.get('n_clean',len(pts)):,}/{stats.get('n_raw',len(pts)):,} kept")
            self._log(f"  Cable={stats.get('n_cable',0)} Light={stats.get('n_light',0)} Person={stats.get('n_person',0)}")
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

        elif key == "4.3b_bspline":
            cl, fr = result
            self.context.centerline = cl; self.context.frenet_frames = fr
            self._render_cl(cl, fr)
            self._log(f"B-Spline C2 centerline (PDF 3.4): {len(cl)} points, {len(fr)} Frenet frames.")
        elif key == "4.3_bspline":
            sm = np.asarray(result, dtype=np.float64); self.context.centerline_smooth = sm
            if self.plotter:
                self.plotter.add_lines(sm, color="#F59E0B", width=4, connected=True, name="cl_sm")
                self.plotter.render()
            self._log(f"B-Spline centerline smoothing complete: {len(sm)} points.")

        elif key == "4.4_frenet":
            self.context.frenet_frames = result; self._log(f"Gravity-aligned section frames generated successfully: {len(result)} N-B frames.")

        elif key == "4.5b_intensity_seams":
            d = result
            n = len(d.get("chainage_m", []))
            self._log(f"Intensity ring seam detection (PDF 3.3): {n} seams detected.")
            if n:
                self._log(f"  Seam chainages (m): {[round(float(x),2) for x in d['chainage_m']]}")
        elif key == "4.5_seams":
            d: Dict = result; self._log(f"Ring seam detection complete: {d['ring_count']} lining rings segmented, {d['total_seams']} seam boundaries identified.")

        elif key in ("5.1_settlement", "5.2_convergence", "5.5_ovality", "5.6_eccentricity"):
            self.context.parameters.update(result); self._show_params(result)

        elif key == "5.3b_hausdorff":
            pts, dist_mm, colors = result
            self.context.heatmap_scalars = dist_mm
            mesh = __import__("tunnel_analysis.common", fromlist=["make_vertex_cloud"]).make_vertex_cloud(pts)
            if len(dist_mm) == mesh.n_points:
                mesh["Hausdorff_mm"] = dist_mm
            if self.plotter is not None:
                self.plotter.clear(); self.plotter.set_background("#F8FAFC")
                self.plotter.add_mesh(mesh, scalars="Hausdorff_mm", cmap="RdYlGn_r",
                    style="points", point_size=2.8, render_points_as_spheres=False,
                    reset_camera=True, scalar_bar_args={"title": "Distance T0→Tn (mm)"})
                self.plotter.add_text("Hausdorff Heatmap T0→Tn", position="upper_left",
                    font_size=11, color="#111827", name="ttl")
                self.plotter.add_axes(color="#111827"); self.plotter.reset_camera(); self.plotter.render()
            self._log(f"Hausdorff heatmap: median={float(__import__('numpy').median(dist_mm)):.2f}mm max={float(__import__('numpy').max(dist_mm)):.2f}mm")
            self.right_tabs.setCurrentIndex(0)
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

        elif key == "5.8_clearance_3d":
            pts, colors, n_viol = result
            mesh = make_vertex_cloud(pts)
            if self.plotter is not None:
                self.plotter.clear(); self.plotter.set_background("#F8FAFC")
                self.plotter.add_mesh(mesh, scalars=None, style="points",
                    point_size=2.5, render_points_as_spheres=False,
                    reset_camera=True, color="#94A3B8")
                # Highlight violation points in red
                if len(colors):
                    viol_pts = pts[colors]
                    if len(viol_pts):
                        viol_mesh = make_vertex_cloud(viol_pts)
                        self.plotter.add_mesh(viol_mesh, color="#DC2626",
                            style="points", point_size=6.0,
                            render_points_as_spheres=True,
                            reset_camera=False, name="clearance_viol")
                self.plotter.add_text(
                    f"Clearance Violations: {n_viol} points",
                    position="upper_left", font_size=11,
                    color="#DC2626" if n_viol > 0 else "#047857",
                    name="ttl")
                self.plotter.add_axes(color="#111827")
                self.plotter.reset_camera(); self.plotter.render()
            self._log(f"Clearance 3D map: {n_viol} violation points detected")
            if n_viol > 0:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Clearance Violation",
                    f"{n_viol} points violate vehicle clearance envelope!" + chr(10) +
                    "Red points shown on 3D viewport.")
        elif key == "5.7_sections":
            sections: List[SectionGeometry] = result; self.context.sections = sections
            self.section_widget.set_sections(sections, profile=self.context.tunnel_profile, vl_box_w=self._sp_vl_w.value(), vl_box_h=self._sp_vl_h.value(), vl_cir_r=self._sp_vl_r.value())
            try: self.section_widget.section_changed.disconnect()
            except Exception: pass
            self.section_widget.section_changed.connect(self._highlight_section)
            self._highlight_section(0)
            # Set T0 reference sections if available
            if len(self.context.scans) >= 2 and hasattr(self.section_widget, "set_ref_sections"):
                try:
                    from ..models import PipelineContext as _PC
                    ctx0 = _PC(scans=[self.context.scans[0]], active_index=0,
                               normalized_points=self.context.scans[0].points,
                               centerline=self.context.centerline,
                               frenet_frames=self.context.frenet_frames,
                               tunnel_profile=self.context.tunnel_profile)
                    ref_secs = self.par_mod.compute_all_sections(ctx0,
                        vl_box_w=self._sp_vl_w.value(),
                        vl_box_h=self._sp_vl_h.value(),
                        vl_cir_r=self._sp_vl_r.value())
                    self.section_widget.set_ref_sections(ref_secs)
                    self._log("T0 reference sections loaded for overlay.")
                except Exception as e:
                    self._log(f"T0 overlay: {e}")
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
            self.ts_plot.set_values(series, "Cloud-to-Cloud Displacement T0->Tn (mm)")
            self.right_tabs.setCurrentIndex(2)
            # Log C2C stats
            try:
                _, _, stats = self.ts_mod.compute_cloud_to_cloud(self.context)
                self._log("--- Cloud-to-Cloud Displacement T0 -> Tn (PDF 3.5) ---")
                self._log(f"  ICP alignment RMSE    : {stats['icp_rmse_mm']:.3f} mm")
                self._log(f"  Mean displacement     : {stats['c2c_mean_mm']:.3f} mm")
                self._log(f"  Median displacement   : {stats['c2c_median_mm']:.3f} mm")
                self._log(f"  Max displacement      : {stats['c2c_max_mm']:.3f} mm")
                self._log(f"  95th percentile       : {stats['c2c_p95_mm']:.3f} mm")
                self._log(f"  Points T0 / Tn        : {stats['n_points_t0']:,} / {stats['n_points_tn']:,}")
                self._log("------------------------------------------------")
            except Exception as ex:
                self._log(f"  [Stats error] {ex}")

        elif key == "8.1_csv":
            path = result
            self._log(f"CSV exported: {path}")
            import subprocess; subprocess.Popen(["explorer", "/select,", path])
        elif key == "8.4_web":
            url = result
            self._log(f"Web dashboard launched: {url}")
        elif key == "8.3_pdf":
            path = result
            self._log(f"PDF report exported: {path}")
            import subprocess; subprocess.Popen(["explorer", "/select,", path])
        elif key == "8.2_excel":
            path = result
            self._log(f"Excel report exported: {path}")
            import subprocess; subprocess.Popen(["explorer", "/select,", path])
        elif key == "7.1_ifc":
            self.ai_resp.setPlainText(json.dumps(result, indent=2)); self.right_tabs.setCurrentIndex(self._ai_tab_idx)

        elif key == "7.2_ai":
            self.ai_resp.setPlainText(str(result)); self.right_tabs.setCurrentIndex(self._ai_tab_idx)

