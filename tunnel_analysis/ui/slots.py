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

class SlotsMixin:
    def _slot_1_1_import(self) -> None:
        self._hdr("LiDAR Data Acquisition", "Load LAS/LAZ/PLY point-cloud data into the project database.")
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load tunnel point-cloud data", "", "Point Clouds (*.las *.laz *.ply);;All Files (*.*)")
        if not fp: return
        max_pts = self._ask_max_points(fp)
        if max_pts is None: return
        self._start_worker("1.1_import", lambda: self.base_mod.load_scan(fp, max_points=max_pts))

    def _ask_max_points(self, fp: str):
        """Check file size and ask user for subsampling if needed."""
        total = self.base_mod.get_point_count(fp)
        from tunnel_analysis.io_layer import MAX_POINTS_DEFAULT
        if total <= 0 or total <= MAX_POINTS_DEFAULT:
            return MAX_POINTS_DEFAULT
        import pathlib
        fname = pathlib.Path(fp).name
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Large File - Loading Options")
        dlg.setMinimumWidth(440)
        lay = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel(
            "Large file: " + fname + chr(10) +
            "Total points: " + str(total) + chr(10) + chr(10) +
            "Loading all points may cause memory issues." + chr(10) +
            "Choose loading option:")
        lbl.setStyleSheet("font-size:10pt;color:#0F172A;")
        lay.addWidget(lbl)
        grp = QtWidgets.QButtonGroup(dlg)
        opts = [
            (f"5M points (recommended, fast)", 5_000_000),
            (f"10M points (more detail)", 10_000_000),
            (f"20M points (needs 2GB+ RAM)", 20_000_000),
            (f"ALL {total:,} points (may crash)", total),
        ]
        radios = []
        for label, val in opts:
            rb = QtWidgets.QRadioButton(label)
            grp.addButton(rb); lay.addWidget(rb)
            radios.append((rb, val))
        radios[0][0].setChecked(True)
        custom_lay = QtWidgets.QHBoxLayout()
        rb_custom = QtWidgets.QRadioButton("Custom:")
        grp.addButton(rb_custom)
        spin = QtWidgets.QSpinBox()
        spin.setRange(100_000, total); spin.setValue(5_000_000)
        spin.setSingleStep(1_000_000); spin.setSuffix(" pts")
        custom_lay.addWidget(rb_custom); custom_lay.addWidget(spin, 1)
        lay.addLayout(custom_lay)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        for rb, val in radios:
            if rb.isChecked(): return val
        return spin.value()

    def _slot_1_5_rough(self) -> None:
        """Open rough alignment dialog for manual pre-alignment."""
        self._hdr("Rough Alignment", "Manually adjust position/rotation before ICP.")
        if len(self.context.scans) < 2:
            self._log("Load at least 2 scan stations first."); return
        dlg = _RoughAlignDialog(self.context, self.reg_mod, self, self.plotter)
        dlg.exec()
        if dlg.result() == QtWidgets.QDialog.Accepted:
            self.context.normalized_points = dlg.aligned_pts
            self._render_pts(dlg.aligned_pts, "Rough Alignment Applied", "#F59E0B")
            self._log(f"Rough alignment applied: offset={dlg.offset} rot={dlg.rotation}")

    def _slot_1_6_chain(self) -> None:
        self._hdr("Chain Register & Merge",
                  "Sequential chain registration S1->S2->S3 (reduces drift).")
        if len(self.context.scans) < 2:
            self._log("Load at least 2 scan stations first."); return
        self._log(f"Chain registering {len(self.context.scans)} stations...")
        self._start_worker("1.6_chain",
            lambda: self.reg_mod.register_and_merge_chain(self.context))

    def _slot_1_7_reg_error(self) -> None:
        self._hdr("Registration Error Heatmap",
                  "Visualize registration error between merged cloud and reference.")
        if self.context.registered_points is None or len(self.context.scans) < 2:
            self._log("Run registration first."); return
        def _task():
            import numpy as _np
            from scipy.spatial import cKDTree as _kd
            pts = self.context.registered_points
            ref = self.context.scans[0].points
            tree = _kd(ref)
            d, _ = tree.query(pts, k=1, workers=-1)
            dist_mm = d * 1e3
            GREEN  = _np.array([0.18, 0.80, 0.44], dtype=_np.float32)
            YELLOW = _np.array([0.95, 0.77, 0.06], dtype=_np.float32)
            RED    = _np.array([0.86, 0.15, 0.15], dtype=_np.float32)
            colors = _np.empty((len(pts), 3), dtype=_np.float32)
            t1 = _np.clip(dist_mm / 2.0, 0, 1).astype(_np.float32)
            t2 = _np.clip((dist_mm - 2.0) / 3.0, 0, 1).astype(_np.float32)
            for ch in range(3):
                colors[:, ch] = GREEN[ch]*(1-t1) + YELLOW[ch]*t1*(1-t2) + RED[ch]*t1*t2
            return pts, dist_mm, colors
        self._start_worker("1.7_reg_error", _task)

    def _slot_1_3_add_scan(self) -> None:
        self._hdr("Add Scan Station", "Load additional scan station to merge with existing scans.")
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Scan Station", "",
            "Point Clouds (*.las *.laz *.ply);;All Files (*.*)")
        if not fp: return
        max_pts = self._ask_max_points(fp)
        if max_pts is None: return
        self._start_worker("1.3_add_scan", lambda: self.base_mod.load_scan(fp, max_points=max_pts))

    def _slot_1_4_merge(self) -> None:
        self._hdr("Register & Merge Stations",
                  "Register all scan stations to reference and merge into one point cloud.")
        if len(self.context.scans) < 2:
            self._log("Load at least 2 scan stations first (1.1 + 1.3)."); return
        self._log(f"Registering {len(self.context.scans)} stations...")
        self._start_worker("1.4_merge", lambda: self.reg_mod.register_and_merge(self.context))

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

    def _slot_2_4_semantic(self) -> None:
        self._hdr("Semantic Noise Removal (PDF 3.2)",
                  "Remove cables, lights and people using geometric feature classification.")
        self._start_worker("2.4_semantic",
            lambda: self.pre_mod.semantic_noise_removal(self.context))

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

    def _init_rag(self) -> None:
        msg = self.rag_mod.initialize()
        # Use QTimer to log after UI is fully built
        try:
            QtCore.QTimer.singleShot(500, lambda: self._log(f"[RAG] {msg}"))
        except Exception:
            pass

    def _slot_auto_pipeline(self) -> None:
        """Run full analysis pipeline in sequence: voxel -> SOR -> lining -> centerline -> params -> sections."""
        if self.context.active_scan is None:
            QtWidgets.QMessageBox.warning(self, "Auto Pipeline",
                "Please load a point cloud first (Step 1.1).")
            return
        self._hdr("Auto Pipeline", "Running full analysis pipeline automatically...")
        self._log("=" * 50)
        self._log("AUTO PIPELINE STARTED")
        self._log("=" * 50)
        if hasattr(self, "_auto_btn"):
            self._auto_btn.setEnabled(False)
            self._auto_btn.setText("Running pipeline...")
        self._auto_step = 0
        self._auto_steps = [
            ("2.1_voxel",      lambda: self.pre_mod.voxel_downsample(self.context),
             "Step 1/7: Voxel downsampling..."),
            ("2.2_sor",        lambda: self.pre_mod.statistical_outlier_removal_run(self.context),
             "Step 2/7: Statistical outlier removal..."),
            ("2.3_lining",     lambda: self.pre_mod.extract_tunnel_lining(self.context),
             "Step 3/7: Tunnel lining extraction..."),
            ("4.1_centerline", lambda: self.geo_mod.extract_centerline(self.context),
             "Step 4/7: Centerline extraction..."),
            ("4.3b_bspline",   lambda: self.geo_mod.extract_centerline_bspline(self.context),
             "Step 5/7: B-spline centerline..."),
            ("5.7_sections",   lambda: self.par_mod.compute_all_sections(
                self.context,
                vl_box_w=self._sp_vl_w.value(),
                vl_box_h=self._sp_vl_h.value(),
                vl_cir_r=self._sp_vl_r.value()),
             "Step 6/7: 2D section analysis..."),
            ("auto_params",    lambda: self._auto_extract_params(),
             "Step 7/7: Parameter extraction..."),
        ]
        self._run_next_auto_step()

    def _auto_extract_params(self) -> Dict:
        par = self.par_mod
        result = {}
        result.update(par.calc_arch_settlement(self.context))
        result.update(par.calc_horizontal_convergence(self.context))
        result.update(par.calc_ovality(self.context))
        result.update(par.calc_eccentricity(self.context))
        return result

    def _run_next_auto_step(self) -> None:
        if self._auto_step >= len(self._auto_steps):
            self._on_auto_pipeline_done()
            return
        key, task, msg = self._auto_steps[self._auto_step]
        total = len(self._auto_steps)
        pct = int(self._auto_step / total * 100)
        self.sb_prog.setValue(pct)
        step_label = f"[{self._auto_step+1}/{total}] {msg}"
        self._log(step_label)
        self.sb_msg.setText(step_label)
        if hasattr(self, "_auto_btn"):
            self._auto_btn.setText(f"Running... {pct}%  ({self._auto_step+1}/{total})")
        self._start_worker(key, task)

    def _on_auto_pipeline_done(self) -> None:
        if hasattr(self, "_auto_btn"):
            self._auto_btn.setEnabled(True)
            self._auto_btn.setText("AUTO PIPELINE  (1-click full analysis)")
        self._log("=" * 50)
        self._log("AUTO PIPELINE COMPLETE")
        p = self.context.parameters
        if p:
            self._log("--- Results Summary ---")
            for k, v in p.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    self._log(f"  {k}: {v:.3f}")
        n_viol = sum(1 for s in self.context.sections if s.clearance_violation)
        if n_viol:
            self._log(f"  WARNING: {n_viol} clearance violation(s) detected!")
        self._log("=" * 50)
        self.right_tabs.setCurrentIndex(self._section_tab_idx)
        QtWidgets.QMessageBox.information(self, "Auto Pipeline Complete",
            f"Pipeline finished successfully!\n\n"
            f"Sections analyzed: {len(self.context.sections)}\n"
            f"Clearance violations: {n_viol}\n\n"
            f"Check the 2D Cross-Section tab for results.")

    def _slot_target_detect(self) -> None:
        """Auto-detect targets with configurable parameters."""
        if self.context.active_scan is None:
            self._log("Load a scan first."); return
        dlg = _TargetDetectDialog(self.context.active_scan, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted: return
        params = dlg.get_params()
        scan_idx = self.context.active_index
        self._log(f"Detecting targets in {len(self.context.active_scan.points):,} pts (max 300K for speed)...")
        def _task():
            import numpy as _np
            from tunnel_analysis.models import PointCloudBundle as _PCB
            # Subsample to max 300K pts for fast detection
            scan = self.context.active_scan
            pts = scan.points
            intensity = scan.intensity
            MAX_DET = 100_000
            if len(pts) > MAX_DET:
                step = max(1, len(pts) // MAX_DET)
                pts_d = pts[::step]
                int_d = intensity[::step] if intensity is not None else None
            else:
                pts_d = pts; int_d = intensity
            b_det = _PCB(points=pts_d, intensity=int_d, path=scan.path)
            return self.tgt_mod.detect_all(
                b_det, scan_idx=scan_idx,
                detect_sphere=params["detect_sphere"],
                detect_flat=params["detect_flat"],
                detect_intensity=params["detect_intensity"],
                sphere_radius_range=params["sphere_radius_range"],
                intensity_percentile=params["intensity_percentile"],
                min_cluster_pts=params["min_cluster_pts"],
                cell_size_range=params.get("cell_size_range", (0.05, 0.30)),
                min_contrast_ratio=params.get("min_contrast_ratio", 2.0))
        self._start_worker("target_detect", _task)

    def _slot_target_manual(self) -> None:
        """Toggle manual target picking mode."""
        if self.plotter is None:
            self._log("Load a point cloud first."); return
        if self._manual_pick_mode:
            self._stop_manual_pick()
        else:
            self._start_manual_pick()

    def _start_manual_pick(self) -> None:
        """Start manual target picking mode."""
        self._manual_pick_mode = True
        self._hdr("Manual Target Picking",
                  "Click on target location in 3D viewport. Tool will auto-refine position.")
        # Update button appearance
        for i in range(self.right_tabs.count()):
            if self.right_tabs.tabText(i) == "Targets":
                self.right_tabs.setCurrentIndex(i); break
        # Show instruction overlay
        if self.plotter:
            self.plotter.add_text(
                "PICK MODE: Click on target location" + chr(10) + "Press [+ Manual] again to exit",
                position="lower_left", font_size=10,
                color="#F59E0B", name="pick_instruction")
            self.plotter.render()
        try:
            self.plotter.enable_point_picking(
                callback=self._on_manual_target_pick,
                show_message=False, color="#F59E0B",
                point_size=14, use_picker=True,
                pickable_window=False)
        except Exception as e:
            self._log(f"Pick mode error: {e}")
        self.sb_msg.setText(
            "PICK MODE active — Click on target in 3D viewport | Click [+ Manual] again to exit")
        self._log("Manual pick mode started. Click on target locations in 3D viewport.")

    def _stop_manual_pick(self) -> None:
        """Stop manual target picking mode."""
        self._manual_pick_mode = False
        if self.plotter:
            try:
                self.plotter.remove_actor("pick_instruction")
                self.plotter.disable_picking()
            except Exception:
                pass
            self.plotter.render()
        self.sb_msg.setText("Pick mode stopped.")
        self._log(f"Manual pick mode stopped. Total targets: {len(self._targets)}")

    def _on_manual_target_pick(self, point) -> None:
        """Handle manual target pick with auto-refinement."""
        if point is None: return
        pts = self.context.working_points
        if pts is None: return
        pick_pt = np.asarray(point, dtype=np.float64)
        n_manual = sum(1 for x in self._targets if x.type == "manual") + 1
        name = "M" + str(n_manual).zfill(2)

        # Auto-refine: fit local plane and find centroid
        refined_center, normal, residual_mm = self._refine_target_position(pick_pt, pts)

        t = Target(
            type="manual",
            name=name,
            center=refined_center,
            normal=normal,
            confidence=1.0,
            n_points=0,
            residual_mm=residual_mm,
            scan_idx=self.context.active_index)
        self._targets.append(t)
        self._refresh_target_table()

        # Show marker immediately
        if self.plotter:
            try:
                self.plotter.add_point_labels(
                    [refined_center], [name],
                    font_size=12, text_color="#F59E0B",
                    bold=True, show_points=True,
                    point_color="#F59E0B", point_size=18,
                    name="tgt_" + t.id, reset_camera=False)
                # Flash effect - temporary large sphere
                import pyvista as _pv
                sp = _pv.Sphere(radius=0.05, center=refined_center)
                self.plotter.add_mesh(sp, color="#F59E0B", opacity=0.6,
                    name="tgt_flash_" + t.id, reset_camera=False)
                self.plotter.render()
            except Exception:
                pass

        self._log(f"Target {name} placed at {np.round(refined_center,3).tolist()} "
                  f"(residual={residual_mm:.1f}mm)")

    def _refine_target_position(
        self, pick_pt: np.ndarray, pts: np.ndarray,
        search_r: float = 0.25
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Refine picked point to local plane centroid."""
        if cKDTree is None:
            return pick_pt, None, 0.0
        try:
            from scipy.spatial import cKDTree as _kd
            tree = _kd(pts)
            local_idx = tree.query_ball_point(pick_pt, search_r)
            if len(local_idx) < 10:
                return pick_pt, None, 0.0
            lp = pts[local_idx]
            # Fit plane
            result = self.tgt_mod._fit_plane(lp, tol=0.015)
            if result is None:
                return lp.mean(axis=0), None, 0.0
            normal, centroid, thickness, inliers = result
            return centroid, normal, thickness * 1e3
        except Exception:
            return pick_pt, None, 0.0

    def _slot_target_match(self) -> None:
        """Auto-match targets between scan stations."""
        if len(self.context.scans) < 2:
            self._log("Need at least 2 scan stations."); return
        src_t = [t for t in self._targets if t.scan_idx == 0]
        tgt_t = [t for t in self._targets if t.scan_idx == 1]
        if not src_t or not tgt_t:
            self._log("Detect targets in both stations first."); return
        matches = self.tgt_mod.match_targets(src_t, tgt_t, max_dist=5.0)
        self._refresh_target_table()
        self._log(f"Auto-matched {len(matches)} target pairs:")
        for st, tt, d in matches:
            self._log(f"  {st.name} <-> {tt.name}  dist={d:.3f}m")

    def _slot_target_register(self) -> None:
        """Register scans using matched targets."""
        src_t = [t for t in self._targets if t.scan_idx == 0 and t.matched_id]
        tgt_t = [t for t in self._targets if t.scan_idx == 1 and t.matched_id]
        if len(src_t) < 3:
            self._log("Need >= 3 matched target pairs. Run Auto Match first."); return
        def _task():
            T, rmse, residuals = self.tgt_mod.register_by_targets(src_t, tgt_t)
            pts = self.context.working_points
            if pts is not None:
                reg_pts = self.tgt_mod.apply_transform(pts, T)
            else:
                reg_pts = None
            return T, rmse, residuals, reg_pts
        self._start_worker("target_register", _task)

    def _refresh_target_table(self) -> None:
        """Update target table widget."""
        if not hasattr(self, "_target_table"): return
        self._target_table.setRowCount(0)
        type_colors = {
            "sphere": "#1D4ED8", "flat": "#047857",
            "intensity": "#D97706", "manual": "#DC2626"}
        for t in self._targets:
            row = self._target_table.rowCount()
            self._target_table.insertRow(row)
            c = t.center if t.center is not None else np.zeros(3)
            matched = " *" if t.matched_id else ""
            vals = [
                t.name + matched, t.type,
                "S" + str(t.scan_idx + 1),
                f"{c[0]:.3f}", f"{c[1]:.3f}", f"{c[2]:.3f}",
                f"{t.confidence:.2f}"]
            for col, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(val)
                item.setData(QtCore.Qt.UserRole, t.id)
                color = type_colors.get(t.type, "#111827")
                if col <= 1:
                    item.setForeground(QtGui.QColor(color))
                    item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
                self._target_table.setItem(row, col, item)
        n = len(self._targets)
        n_matched = sum(1 for t in self._targets if t.matched_id)
        if hasattr(self, "_tgt_status"):
            self._tgt_status.setText(
                f"{n} targets  |  {n_matched} matched  |  "
                f"sphere:{sum(1 for t in self._targets if t.type=='sphere')}  "
                f"flat:{sum(1 for t in self._targets if t.type=='flat')}  "
                f"intensity:{sum(1 for t in self._targets if t.type=='intensity')}  "
                f"manual:{sum(1 for t in self._targets if t.type=='manual')}")

    def _render_target_markers(self) -> None:
        """Render target markers on 3D viewport."""
        if self.plotter is None: return
        try: self.plotter.remove_actor("target_markers")
        except Exception: pass
        if not self._targets: return
        type_colors = {
            "sphere": "#1D4ED8", "flat": "#047857",
            "intensity": "#D97706", "manual": "#DC2626"}
        for t in self._targets:
            if t.center is None: continue
            color = type_colors.get(t.type, "#888888")
            label = t.name + (" *" if t.matched_id else "")
            try:
                self.plotter.add_point_labels(
                    [t.center], [label],
                    font_size=10, text_color=color,
                    bold=True, show_points=True,
                    point_color=color, point_size=16,
                    name="tgt_" + t.id, reset_camera=False)
            except Exception:
                pass
        self.plotter.render()

    def _on_target_selected(self, row: int, col: int) -> None:
        """Focus camera on selected target."""
        item = self._target_table.item(row, 0)
        if item is None: return
        tid = item.data(QtCore.Qt.UserRole)
        t = next((x for x in self._targets if x.id == tid), None)
        if t is None or t.center is None: return
        if self.plotter:
            self.plotter.camera.focal_point = t.center.tolist()
            self.plotter.render()

    def _target_context_menu(self, pos) -> None:
        """Right-click menu for target table."""
        row = self._target_table.rowAt(pos.y())
        if row < 0: return
        item = self._target_table.item(row, 0)
        if item is None: return
        tid = item.data(QtCore.Qt.UserRole)
        t = next((x for x in self._targets if x.id == tid), None)
        if t is None: return
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:4px;padding:4px;}"
            "QMenu::item{padding:6px 20px;color:#111827;font-size:9pt;}"
            "QMenu::item:selected{background:#DBEAFE;color:#1D4ED8;}")
        act_focus = menu.addAction("Focus Camera")
        act_focus.triggered.connect(lambda: self._on_target_selected(row, 0))
        act_rename = menu.addAction("Rename...")
        act_rename.triggered.connect(lambda: self._rename_target(t))
        act_unmatch = menu.addAction("Unmatch")
        act_unmatch.triggered.connect(lambda: self._unmatch_target(t))
        menu.addSeparator()
        act_del = menu.addAction("Delete")
        act_del.triggered.connect(lambda: self._delete_target(t))
        menu.exec(self._target_table.viewport().mapToGlobal(pos))

    def _rename_target(self, t: Target) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Target", "Name:", text=t.name)
        if ok and name.strip():
            t.name = name.strip()
            self._refresh_target_table()

    def _unmatch_target(self, t: Target) -> None:
        paired = next((x for x in self._targets if x.id == t.matched_id), None)
        if paired: paired.matched_id = ""
        t.matched_id = ""
        self._refresh_target_table()
        self._log(f"Target {t.name} unmatched.")

    def _delete_target(self, t: Target) -> None:
        self._targets = [x for x in self._targets if x.id != t.id]
        try: self.plotter.remove_actor("tgt_" + t.id)
        except Exception: pass
        self._refresh_target_table()
        self._log(f"Target {t.name} deleted.")

    def _slot_copy_clipboard(self) -> None:
        """Copy current results to clipboard as text."""
        lines = []
        p = self.context.parameters
        if p:
            lines.append("=== Tunnel Analysis Results ===")
            for k, v in p.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    lines.append(f"  {k}: {v:.3f}")
        if self.context.sections:
            lines.append(f"Sections analyzed: {len(self.context.sections)}")
            n_viol = sum(1 for s in self.context.sections if s.clearance_violation)
            lines.append(f"Clearance violations: {n_viol}")
            lines.append("")
            lines.append("Chainage(m) | H1(m) | W1(m) | Ovality(%) | Ecc(mm)")
            lines.append("-" * 55)
            for s in self.context.sections:
                h1 = f"{s.H1:.3f}" if np.isfinite(s.H1) else "-"
                w1 = f"{s.W1:.3f}" if np.isfinite(s.W1) else "-"
                ov = f"{s.ovality:.2f}" if np.isfinite(s.ovality) else "-"
                ec = f"{s.eccentricity:.1f}" if np.isfinite(s.eccentricity) else "-"
                lines.append(f"{s.chainage:.2f}       | {h1}   | {w1}   | {ov}       | {ec}")
        if not lines:
            self._log("No results to copy. Run analysis first."); return
        text = chr(10).join(lines)
        QtWidgets.QApplication.clipboard().setText(text)
        self._log(f"Results copied to clipboard ({len(lines)} lines).")
        QtWidgets.QMessageBox.information(self, "Copied",
            f"Results copied to clipboard ({len(lines)} lines).")

    def _slot_reset_pipeline(self) -> None:
        """Reset pipeline — clear all results, keep raw scans."""
        reply = QtWidgets.QMessageBox.question(self, "Reset Pipeline",
            "Clear all analysis results?" + chr(10) +
            "(Raw scan data will be kept)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes: return
        self.context.normalized_points  = None
        self.context.registered_points  = None
        self.context.centerline         = None
        self.context.centerline_smooth  = None
        self.context.frenet_frames      = []
        self.context.parameters         = {}
        self.context.heatmap_scalars    = None
        self.context.sections           = []
        self.context.polar_map          = None
        self.context.polar_angles       = None
        self.context.polar_centers      = None
        self._targets                   = []
        self._noise_pts                 = None
        self._kept_pts                  = None
        if hasattr(self, "_noise_panel") and self._noise_panel:
            self._noise_panel.deleteLater()
            self._noise_panel = None
        if self.plotter:
            try:
                self.plotter.clear()
                self.plotter.set_background("#F8FAFC")
                self.plotter.render()
            except Exception: pass
        self.results_text.clear()
        self.pt_label.setText("Points: --")
        self.sb_pts.setText("Points: --")
        self.sb_msg.setText("Pipeline reset. Raw scans preserved.")
        self.sb_prog.setValue(0)
        self._refresh_target_table()
        self._log("Pipeline reset complete. Raw scans preserved.")

    def _slot_4_3b_bspline(self) -> None:
        self._hdr("B-Spline C2 Centerline (PDF 3.4)", "Sliding-window curvature detection + B-spline C2 fit.")
        self._start_worker("4.3b_bspline", lambda: self.geo_mod.extract_centerline_bspline(self.context))

    def _slot_4_5b_intensity_seams(self) -> None:
        self._hdr("Intensity Ring Seam Detection (PDF 3.3)", "Detect ring seams from LiDAR intensity derivative.")
        if self.context.active_scan is None or self.context.active_scan.intensity is None:
            self._log("Intensity data required. Load a scan with intensity channel first."); return
        self._start_worker("4.5b_intensity_seams", lambda: self.seg_mod.detect_ring_seams_by_intensity(self.context))

    def _slot_5_3b_hausdorff(self) -> None:
        self._hdr("Hausdorff Heatmap T0→Tn (PDF 3.5)", "Surface distance heatmap between reference and current scan.")
        if len(self.context.scans) < 2:
            self._log("Load at least 2 scans (T0 and Tn) first."); return
        ref = self.context.scans[0].points
        self._start_worker("5.3b_hausdorff", lambda: self.par_mod.generate_hausdorff_heatmap(self.context, ref))

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

    def _slot_5_8_clearance_3d(self) -> None:
        self._hdr("Clearance 3D Violation Map (PDF 3.6)",
                  "Highlight points violating vehicle clearance envelope on 3D viewport.")
        if not self.context.sections:
            self._log("Run Step 5.7 first."); return
        def _task():
            import numpy as _np
            pts = self.context.working_points
            if pts is None: raise RuntimeError("No point cloud.")
            pts = validate_xyz(pts)
            # Get clearance violations from sections
            viol_centers = [s.center_3d for s in self.context.sections
                            if s.clearance_violation and s.center_3d is not None]
            if not viol_centers:
                return pts, _np.zeros(len(pts), dtype=bool), 0
            # Mark points near violation section centers
            from scipy.spatial import cKDTree as _kd
            viol_arr = _np.array(viol_centers)
            tree = _kd(viol_arr)
            d, _ = tree.query(pts, k=1, workers=-1)
            viol_mask = d < 0.5
            return pts, viol_mask, int(viol_mask.sum())
        self._start_worker("5.8_clearance_3d", _task)

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

    def _slot_8_1_csv(self) -> None:
        self._hdr("Export CSV", "Export section parameters to CSV file.")
        if not self.context.sections and not self.context.parameters:
            self._log("Run parameter extraction first (Step 5)."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "tunnel_report.csv", "CSV Files (*.csv)")
        if not path: return
        self._start_worker("8.1_csv", lambda: self.exp_mod.export_csv(self.context, path))

    def _slot_8_2_excel(self) -> None:
        self._hdr("Export Excel Report", "Export full analysis report with charts and warnings.")
        if not self.context.sections and not self.context.parameters:
            self._log("Run parameter extraction first (Step 5)."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Excel Report", "tunnel_report.xlsx", "Excel Files (*.xlsx)")
        if not path: return
        scan = self.context.active_scan
        proj = scan.path if scan and scan.path else "Tunnel Analysis"
        self._start_worker("8.2_excel", lambda: self.exp_mod.export_excel(
            self.context, path, project_name=proj, engineer="CBNU Smart Structure Lab"))

    def _slot_8_4_web(self) -> None:
        self._hdr("Web Dashboard", "Launch interactive web dashboard in browser.")
        import threading, webbrowser
        from ..web_dashboard import build_app
        def _launch():
            try:
                app = build_app(self.context)
                port = 8050
                url = f"http://127.0.0.1:{port}"
                threading.Timer(1.5, lambda: webbrowser.open(url)).start()
                app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
                return url
            except Exception as e:
                raise RuntimeError(f"Dashboard error: {e}")
        t = threading.Thread(target=_launch, daemon=True)
        t.start()
        self._log("Web dashboard starting at http://127.0.0.1:8050 ...")
        import time; time.sleep(1.5)
        import webbrowser; webbrowser.open("http://127.0.0.1:8050")

    def _slot_8_3_pdf(self) -> None:
        self._hdr("Export PDF Report", "Generate professional PDF inspection report.")
        if not self.context.sections and not self.context.parameters:
            self._log("Run parameter extraction first (Step 5)."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save PDF Report", "tunnel_report.pdf", "PDF Files (*.pdf)")
        if not path: return
        scan = self.context.active_scan
        proj = scan.path if scan and scan.path else "Tunnel Analysis"
        self._start_worker("8.3_pdf", lambda: self.pdf_mod.export_pdf(
            self.context, path, project_name=proj, engineer="CBNU Smart Structure Lab"))

    def _slot_7_1_ifc(self) -> None:
        self._hdr("IFC/BIM Export (IFC4)", "Export tunnel geometry and parameters to IFC4 format.")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save IFC Model", "tunnel_model.ifc", "IFC Files (*.ifc)")
        if not path: return
        scan = self.context.active_scan
        proj = scan.path if scan and scan.path else "Tunnel Analysis"
        self._start_worker("7.1_ifc", lambda: self.ifc_mod.export_ifc(
            self.context, path, project_name=proj, engineer="CBNU Smart Structure Lab"))

    def _slot_7_2_query_ai(self) -> None:
        self._hdr("AI Engineering Assistant (RAG)", "Query local LLM with safety standards knowledge base.")
        prompt = self.ai_prompt.toPlainText().strip() or "Summarize the tunnel inspection results and identify locations that require engineering attention."
        self.right_tabs.setCurrentIndex(self._ai_tab_idx)
        self._start_worker("7.2_ai", lambda: self.rag_mod.query(prompt, self.context))

    def _check_auto_pipeline(self, key: str) -> None:
        """After each worker finishes, check if we are in auto pipeline mode."""
        if not hasattr(self, "_auto_steps") or not hasattr(self, "_auto_step"):
            return
        if self._auto_step >= len(self._auto_steps):
            return
        current_key = self._auto_steps[self._auto_step][0]
        if key == current_key:
            self._auto_step += 1
            QtCore.QTimer.singleShot(200, self._run_next_auto_step)

