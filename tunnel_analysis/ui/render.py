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

class RenderMixin:
    def _render_bundle(self, b: PointCloudBundle, title: str) -> None:
        mesh = b.cloud or make_vertex_cloud(b.points, b.intensity, b.colors_raw); self._render_mesh(mesh, title)

    def _show_noise_panel(self) -> None:
        """Show interactive noise review panel above viewport."""
        if self._noise_panel:
            self._noise_panel.deleteLater()
        panel = QtWidgets.QFrame()
        panel.setStyleSheet(
            "QFrame{background:#FEF3C7;border:2px solid #D97706;"
            "border-radius:8px;padding:4px;}")
        lay = QtWidgets.QHBoxLayout(panel)
        lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(8)
        lbl = QtWidgets.QLabel(
            f"Noise review: {len(self._noise_pts):,} noise points (red) | "
            f"{len(self._kept_pts):,} kept (blue)")
        lbl.setStyleSheet("color:#92400E;font-weight:700;font-size:9.5pt;")
        btn_style_green = (
            "QPushButton{background:#047857;color:white;border-radius:5px;"
            "padding:5px 14px;font-weight:700;border:none;}"
            "QPushButton:hover{background:#065F46;}")
        btn_style_red = (
            "QPushButton{background:#DC2626;color:white;border-radius:5px;"
            "padding:5px 14px;font-weight:700;border:none;}"
            "QPushButton:hover{background:#B91C1C;}")
        btn_style_blue = (
            "QPushButton{background:#1D4ED8;color:white;border-radius:5px;"
            "padding:5px 14px;font-weight:700;border:none;}"
            "QPushButton:hover{background:#1E40AF;}")
        btn_style_gray = (
            "QPushButton{background:#64748B;color:white;border-radius:5px;"
            "padding:5px 14px;font-weight:700;border:none;}"
            "QPushButton:hover{background:#475569;}")
        btn_confirm = QtWidgets.QPushButton("✓ Confirm Remove")
        btn_add     = QtWidgets.QPushButton("+ Select More Noise")
        btn_restore = QtWidgets.QPushButton("↩ Restore Point")
        btn_cancel  = QtWidgets.QPushButton("✗ Keep All")
        btn_confirm.setStyleSheet(btn_style_green)
        btn_add.setStyleSheet(btn_style_red)
        btn_restore.setStyleSheet(btn_style_blue)
        btn_cancel.setStyleSheet(btn_style_gray)
        btn_confirm.setToolTip("Remove all red noise points and keep blue points")
        btn_add.setToolTip("Click points in 3D viewport to mark as noise")
        btn_restore.setToolTip("Click red points to restore them")
        btn_cancel.setToolTip("Cancel — keep all points including noise")
        btn_confirm.clicked.connect(self._confirm_noise_removal)
        btn_add.clicked.connect(self._start_add_noise_selection)
        btn_restore.clicked.connect(self._start_restore_selection)
        btn_cancel.clicked.connect(self._cancel_noise_removal)
        lay.addWidget(lbl, 1)
        lay.addWidget(btn_add)
        lay.addWidget(btn_restore)
        lay.addWidget(btn_confirm)
        lay.addWidget(btn_cancel)
        self._noise_panel = panel
        # Insert panel above viewport
        self.vp_layout.insertWidget(0, panel)

    def _confirm_noise_removal(self) -> None:
        """Apply noise removal — keep only kept_pts."""
        if self._kept_pts is None: return
        self.context.normalized_points = self._kept_pts
        self._render_pts(self._kept_pts, "2.2 Noise Removed — Clean Point Cloud", "#0EA5E9")
        self.pt_label.setText(f"Points: {len(self._kept_pts):,}")
        self.sb_pts.setText(f"Points: {len(self._kept_pts):,}")
        self._log(f"Noise removal confirmed: {len(self._kept_pts):,} clean points retained.")
        if self._noise_pts is not None:
            self._log(f"Removed: {len(self._noise_pts):,} noise points.")
        self._noise_pts = None
        self._hide_noise_panel()

    def _cancel_noise_removal(self) -> None:
        """Cancel — keep all points including noise."""
        if self.context.active_scan:
            all_pts = validate_xyz(self.context.active_scan.points)
            self.context.normalized_points = all_pts
            self._render_pts(all_pts, "2.2 Cancelled — All Points Kept", "#64748B")
            self.pt_label.setText(f"Points: {len(all_pts):,}")
            self.sb_pts.setText(f"Points: {len(all_pts):,}")
        self._log("Noise removal cancelled — all points kept.")
        self._noise_pts = None; self._kept_pts = None
        self._hide_noise_panel()

    def _start_add_noise_selection(self) -> None:
        """Enable picking mode — click points to mark as noise."""
        if self.plotter is None: return
        self._log("Click on points in 3D viewport to mark as noise. Click again to deselect.")
        try:
            self.plotter.enable_point_picking(
                callback=self._on_pick_noise,
                show_message=True,
                color="#DC2626",
                point_size=10,
                use_picker=True,
                pickable_window=False)
            self.sb_msg.setText("Pick mode: click points to mark as noise. Press Q to exit.")
        except Exception as e:
            self._log(f"Pick mode: {e}")

    def _start_restore_selection(self) -> None:
        """Enable picking mode — click red points to restore them."""
        if self.plotter is None: return
        self._log("Click on red noise points to restore them.")
        try:
            self.plotter.enable_point_picking(
                callback=self._on_pick_restore,
                show_message=True,
                color="#2563EB",
                point_size=10,
                use_picker=True,
                pickable_window=False)
            self.sb_msg.setText("Pick mode: click red points to restore. Press Q to exit.")
        except Exception as e:
            self._log(f"Pick mode: {e}")

    def _on_pick_noise(self, point) -> None:
        """Mark picked point as noise."""
        if self._kept_pts is None or point is None: return
        pt = np.asarray(point, dtype=np.float64)
        dists = np.linalg.norm(self._kept_pts - pt, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > 0.5: return  # too far
        new_noise = self._kept_pts[idx:idx+1]
        self._kept_pts  = np.delete(self._kept_pts, idx, axis=0)
        self._noise_pts = np.vstack([self._noise_pts, new_noise]) if self._noise_pts is not None and len(self._noise_pts) else new_noise
        self._render_filter_result(self._kept_pts, self._noise_pts,
                                    "2.2 SOR — Review noise (red) before confirming")
        # Update panel label
        if self._noise_panel:
            lbl = self._noise_panel.findChild(QtWidgets.QLabel)
            if lbl:
                lbl.setText(f"Noise review: {len(self._noise_pts):,} noise (red) | {len(self._kept_pts):,} kept (blue)")

    def _on_pick_restore(self, point) -> None:
        """Restore picked noise point back to kept."""
        if self._noise_pts is None or point is None: return
        pt = np.asarray(point, dtype=np.float64)
        dists = np.linalg.norm(self._noise_pts - pt, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > 0.5: return
        restored = self._noise_pts[idx:idx+1]
        self._noise_pts = np.delete(self._noise_pts, idx, axis=0)
        self._kept_pts  = np.vstack([self._kept_pts, restored]) if self._kept_pts is not None and len(self._kept_pts) else restored
        self._render_filter_result(self._kept_pts, self._noise_pts,
                                    "2.2 SOR — Review noise (red) before confirming")
        if self._noise_panel:
            lbl = self._noise_panel.findChild(QtWidgets.QLabel)
            if lbl:
                lbl.setText(f"Noise review: {len(self._noise_pts):,} noise (red) | {len(self._kept_pts):,} kept (blue)")

    def _hide_noise_panel(self) -> None:
        if self._noise_panel:
            self._noise_panel.deleteLater()
            self._noise_panel = None

    def _render_filter_result(self, kept_pts: np.ndarray, removed_pts: np.ndarray, title: str) -> None:
        if self.plotter is None:
            return
        self.plotter.clear(); self.plotter.set_background("#F8FAFC")
        if len(kept_pts):
            kept = make_vertex_cloud(kept_pts)
            self.plotter.add_mesh(kept, color="#0EA5E9", style="points", point_size=2.4,
                                  render_points_as_spheres=False, reset_camera=True)
        if len(removed_pts):
            removed = make_vertex_cloud(removed_pts)
            self.plotter.add_mesh(removed, color="#DC2626", style="points", point_size=5.0,
                                  render_points_as_spheres=True, reset_camera=False)
            self.plotter.add_text(f"Removed noise/outliers: {len(removed_pts):,} red points",
                                  position="lower_left", font_size=10, color="#DC2626", name="removed")
        self.plotter.add_text(title + " | kept=blue, removed=red", position="upper_left",
                              font_size=11, color="#111827", name="ttl")
        self.plotter.add_axes(color="#111827")
        self.plotter.show_bounds(color="#94A3B8", grid="front", location="outer", font_size=8)
        self.plotter.camera.parallel_projection = True
        self.plotter.reset_camera(); self.plotter.render()

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


