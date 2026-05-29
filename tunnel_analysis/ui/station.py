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

class StationMixin:
    def _refresh_station_list(self) -> None:
        """Update station tree widget (Faro SCENE style)."""
        if not hasattr(self, "_station_tree"): return
        self._station_tree.clear()
        root = QtWidgets.QTreeWidgetItem(self._station_tree, ["Project"])
        root.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        root.setExpanded(True)
        scans_grp = QtWidgets.QTreeWidgetItem(root, ["Scans"])
        scans_grp.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))
        scans_grp.setExpanded(True)
        for i, sc in enumerate(self.context.scans):
            import pathlib
            color = self._station_colors[i % len(self._station_colors)]
            fname = pathlib.Path(sc.path).name if sc.path else ("scan_" + str(i+1))
            label = "S" + str(i+1) + "  " + fname
            item = QtWidgets.QTreeWidgetItem(scans_grp, [label])
            item.setCheckState(0, QtCore.Qt.Checked)
            item.setData(0, QtCore.Qt.UserRole, i)
            pix = QtGui.QPixmap(16, 16)
            pix.fill(QtGui.QColor(color))
            item.setIcon(0, QtGui.QIcon(pix))
            item.setFont(0, QtGui.QFont("Segoe UI", 9))
            tip = "Station " + str(i+1)
            if i == 0: tip = tip + " (Reference)"
            tip = tip + chr(10) + str(len(sc.points)) + " points"
            if sc.path: tip = tip + chr(10) + str(sc.path)
            item.setToolTip(0, tip)
            if i == 0:
                item.setForeground(0, QtGui.QColor("#DC2626"))
                item.setFont(0, QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        self._station_tree.expandAll()
        for i in range(self.right_tabs.count()):
            if self.right_tabs.tabText(i) == "Stations":
                self.right_tabs.setCurrentIndex(i)
                break


    def _render_station_markers(self) -> None:
        """Render colored sphere + label for each scan station on 3D viewport."""
        if self.plotter is None: return
        # Remove old markers
        for i in range(20):
            try: self.plotter.remove_actor(f"station_marker_{i}")
            except Exception: pass
            try: self.plotter.remove_actor(f"station_label_{i}")
            except Exception: pass
        for i, sc in enumerate(self.context.scans):
            try:
                pts = validate_xyz(sc.points)
                center = pts.mean(axis=0)
                color = self._station_colors[i % len(self._station_colors)]
                # Small dot marker at centroid
                label = "S" + str(i+1) + (" (Ref)" if i == 0 else "")
                self.plotter.add_point_labels(
                    [center], [label],
                    font_size=11, text_color=color,
                    bold=True, show_points=True,
                    point_color=color, point_size=12,
                    name="station_label_" + str(i),
                    reset_camera=False)
            except Exception as e:
                self._log(f"Station marker {i+1}: {e}")
        self.plotter.render()

    def _on_station_item_changed(self, item, column) -> None:
        """Handle checkbox toggle for station visibility."""
        idx = item.data(0, QtCore.Qt.UserRole)
        if idx is None: return
        if self.plotter is None: return
        is_visible = item.checkState(0) == QtCore.Qt.Checked
        try:
            if not is_visible:
                # Hide: remove point cloud actor
                self.plotter.remove_actor(f"station_pts_{idx}")
                self.plotter.remove_actor(f"station_marker_{idx}")
                self.plotter.remove_actor(f"station_label_{idx}")
                self.plotter.remove_actor(f"station_highlight_pts")
            else:
                # Show: re-render station
                sc = self.context.scans[idx]
                color = self._station_colors[idx % len(self._station_colors)]
                pts = validate_xyz(sc.points)
                step = max(1, len(pts) // 80000)
                mesh = make_vertex_cloud(pts[::step])
                self.plotter.add_mesh(mesh, color=color, style="points",
                    point_size=2.0, name=f"station_pts_{idx}", reset_camera=False)
                center = pts.mean(axis=0)
                label = "S" + str(idx+1) + (" (Ref)" if idx == 0 else "")
                self.plotter.add_point_labels(
                    [center], [label],
                    font_size=12, text_color=color,
                    bold=True, show_points=True,
                    point_color=color, point_size=14,
                    name=f"station_label_{idx}", reset_camera=False)
            self.plotter.render()
        except Exception as e:
            self._log(f"Visibility: {e}")

    def _on_station_tree_changed(self, current, previous) -> None:
        """Handle station tree selection change."""
        if current is None: return
        idx = current.data(0, QtCore.Qt.UserRole)
        if idx is None: return
        self._on_station_selected(idx)

    def _on_station_selected(self, idx: int) -> None:
        """Highlight selected station on 3D viewport."""
        if idx < 0 or idx >= len(self.context.scans): return
        if self.plotter is None: return
        sc = self.context.scans[idx]
        color = self._station_colors[idx % len(self._station_colors)]

        # Remove previous highlight
        try: self.plotter.remove_actor("station_highlight")
        except Exception: pass
        try: self.plotter.remove_actor("station_highlight_pts")
        except Exception: pass

        try:
            pts = validate_xyz(sc.points)
            center = pts.mean(axis=0)
            import pyvista as _pv

            # 1. Highlight point cloud with bright color
            step = max(1, len(pts) // 80000)
            mesh = make_vertex_cloud(pts[::step])
            self.plotter.add_mesh(mesh, color=color, style="points",
                point_size=3.5, opacity=0.9, name="station_highlight_pts",
                reset_camera=False)

            # 2. Large glowing sphere at centroid
            # No sphere - just highlight with point labels

            # 3. Camera focus on selected station
            self.plotter.camera.focal_point = center.tolist()
            self.plotter.camera.position = (
                center[0] + r * 8,
                center[1] + r * 8,
                center[2] + r * 6)
            self.plotter.render()

            # 4. Update status bar
            name = f"Station {idx+1}"
            if sc.path:
                import pathlib
                name += f" — {pathlib.Path(sc.path).name}"
            self.sb_msg.setText(f"Selected: {name}  |  {len(pts):,} pts  |  Color: {color}")
        except Exception as e:
            self._log(f"Station highlight error: {e}")

    def _station_context_menu(self, pos) -> None:
        """Right-click context menu for station tree (Faro SCENE style)."""
        item = self._station_tree.itemAt(pos)
        if item is None: return
        idx = item.data(0, QtCore.Qt.UserRole)
        if idx is None: return

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:4px;padding:4px;}
            QMenu::item{padding:6px 24px;color:#111827;font-size:9.5pt;}
            QMenu::item:selected{background:#DBEAFE;color:#1D4ED8;}
            QMenu::separator{height:1px;background:#E2E8F0;margin:4px 0;}
        """)

        # Visible toggle
        is_visible = item.checkState(0) == QtCore.Qt.Checked
        act_vis = menu.addAction("Hide" if is_visible else "Show")
        act_vis.triggered.connect(lambda: self._toggle_station_visibility(item, idx))

        menu.addSeparator()

        # Set as reference
        act_ref = menu.addAction("Set as Reference (S1)")
        act_ref.triggered.connect(lambda: self._set_station_reference(idx))

        # Focus camera
        act_focus = menu.addAction("Focus Camera Here")
        act_focus.triggered.connect(lambda: self._on_station_selected(idx))

        menu.addSeparator()

        # Rename
        act_rename = menu.addAction("Rename...")
        act_rename.triggered.connect(lambda: self._rename_station(item, idx))

        # Properties
        act_prop = menu.addAction("Properties...")
        act_prop.triggered.connect(lambda: self._show_station_properties(idx))

        menu.addSeparator()

        # Delete
        act_del = menu.addAction("Delete Station")
        act_del.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon))
        act_del.triggered.connect(lambda: self._delete_station(idx))

        menu.exec(self._station_tree.viewport().mapToGlobal(pos))

    def _toggle_station_visibility(self, item, idx: int) -> None:
        """Toggle station visibility on 3D viewport."""
        is_checked = item.checkState(0) == QtCore.Qt.Checked
        new_state = QtCore.Qt.Unchecked if is_checked else QtCore.Qt.Checked
        item.setCheckState(0, new_state)
        if self.plotter is None: return
        try:
            if new_state == QtCore.Qt.Unchecked:
                self.plotter.remove_actor(f"station_pts_{idx}")
                self.plotter.remove_actor(f"station_marker_{idx}")
                self.plotter.remove_actor(f"station_label_{idx}")
            else:
                sc = self.context.scans[idx]
                color = self._station_colors[idx % len(self._station_colors)]
                pts = validate_xyz(sc.points)
                step = max(1, len(pts) // 80000)
                mesh = make_vertex_cloud(pts[::step])
                self.plotter.add_mesh(mesh, color=color, style="points",
                    point_size=2.0, name=f"station_pts_{idx}", reset_camera=False)
            self.plotter.render()
        except Exception as e:
            self._log(f"Visibility toggle: {e}")

    def _set_station_reference(self, idx: int) -> None:
        """Move selected station to position 0 (reference)."""
        if idx == 0: self._log("Already reference station."); return
        sc = self.context.scans.pop(idx)
        self.context.scans.insert(0, sc)
        self.context.active_index = 0
        self._refresh_station_list()
        self._render_station_markers()
        self._log(f"Station {idx+1} set as reference (S1).")

    def _rename_station(self, item, idx: int) -> None:
        """Rename station via input dialog."""
        sc = self.context.scans[idx]
        import pathlib
        current = pathlib.Path(sc.path).stem if sc.path else f"Station_{idx+1}"
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Station", "Station name:", text=current)
        if ok and new_name.strip():
            if sc.path:
                sc.metadata["display_name"] = new_name.strip()
            item.setText(0, f"S{idx+1}  {new_name.strip()}")
            self._log(f"Station {idx+1} renamed to: {new_name.strip()}")

    def _show_station_properties(self, idx: int) -> None:
        """Show station properties dialog."""
        sc = self.context.scans[idx]
        import pathlib
        lines = [
            f"Station: {idx+1}" + (" (Reference)" if idx == 0 else ""),
            f"File: {sc.path or 'N/A'}",
            f"Points: {len(sc.points):,}",
            f"Has intensity: {sc.intensity is not None}",
            f"Has colors: {sc.colors_raw is not None}",
        ]
        if sc.metadata:
            for k, v in sc.metadata.items():
                lines.append(f"{k}: {v}")
        QtWidgets.QMessageBox.information(
            self, f"Station {idx+1} Properties",
            chr(10).join(lines))

    def _delete_station(self, idx: int) -> None:
        """Delete a scan station."""
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Station",
            f"Delete Station {idx+1}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes: return
        self.context.scans.pop(idx)
        if self.context.active_index >= len(self.context.scans):
            self.context.active_index = len(self.context.scans) - 1
        self._refresh_station_list()
        self._render_station_markers()
        self._log(f"Station {idx+1} deleted.")

    def _clear_all_stations(self) -> None:
        """Clear all loaded scan stations."""
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All Stations",
            "Remove all loaded scan stations?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes: return
        self.context.scans.clear()
        self.context.active_index = -1
        self.context.normalized_points = None
        self.context.registered_points = None
        if hasattr(self, "_station_list"):
            self._station_list.clear()
        for i in range(20):
            try: self.plotter.remove_actor(f"station_marker_{i}")
            except Exception: pass
            try: self.plotter.remove_actor(f"station_label_{i}")
            except Exception: pass
        if self.plotter: self.plotter.render()
        self._log("All scan stations cleared.")

    def _highlight_section(self, idx: int) -> None:
        """Highlight current section plane on 3D viewport."""
        if self.plotter is None: return
        sections = self.context.sections
        frames   = self.context.frenet_frames
        cl       = self.context.centerline
        if not sections or not frames or cl is None: return
        if idx < 0 or idx >= len(sections): return

        sg  = sections[idx]
        fr  = frames[min(idx, len(frames) - 1)]
        C   = np.asarray(fr["center"], dtype=np.float64)
        N   = np.asarray(fr["N"],      dtype=np.float64)
        B   = np.asarray(fr["B"],      dtype=np.float64)

        # Remove previous section marker
        try: self.plotter.remove_actor("sec_plane"); self.plotter.remove_actor("sec_center")
        except Exception: pass

        # Draw section disc
        import pyvista as _pv
        radius = float(sg.radius_fit) if np.isfinite(sg.radius_fit) else 4.0
        disc = _pv.Disc(center=C, normal=fr["T"], inner=0, outer=radius * 1.05, r_res=1, c_res=60)
        self.plotter.add_mesh(disc, color="#F59E0B", opacity=0.35, style="surface",
                              name="sec_plane", reset_camera=False)

        # Draw center point
        sphere = _pv.Sphere(radius=radius * 0.04, center=C)
        self.plotter.add_mesh(sphere, color="#EF4444", name="sec_center", reset_camera=False)

        # Draw N and B axes
        for vec, col, nm in [(N, "#16A34A", "sec_N"), (B, "#2563EB", "sec_B")]:
            ln = np.vstack([C, C + vec * radius * 0.6])
            try: self.plotter.remove_actor(nm)
            except Exception: pass
            self.plotter.add_lines(ln, color=col, width=3, connected=True, name=nm)

        self.plotter.render()

