from ..common import *
from ..models import SectionGeometry

class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title: str, step: int, tag: str, parent=None):
        super().__init__(parent)
        self._btn = QtWidgets.QToolButton()
        self._btn.setCheckable(True); self._btn.setChecked(False)
        self._btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(QtCore.Qt.RightArrow)
        self._btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._btn.setMinimumHeight(44); self._btn.setObjectName("SectionToggle")
        self._btn.setText(f"  Step {step}: {title}  [{tag}]")
        self._btn.toggled.connect(self._toggle)
        self._body = QtWidgets.QWidget(); self._body.setObjectName("SectionContent")
        self._blay = QtWidgets.QVBoxLayout(self._body)
        self._blay.setContentsMargins(12,4,4,8); self._blay.setSpacing(4)
        self._body.setVisible(False)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._btn); root.addWidget(self._body)

    def _toggle(self, checked: bool) -> None:
        self._btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._body.setVisible(checked)

    def add_sub_button(self, label: str, slot: Callable, tooltip: str = "") -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(f"  - {label}")
        b.setObjectName("SubButton"); b.setMinimumHeight(32)
        b.setCursor(QtCore.Qt.PointingHandCursor); b.clicked.connect(slot)
        if tooltip:
            b.setToolTip(tooltip)
            b.setToolTipDuration(4000)
        self._blay.addWidget(b); return b


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class MatplotlibSectionWidget(QtWidgets.QWidget):

    section_changed = QtCore.Signal(int)  # emits current index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sections: List[SectionGeometry] = []
        self._idx: int = 0
        self._profile: str = "Circle"
        self._vl_box_w  = VL_BOX_W
        self._vl_box_h  = VL_BOX_H
        self._vl_cir_r  = VL_CIR_R

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)

        # Navigation bar
        nav_frame = QtWidgets.QFrame()
        nav_frame.setStyleSheet("QFrame{background:#0F4C81;border-radius:6px;padding:2px;}")
        nav_frame.setMinimumHeight(44)
        nav = QtWidgets.QHBoxLayout(nav_frame)
        nav.setContentsMargins(6, 4, 6, 4); nav.setSpacing(6)
        btn_style = ("QPushButton{background:#1D4ED8;color:white;border-radius:5px;"
            "padding:4px 10px;font-weight:700;font-size:9.5pt;border:none;min-width:60px;}"
            "QPushButton:hover{background:#2563EB;}")
        expand_style = ("QPushButton{background:#047857;color:white;border-radius:5px;"
            "padding:4px 10px;font-weight:700;font-size:9.5pt;border:none;min-width:55px;}"
            "QPushButton:hover{background:#065F46;}")
        self._btn_prev = QtWidgets.QPushButton("\u25C0 Prev")
        self._btn_next = QtWidgets.QPushButton("Next \u25B6")
        self._btn_reset = QtWidgets.QPushButton("\u27F3 Zoom")
        self._btn_prev.setStyleSheet(btn_style)
        self._btn_next.setStyleSheet(btn_style)
        self._btn_reset.setStyleSheet(expand_style)
        self._btn_prev.setMinimumWidth(80)
        self._btn_next.setMinimumWidth(80)
        self._btn_reset.setMinimumWidth(110)
        self._btn_reset.setToolTip("Reset zoom (R)")
        self._btn_info = QtWidgets.QPushButton("\u24d8 Info")
        self._btn_info.setStyleSheet(expand_style)
        self._btn_info.setMinimumWidth(55)
        self._btn_info.setToolTip("Show section parameters")
        self._current_sg = None
        self._lbl_ch = QtWidgets.QLabel("Ch: --")
        self._lbl_ch.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_ch.setStyleSheet("color:white;font-weight:bold;font-size:10pt;background:transparent;min-width:100px;")
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next.clicked.connect(self._next)
        self._lbl_ch.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_ch.setStyleSheet("color:white;font-weight:bold;font-size:10pt;background:transparent;min-width:100px;")
        self._btn_next.clicked.connect(self._next)
        self._btn_reset.clicked.connect(self._reset_zoom)
        self._btn_info.clicked.connect(self._show_info_dialog)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._lbl_ch, 1)
        nav.addWidget(self._btn_next)
        nav.addWidget(self._btn_reset)
        nav.addWidget(self._btn_info)
        lay.addWidget(nav_frame)

        # Section slider
        slider_frame = QtWidgets.QFrame()
        slider_frame.setStyleSheet("QFrame{background:#F1F5F9;border-bottom:1px solid #E2E8F0;padding:2px;}")
        slider_lay = QtWidgets.QHBoxLayout(slider_frame)
        slider_lay.setContentsMargins(8, 2, 8, 2); slider_lay.setSpacing(6)
        lbl_slider = QtWidgets.QLabel("Ch:")
        lbl_slider.setStyleSheet("color:#475569;font-size:8.5pt;")
        self._slider_ch = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider_ch.setRange(0, 0)
        self._slider_ch.setValue(0)
        self._slider_ch.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._slider_ch.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#CBD5E1;border-radius:2px;}"
            "QSlider::handle:horizontal{width:14px;height:14px;background:#0F4C81;"
            "border-radius:7px;margin:-5px 0;}"
            "QSlider::handle:horizontal:hover{background:#1D4ED8;}")
        self._slider_ch.valueChanged.connect(self._on_slider_changed)
        self._lbl_slider_val = QtWidgets.QLabel("--")
        self._lbl_slider_val.setStyleSheet("color:#0F4C81;font-size:8.5pt;font-weight:600;min-width:60px;")
        slider_lay.addWidget(lbl_slider)
        slider_lay.addWidget(self._slider_ch, 1)
        slider_lay.addWidget(self._lbl_slider_val)
        lay.addWidget(slider_frame)

        if _MPL_OK:
            self._fig = Figure(figsize=(7.5, 6.5), facecolor=_BG)
            self._ax  = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
            lay.addWidget(self._canvas, 1)
            # Matplotlib navigation toolbar (zoom/pan)
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            self._toolbar.setStyleSheet(
                "QToolBar{background:#1E3A5F;border-top:2px solid #0F4C81;spacing:2px;padding:2px 4px;}"
                "QToolButton{background:#2D5A8E;border:1px solid #3B7DD8;border-radius:4px;"
                "padding:4px 6px;margin:1px;color:white;font-size:9pt;min-width:26px;min-height:22px;}"
                "QToolButton:hover{background:#3B7DD8;border-color:#60A5FA;}"
                "QToolButton:checked{background:#1D4ED8;border-color:#93C5FD;}")
            self._toolbar.setIconSize(QtCore.QSize(16, 16))
            lay.addWidget(QtWidgets.QLabel("Matplotlib is required for 2D cross-section plotting."))
        # Epoch overlay + animation controls
        ctrl = QtWidgets.QHBoxLayout(); ctrl.setSpacing(6)
        self._chk_overlay = QtWidgets.QCheckBox("Show T0 overlay")
        self._chk_overlay.setStyleSheet("color:#0F172A;font-size:9pt;font-weight:600;")
        self._chk_overlay.setToolTip("Overlay reference epoch T0 on current section")
        self._chk_overlay.toggled.connect(self._refresh)
        self._btn_anim = QtWidgets.QPushButton("▶ Animate")
        self._btn_anim.setStyleSheet(
            "QPushButton{background:#7C3AED;color:white;border-radius:5px;"
            "padding:4px 12px;font-weight:700;font-size:9pt;border:none;}"
            "QPushButton:hover{background:#6D28D9;}"
            "QPushButton:checked{background:#5B21B6;}")
        self._btn_anim.setCheckable(True)
        self._btn_anim.setToolTip("Animate deformation T0 -> Tn")
        self._btn_anim.toggled.connect(self._toggle_animation)
        self._anim_timer = QtCore.QTimer()
        self._anim_timer.setInterval(80)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_alpha = 0.0; self._anim_dir = 1
        self._ref_sections: List[SectionGeometry] = []
        ctrl.addWidget(self._chk_overlay); ctrl.addWidget(self._btn_anim); ctrl.addStretch()
        lay.addLayout(ctrl)
        self._info_label = QtWidgets.QLabel("Run Step 5.7 to display section parameters.")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            "color:#475569; font-family:monospace; font-size:9pt; "
            "padding:4px 6px; background:#F8FAFC; border-top:1px solid #E2E8F0;")
        self._info_label.setMinimumHeight(24)
        self._info_label.setMaximumHeight(48)
        lay.addWidget(self._info_label, 0)
        self._draw_empty()

    def set_ref_sections(self, sections) -> None:
        self._ref_sections = sections

    def _toggle_animation(self, checked: bool) -> None:
        if checked:
            self._anim_alpha = 0.0; self._anim_dir = 1
            self._btn_anim.setText("⏹ Stop"); self._anim_timer.start()
        else:
            self._anim_timer.stop(); self._btn_anim.setText("▶ Animate"); self._refresh()

    def _anim_step(self) -> None:
        self._anim_alpha += 0.05 * self._anim_dir
        if self._anim_alpha >= 1.0: self._anim_alpha = 1.0; self._anim_dir = -1
        elif self._anim_alpha <= 0.0: self._anim_alpha = 0.0; self._anim_dir = 1
        if not _MPL_OK or not self._sections: return
        sg_n = self._sections[self._idx]
        sg_0 = self._ref_sections[self._idx] if self._ref_sections and self._idx < len(self._ref_sections) else None
        self._draw_section(sg_n, ref_sg=sg_0, alpha=self._anim_alpha)

    def _reset_zoom(self) -> None:
        """Reset matplotlib view to fit the full section."""
        if not _MPL_OK: return
        self._ax.autoscale()
        self._ax.set_aspect("equal", adjustable="box")
        self._canvas.draw_idle()

    def _show_info_dialog(self) -> None:
        """Show section parameters in clean readable dialog."""
        sg = getattr(self, "_current_sg", None)
        if sg is None:
            if not self._sections: return
            sg = self._sections[self._idx]
        import numpy as _np
        dlg = QtWidgets.QDialog(self.parent() if self.parent() else self)
        dlg.setWindowTitle("Section Info  |  Ch. " + f"{sg.chainage:.3f} m")
        dlg.setMinimumWidth(380)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setSpacing(0); lay.setContentsMargins(0,0,0,0)
        hdr = QtWidgets.QFrame()
        hdr.setStyleSheet("QFrame{background:#0F4C81;padding:10px;}")
        hl = QtWidgets.QVBoxLayout(hdr); hl.setContentsMargins(16,10,16,10)
        t1 = QtWidgets.QLabel("Chainage: " + f"{sg.chainage:.3f} m")
        t1.setStyleSheet("color:white;font-size:14pt;font-weight:bold;background:transparent;")
        t2 = QtWidgets.QLabel("Profile: " + self._profile)
        t2.setStyleSheet("color:#CBD5E1;font-size:10pt;background:transparent;")
        hl.addWidget(t1); hl.addWidget(t2); lay.addWidget(hdr)
        grid = QtWidgets.QWidget()
        grid.setStyleSheet("QWidget{background:#F8FAFC;}")
        gl = QtWidgets.QGridLayout(grid)
        gl.setContentsMargins(16,12,16,12); gl.setSpacing(8)
        # (label, value, tooltip/description)
        rows = [
            ("H1  Clear Height",
             f"{sg.H1:.4f} m" if _np.isfinite(sg.H1) else "N/A",
             "Vertical clearance from floor to crown (total internal height)"),
            ("W1  Clear Width",
             f"{sg.W1:.4f} m" if _np.isfinite(sg.W1) else "N/A",
             "Horizontal clearance between left and right walls (internal width)"),
            ("H2  Crown Height",
             f"{sg.H2:.4f} m" if _np.isfinite(sg.H2) else "N/A",
             "Height from springline to crown (upper arch height)"),
            ("H3  Invert Height",
             f"{sg.H3:.4f} m" if _np.isfinite(sg.H3) else "N/A",
             "Height from floor to springline (lower section height)"),
            ("W2  Base Width",
             f"{sg.W2:.4f} m" if _np.isfinite(sg.W2) else "N/A",
             "Width at floor level (base width)"),
            ("R   Fitted Radius",
             f"{sg.radius_fit:.4f} m" if _np.isfinite(sg.radius_fit) else "N/A",
             "Best-fit circle radius to section points (design radius comparison)"),
            ("epsilon  Ovality",
             f"{sg.ovality:.4f} %" if _np.isfinite(sg.ovality) else "N/A",
             "Shape distortion: (a-b)/a x 100% where a=major, b=minor semi-axis. Caution>0.5%, Critical>1.0%"),
            ("e  Eccentricity",
             f"{sg.eccentricity:.2f} mm" if _np.isfinite(sg.eccentricity) else "N/A",
             "Distance between measured center and design center |C_meas - C_design|. Caution>10mm, Critical>25mm"),
            ("Clearance Min",
             f"{sg.min_clearance_dist:.4f} m" if _np.isfinite(sg.min_clearance_dist) else "N/A",
             "Minimum distance from tunnel surface to vehicle clearance envelope. Negative = violation"),
            ("Angle L  Wall-Floor",
             f"{sg.wall_angle_L:.2f} deg" if _np.isfinite(sg.wall_angle_L) else "N/A",
             "Angle between left wall and floor (90 deg = perfectly vertical wall)"),
            ("Angle R  Wall-Floor",
             f"{sg.wall_angle_R:.2f} deg" if _np.isfinite(sg.wall_angle_R) else "N/A",
             "Angle between right wall and floor (90 deg = perfectly vertical wall)"),
        ]
        for i,(lbl,val,tip) in enumerate(rows):
            # Label with tooltip
            l = QtWidgets.QLabel(lbl)
            l.setStyleSheet("color:#64748B;font-size:9.5pt;font-weight:600;")
            l.setToolTip(tip)
            # Value
            v = QtWidgets.QLabel(val)
            warn = False
            if "Ovality" in lbl and val != "N/A":
                try:
                    warn = float(val.replace("%","").strip()) >= 0.5
                except Exception: pass
            if "Eccentricity" in lbl and val != "N/A":
                try:
                    warn = float(val.replace("mm","").strip()) >= 10.0
                except Exception: pass
            if "Clearance" in lbl and val != "N/A":
                try:
                    warn = float(val.replace("m","").strip()) < 0
                except Exception: pass
            color = "#DC2626" if warn else "#0F172A"
            v.setStyleSheet(f"color:{color};font-size:10.5pt;font-weight:bold;font-family:monospace;")
            v.setToolTip(tip)
            # Info icon
            info_lbl = QtWidgets.QLabel("?")
            info_lbl.setStyleSheet(
                "color:#94A3B8;font-size:8pt;font-weight:bold;"
                "background:#F1F5F9;border-radius:8px;padding:1px 5px;")
            info_lbl.setToolTip(tip)
            gl.addWidget(l, i, 0)
            gl.addWidget(v, i, 1, QtCore.Qt.AlignRight)
            gl.addWidget(info_lbl, i, 2)
        lay.addWidget(grid)
        sf = QtWidgets.QFrame()
        bg = "#FEE2E2" if sg.clearance_violation else "#D1FAE5"
        bc = "#DC2626" if sg.clearance_violation else "#047857"
        sf.setStyleSheet(f"QFrame{{background:{bg};border-top:2px solid {bc};padding:8px;}}")
        sl = QtWidgets.QHBoxLayout(sf); sl.setContentsMargins(16,8,16,8)
        st = "CLEARANCE VIOLATION" if sg.clearance_violation else "OK - Within Limits"
        slbl = QtWidgets.QLabel(st)
        slbl.setStyleSheet(f"color:{bc};font-size:12pt;font-weight:bold;background:transparent;")
        sl.addWidget(slbl); lay.addWidget(sf)
        btn = QtWidgets.QPushButton("Close")
        btn.setStyleSheet("QPushButton{background:#0F4C81;color:white;border-radius:0;padding:10px;font-weight:700;font-size:10pt;border:none;}QPushButton:hover{background:#1D4ED8;}")
        btn.clicked.connect(dlg.accept); lay.addWidget(btn)
        dlg.exec()

    def _open_fullscreen(self) -> None:
        """Open current section in a resizable full-screen dialog."""
        if not self._sections or not _MPL_OK:
            return
        sg = self._sections[self._idx]
        if sg.pts_2d is None or len(sg.pts_2d) < 4:
            return
        dlg = _SectionFullscreenDialog(sg, self._profile,
                                        self._vl_box_w, self._vl_box_h,
                                        self._vl_cir_r, parent=self)
        dlg.exec()

    def _on_slider_changed(self, value: int) -> None:
        if not self._sections: return
        if value != self._idx:
            self._idx = value
            self._refresh()
        if hasattr(self, "_lbl_slider_val") and self._sections:
            sg = self._sections[self._idx]
            self._lbl_slider_val.setText(f"{sg.chainage:.2f}m")

    def _prev(self) -> None:
        if not self._sections: return
        self._idx = (self._idx - 1) % len(self._sections); self._refresh()

    def _next(self) -> None:
        if not self._sections: return
        self._idx = (self._idx + 1) % len(self._sections); self._refresh()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_Left:   self._prev()
        elif key == QtCore.Qt.Key_Right: self._next()
        elif key == QtCore.Qt.Key_Home:
            if self._sections: self._idx = 0; self._refresh()
        elif key == QtCore.Qt.Key_End:
            if self._sections: self._idx = len(self._sections)-1; self._refresh()
        elif key == QtCore.Qt.Key_PageUp:
            if self._sections:
                self._idx = max(0, self._idx - 5); self._refresh()
        elif key == QtCore.Qt.Key_PageDown:
            if self._sections:
                self._idx = min(len(self._sections)-1, self._idx + 5); self._refresh()
        elif key == QtCore.Qt.Key_R:
            self._reset_zoom()
        elif key == QtCore.Qt.Key_A:
            self._toggle_animation(not self._btn_anim.isChecked())
            self._btn_anim.setChecked(not self._btn_anim.isChecked())
        super().keyPressEvent(event)

    def set_sections(self, sections: List[SectionGeometry], profile: str, vl_box_w: float, vl_box_h: float, vl_cir_r: float) -> None:
        self._sections = sections; self._idx = 0; self._profile = profile
        self._vl_box_w = vl_box_w; self._vl_box_h = vl_box_h; self._vl_cir_r = vl_cir_r
        if hasattr(self, "_slider_ch") and sections:
            self._slider_ch.setRange(0, len(sections) - 1)
            self._slider_ch.setValue(0)
        self._refresh()

    def _draw_empty(self) -> None:
        if not _MPL_OK: return
        ax = self._ax; ax.clear(); ax.set_facecolor(_BG)
        ax.text(0.5, 0.5, "Run Step 5.7: Plot 2D Technical Section\nto display tunnel cross-sections and engineering dimensions.",
                ha="center", va="center", color=_FG, fontsize=11, transform=ax.transAxes)
        for s in ax.spines.values(): s.set_color(_GRID)
        ax.tick_params(colors=_FG); self._canvas.draw_idle()

    def _refresh(self) -> None:
        if not _MPL_OK or not self._sections: self._draw_empty(); return
        sg = self._sections[self._idx]
        self._lbl_ch.setText(f"Ch: {sg.chainage:.2f}m  [{self._idx + 1}/{len(self._sections)}]")
        if hasattr(self, "_slider_ch"):
            self._slider_ch.blockSignals(True)
            self._slider_ch.setValue(self._idx)
            self._slider_ch.blockSignals(False)
        if hasattr(self, "_lbl_slider_val"):
            self._lbl_slider_val.setText(f"{sg.chainage:.2f}m")
        self.section_changed.emit(self._idx)
        if sg.pts_2d is None or len(sg.pts_2d) < 4: self._draw_empty(); return
        ref_sg = None
        if hasattr(self, "_chk_overlay") and self._chk_overlay.isChecked():
            if self._ref_sections and self._idx < len(self._ref_sections):
                ref_sg = self._ref_sections[self._idx]
        self._draw_section(sg, ref_sg=ref_sg)

    def _draw_section(self, sg: SectionGeometry, ref_sg=None, alpha: float = 1.0) -> None:
        """Engineering Drawing style v2 - all 5 improvements."""
        from scipy.spatial import ConvexHull
        ax = self._ax
        ax.clear()

        # ── Background & grid ──────────────────────────────────────────────
        ax.set_facecolor("#FFFFFF")
        self._fig.patch.set_facecolor("#FFFFFF")
        ax.grid(True, color="#DDDDDD", lw=0.4, linestyle="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#888888"); spine.set_linewidth(0.8)
        ax.tick_params(colors="#333333", labelsize=7.5, direction="in", length=3)

        pts2d = sg.pts_2d
        if pts2d is None or len(pts2d) < 4:
            self._draw_empty(); return
        labels = sg.labels if sg.labels is not None and len(sg.labels) == len(pts2d) else np.zeros(len(pts2d), dtype=np.int32)
        finite = np.isfinite(pts2d[:, 0]) & np.isfinite(pts2d[:, 1])
        pts2d  = pts2d[finite]; labels = labels[finite]
        if len(pts2d) < 4:
            self._draw_empty(); return
        x = pts2d[:, 0]; z = pts2d[:, 1]

        # ── 3. Deviation colormap from best-fit circle ─────────────────────
        r_ref = sg.radius_fit if np.isfinite(sg.radius_fit) else float(np.median(np.hypot(x, z)))
        radii = np.hypot(x, z)
        dev_mm = (radii - r_ref) * 1e3  # mm
        dev_abs = np.abs(dev_mm)
        # green < 1mm, yellow 1-3mm, red > 3mm
        pt_colors = np.where(dev_abs < 1.0, "#16A34A",
                    np.where(dev_abs < 3.0, "#D97706", "#DC2626"))

        # ── 1. Wall/Crown/Floor colour override ────────────────────────────
        WALL_C   = "#1D4ED8"
        CROWN_C  = "#C2410C"
        FLOOR_C  = "#047857"
        struct_colors = np.where(labels == 1, CROWN_C,
                        np.where(labels == 2, FLOOR_C, WALL_C))
        # blend: use struct color for classified, deviation color for unclassified
        has_labels = np.any(labels != 0)
        final_colors = struct_colors if has_labels else pt_colors

        pt_alpha = max(0.3, min(0.75, 0.35 + 0.40 * alpha))
        ax.scatter(x, z, c=final_colors, s=2.2, alpha=pt_alpha,
                   linewidths=0, rasterized=True, zorder=2)
        # PDF params overlay
        # ── δv Crown settlement arrow ──────────────────────────────────────
        if hasattr(sg, "H1") and np.isfinite(sg.H1):
            crown_z = float(np.percentile(z, 97))
            spring_z = float(np.percentile(z, 50))
            dv_mm = (crown_z - spring_z) * 1e3
            # Arrow at crown pointing down
            ax.annotate("", xy=(0, crown_z), xytext=(0, crown_z + 0.3),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2.0),
                zorder=9)
            ax.text(0.05, crown_z + 0.35, f"δv={dv_mm:.0f}mm",
                color="#DC2626", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="#DC2626",
                          boxstyle="round,pad=0.2", alpha=0.9), zorder=10)

        # ── δh Convergence arrows ───────────────────────────────────────────
        if hasattr(sg, "W1") and np.isfinite(sg.W1):
            mid_z = float(np.percentile(z, 50))
            left_x  = float(np.percentile(x, 2))
            right_x = float(np.percentile(x, 98))
            dh_mm = (right_x - left_x) * 1e3
            # Left arrow pointing right
            ax.annotate("", xy=(left_x + 0.25, mid_z),
                xytext=(left_x, mid_z),
                arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=2.0), zorder=9)
            # Right arrow pointing left
            ax.annotate("", xy=(right_x - 0.25, mid_z),
                xytext=(right_x, mid_z),
                arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=2.0), zorder=9)
            ax.text(0.0, mid_z + 0.15, f"δh={dh_mm:.0f}mm",
                color="#1D4ED8", fontsize=8, fontweight="bold", ha="center",
                bbox=dict(facecolor="white", edgecolor="#1D4ED8",
                          boxstyle="round,pad=0.2", alpha=0.9), zorder=10)

        # ── e Eccentricity: measured center dot ─────────────────────────────
        cx_meas = float(np.mean(x)); cz_meas = float(np.mean(z))
        ax.plot(cx_meas, cz_meas, "D", color="#7C3AED", ms=7, zorder=10,
                label=f"C_meas")
        ax.plot(0, 0, "+", color="#64748B", ms=10, mew=2, zorder=10,
                label="C_design")
        ecc_mm = np.sqrt(cx_meas**2 + cz_meas**2) * 1e3
        if ecc_mm > 1.0:
            ax.plot([0, cx_meas], [0, cz_meas], "--",
                color="#7C3AED", lw=1.2, alpha=0.7, zorder=8)
            ax.text(cx_meas/2, cz_meas/2 + 0.1, f"e={ecc_mm:.0f}mm",
                color="#7C3AED", fontsize=7.5, ha="center",
                bbox=dict(facecolor="white", edgecolor="#7C3AED",
                          boxstyle="round,pad=0.15", alpha=0.85), zorder=10)

        # ── ε Ovality: show fitted ellipse ──────────────────────────────────
        if hasattr(sg, "ovality") and np.isfinite(sg.ovality) and sg.ovality > 0.1:
            a_semi = float(np.max(np.abs(x)))
            b_semi = float(np.max(np.abs(z)))
            if a_semi > 0.1 and b_semi > 0.1:
                theta_e = np.linspace(0, 2*np.pi, 100)
                ex = a_semi * np.cos(theta_e)
                ez = b_semi * np.sin(theta_e)
                ax.plot(ex, ez, "--", color="#D97706", lw=1.2, alpha=0.6,
                        zorder=3, label=f"ε={sg.ovality:.1f}%")

        # T0 reference overlay

        if ref_sg is not None and ref_sg.pts_2d is not None and len(ref_sg.pts_2d) >= 4:
            rx = ref_sg.pts_2d[:, 0]; rz = ref_sg.pts_2d[:, 1]
            ref_alpha = max(0.15, 0.55 * (1.0 - alpha))
            ax.scatter(rx, rz, c="#94A3B8", s=1.5, alpha=ref_alpha,
                       linewidths=0, rasterized=True, zorder=1, label="T0 reference")
            if self._profile == "Circle" and np.isfinite(ref_sg.radius_fit):
                ax.add_patch(plt.Circle((0.0, 0.0), ref_sg.radius_fit,
                    fill=False, edgecolor="#94A3B8", lw=1.2, ls=":", alpha=0.7,
                    zorder=2, label=f"T0 R={ref_sg.radius_fit:.3f}m"))
            if np.isfinite(sg.radius_fit) and np.isfinite(ref_sg.radius_fit):
                dr = (sg.radius_fit - ref_sg.radius_fit) * 1e3
                for ang_deg in [0, 90, 180, 270]:
                    rad = math.radians(ang_deg)
                    ax.annotate("", xy=(sg.radius_fit*math.cos(rad), sg.radius_fit*math.sin(rad)),
                        xytext=(ref_sg.radius_fit*math.cos(rad), ref_sg.radius_fit*math.sin(rad)),
                        arrowprops=dict(arrowstyle="->",
                            color="#DC2626" if dr < 0 else "#16A34A", lw=1.5), zorder=8)
                col = "#DC2626" if dr < 0 else "#16A34A"
                lbl = "convergence" if dr < 0 else "expansion"
                ax.text(0.01, 0.01, f"ΔR = {dr:+.1f} mm ({lbl})",
                    transform=ax.transAxes, fontsize=8, color=col, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="#CBD5E1",
                    boxstyle="round,pad=0.3", alpha=0.9), zorder=10)

        # ── 2. Convex hull outline ─────────────────────────────────────────
        if len(pts2d) >= 4:
            try:
                hull = ConvexHull(pts2d)
                hull_pts = pts2d[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        color="#475569", lw=1.0, ls="-", alpha=0.5,
                        zorder=3, label="Section outline")
            except Exception:
                pass

        # ── Best-fit circle ────────────────────────────────────────────────
        if self._profile == "Circle" and np.isfinite(sg.radius_fit):
            fit_c = plt.Circle((0.0, 0.0), sg.radius_fit,
                               fill=False, edgecolor="#2563EB", lw=1.6,
                               ls="--", alpha=0.9, zorder=4,
                               label=f"Best-fit R={sg.radius_fit:.3f}m")
            ax.add_patch(fit_c)

        # ── 4. Radial lines every 30° ──────────────────────────────────────
        r_max = float(np.percentile(radii, 97)) * 1.05
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            ax.plot([0, r_max * math.cos(rad)], [0, r_max * math.sin(rad)],
                    color="#CCCCCC", lw=0.5, ls=":", zorder=1, alpha=0.7)
            # label at 45/135/225/315
            if deg % 90 == 45:
                ax.text(r_max * 1.05 * math.cos(rad),
                        r_max * 1.05 * math.sin(rad),
                        f"{deg}°", color="#AAAAAA", fontsize=6.5,
                        ha="center", va="center")

        # ── Vehicle clearance envelope ─────────────────────────────────────
        vl_ok    = not sg.clearance_violation
        vl_color = "#888888" if vl_ok else "#DC2626"
        vl_lw    = 1.6 if vl_ok else 2.4
        vl_ls    = "-." if vl_ok else "-"
        vl_label = "Clearance limit" if vl_ok else "⚠ CLEARANCE VIOLATION"
        if self._profile == "Circle":
            ax.add_patch(plt.Circle((0.0, 0.0), self._vl_cir_r,
                fill=False, edgecolor=vl_color, lw=vl_lw, ls=vl_ls,
                alpha=0.95, zorder=5, label=vl_label))
        else:
            ax.add_patch(mpatches.Rectangle(
                (-self._vl_box_w, 0.0), 2*self._vl_box_w, self._vl_box_h,
                fill=False, edgecolor=vl_color, lw=vl_lw, ls=vl_ls,
                alpha=0.95, zorder=5, label=vl_label))
            if self._profile == "Box 2-cell":
                ax.plot([0.0, 0.0], [0.0, self._vl_box_h],
                        color=vl_color, lw=1.0, ls=":", zorder=5)

        # ── Centre cross ───────────────────────────────────────────────────
        cs = max(0.12, r_max * 0.04)
        ax.plot([-cs, cs], [0, 0], color="#333333", lw=1.0, zorder=6)
        ax.plot([0, 0], [-cs, cs], color="#333333", lw=1.0, zorder=6)

        # ── Dimension helpers ──────────────────────────────────────────────
        xmn = float(np.percentile(x, 1)); xmx = float(np.percentile(x, 99))
        zmn = float(np.percentile(z, 1)); zmx = float(np.percentile(z, 99))
        zmid = (zmn + zmx) / 2.0
        x_span = max(xmx - xmn, 1.0); z_span = max(zmx - zmn, 1.0)
        dim_gap = max(0.30, 0.07 * max(x_span, z_span))
        lbox = dict(facecolor="#FFFFFF", edgecolor="#AAAAAA",
                    boxstyle="round,pad=0.18", alpha=0.95)
        arr  = dict(arrowstyle="<->", color="#333333", lw=1.0)

        def _hdim(x0, x1, y, text):
            ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=arr, zorder=7)
            ax.plot([x0,x0],[y-dim_gap*0.07,y+dim_gap*0.07], color="#333333", lw=0.8, zorder=7)
            ax.plot([x1,x1],[y-dim_gap*0.07,y+dim_gap*0.07], color="#333333", lw=0.8, zorder=7)
            ax.text((x0+x1)/2.0, y+dim_gap*0.14, text, color="#111111",
                    fontsize=7.5, ha="center", va="bottom", bbox=lbox, zorder=8,
                    fontfamily="monospace")

        def _vdim(y0, y1, x_pos, text):
            ax.annotate("", xy=(x_pos, y1), xytext=(x_pos, y0), arrowprops=arr, zorder=7)
            ax.plot([x_pos-dim_gap*0.07,x_pos+dim_gap*0.07],[y0,y0], color="#333333", lw=0.8, zorder=7)
            ax.plot([x_pos-dim_gap*0.07,x_pos+dim_gap*0.07],[y1,y1], color="#333333", lw=0.8, zorder=7)
            ax.text(x_pos+dim_gap*0.16, (y0+y1)/2.0, text, color="#111111",
                    fontsize=7.5, ha="left", va="center", bbox=lbox, zorder=8,
                    fontfamily="monospace")

        dim_y_top    = zmx + dim_gap * 0.55
        dim_y_bottom = zmn - dim_gap * 0.55
        dim_x_right  = xmx + dim_gap * 0.55
        dim_x_left   = xmn - dim_gap * 0.55

        if np.isfinite(sg.W1): _hdim(xmn, xmx, dim_y_top,    f"W1={sg.W1:.3f}m")
        if np.isfinite(sg.W2): _hdim(xmn, xmx, dim_y_bottom, f"W2={sg.W2:.3f}m")
        if np.isfinite(sg.H1): _vdim(zmn, zmx, dim_x_right,  f"H1={sg.H1:.3f}m")
        if np.isfinite(sg.H2): _vdim(zmid, zmx, dim_x_left,  f"H2={sg.H2:.3f}m")
        if np.isfinite(sg.H3): _vdim(zmn, zmid, dim_x_left-dim_gap*0.55, f"H3={sg.H3:.3f}m")

        # ── Wall angle arcs ────────────────────────────────────────────────
        def _angle_arc(angle, cx, cz, side):
            if not np.isfinite(angle): return
            r_arc = min(x_span, z_span) * 0.12
            sa = 90.0; ext = angle if side == "left" else -angle
            ax.add_patch(mpatches.Arc((cx, cz), 2*r_arc, 2*r_arc, angle=0,
                theta1=min(sa,sa+ext), theta2=max(sa,sa+ext),
                color="#D97706", lw=1.4, zorder=7))
            m_rad = math.radians(sa + ext/2.0)
            ax.text(cx+r_arc*1.5*math.cos(m_rad), cz+r_arc*1.5*math.sin(m_rad),
                    f"{angle:.1f}°", color="#D97706", fontsize=7.5, fontweight="bold",
                    ha="center", va="center", bbox=lbox, zorder=8)

        _angle_arc(sg.wall_angle_L, xmn+x_span*0.10, zmid, "left")
        _angle_arc(sg.wall_angle_R, xmx-x_span*0.10, zmid, "right")

        # ── Clearance violation banner ─────────────────────────────────────
        if sg.clearance_violation:
            ax.text(0.5, 0.97, "⚠  CLEARANCE VIOLATION DETECTED",
                    transform=ax.transAxes, ha="center", va="top",
                    color="#DC2626", fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="#FFF1F1", edgecolor="#DC2626",
                              boxstyle="round,pad=0.4", alpha=0.95), zorder=10)

        # ── Limits & aspect ────────────────────────────────────────────────
        if self._profile == "Circle":
            vl_x0, vl_x1 = -self._vl_cir_r, self._vl_cir_r
            vl_z0, vl_z1 = -self._vl_cir_r, self._vl_cir_r
        else:
            vl_x0, vl_x1 = -self._vl_box_w, self._vl_box_w
            vl_z0, vl_z1 = 0.0, self._vl_box_h
        pad = max(0.5, 0.08 * max(x_span, z_span))
        plot_x0 = min(float(np.min(x)), vl_x0, dim_x_left  - dim_gap) - pad
        plot_x1 = max(float(np.max(x)), vl_x1, dim_x_right + dim_gap) + pad
        plot_z0 = min(float(np.min(z)), vl_z0, dim_y_bottom - dim_gap) - pad
        plot_z1 = max(float(np.max(z)), vl_z1, dim_y_top   + dim_gap) + pad
        cap = 18.0
        plot_x0 = max(plot_x0,-cap); plot_x1 = min(plot_x1, cap)
        plot_z0 = max(plot_z0,-cap); plot_z1 = min(plot_z1, cap)
        if plot_x1-plot_x0 < 1.0:
            mid=(plot_x0+plot_x1)/2.0; plot_x0,plot_x1=mid-0.5,mid+0.5
        if plot_z1-plot_z0 < 1.0:
            mid=(plot_z0+plot_z1)/2.0; plot_z0,plot_z1=mid-0.5,mid+0.5
        ax.set_xlim(plot_x0, plot_x1); ax.set_ylim(plot_z0, plot_z1)
        ax.set_aspect("equal", adjustable="box")

        # ── Axes labels & title ────────────────────────────────────────────
        ax.set_xlabel("X_2D  (N vector, m)", color="#333333", fontsize=8, labelpad=3)
        ax.set_ylabel("Z_2D  (B vector, m)", color="#333333", fontsize=8, labelpad=3)
        ax.set_title(
            f"TUNNEL CROSS-SECTION  |  Ch. {sg.chainage:.2f} m  |  {self._profile}",
            color="#0F172A", fontsize=9.5, fontweight="bold",
            fontfamily="monospace", pad=5)

        # ── Legend (structure labels) ──────────────────────────────────────
        legend_handles = []
        if has_labels:
            legend_handles += [
                mpatches.Patch(color=WALL_C,  label="Wall"),
                mpatches.Patch(color=CROWN_C, label="Crown"),
                mpatches.Patch(color=FLOOR_C, label="Floor"),
            ]
        else:
            legend_handles += [
                mpatches.Patch(color="#16A34A", label="Dev <1mm"),
                mpatches.Patch(color="#D97706", label="Dev 1-3mm"),
                mpatches.Patch(color="#DC2626", label="Dev >3mm"),
            ]
        ax.legend(handles=legend_handles, fontsize=7.5, facecolor="#FFFFFF",
                  edgecolor="#CCCCCC", labelcolor="#111111",
                  loc="lower right", framealpha=0.95, borderpad=0.6)

        # Title block moved to Info dialog button
        self._current_sg = sg
        # ── Info panel below plot ──────────────────────────────────────────
        if hasattr(self, "_info_label"):
            parts = [f"Ch:{sg.chainage:.2f}m"]
            if np.isfinite(sg.W1): parts.append(f"W1={sg.W1:.3f}m")
            if np.isfinite(sg.H1): parts.append(f"H1={sg.H1:.3f}m")
            if np.isfinite(sg.ovality): parts.append(f"ε={sg.ovality:.2f}%")
            if np.isfinite(sg.eccentricity): parts.append(f"e={sg.eccentricity:.1f}mm")
            if np.isfinite(sg.min_clearance_dist): parts.append(f"Clr={sg.min_clearance_dist:.3f}m")
            if self._profile=="Circle" and np.isfinite(sg.radius_fit): parts.append(f"R={sg.radius_fit:.3f}m")
            if sg.clearance_violation: parts.append("⚠ VIOLATION")
            color = "#DC2626" if sg.clearance_violation else "#0F172A"
            self._info_label.setStyleSheet(
                f"color:{color}; font-family:monospace; font-size:9pt; "
                f"padding:4px 8px; background:#F8FAFC; border-top:1px solid #CBD5E1;")
            self._info_label.setText("   |   ".join(parts))

        self._fig.tight_layout(pad=0.8)
        self._canvas.draw_idle()



# ------------------------------------------------------------------------------
# Full-screen section dialog
# ------------------------------------------------------------------------------

class _SectionFullscreenDialog(QtWidgets.QDialog):
    """Resizable full-screen dialog for 2D cross-section."""

    def __init__(self, sg, profile, vl_w, vl_h, vl_r, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Cross-Section  |  Ch. {sg.chainage:.3f} m  |  {profile}")
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowMaximizeButtonHint |
                            QtCore.Qt.WindowMinimizeButtonHint)
        self.resize(1000, 800)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        # Toolbar
        tb = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(
            f"Chainage: {sg.chainage:.3f} m  |  Profile: {profile}")
        lbl.setStyleSheet("font-weight:bold; font-size:11pt; color:#0F172A;")
        btn_save = QtWidgets.QPushButton("Save PNG")
        btn_save.setStyleSheet(
            "QPushButton{background:#047857;color:white;border-radius:5px;"
            "padding:5px 14px;font-weight:600;}"
            "QPushButton:hover{background:#065F46;}")
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setStyleSheet(
            "QPushButton{background:#64748B;color:white;border-radius:5px;"
            "padding:5px 14px;font-weight:600;}"
            "QPushButton:hover{background:#475569;}")
        tb.addWidget(lbl, 1)
        tb.addWidget(btn_save)
        tb.addWidget(btn_close)
        lay.addLayout(tb)

        # Large matplotlib canvas
        if _MPL_OK:
            self._fig = Figure(figsize=(12, 10), facecolor="white", dpi=100)
            self._ax  = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            lay.addWidget(self._canvas, 1)

            # Reuse MatplotlibSectionWidget draw logic
            tmp = MatplotlibSectionWidget.__new__(MatplotlibSectionWidget)
            tmp._fig = self._fig
            tmp._ax  = self._ax
            tmp._profile  = profile
            tmp._vl_box_w = vl_w
            tmp._vl_box_h = vl_h
            tmp._vl_cir_r = vl_r
            tmp._draw_section(sg)

            btn_save.clicked.connect(lambda: self._save_png(sg))
        else:
            lay.addWidget(QtWidgets.QLabel("Matplotlib not available."))

        btn_close.clicked.connect(self.accept)

    def _save_png(self, sg) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Section PNG",
            f"section_ch{sg.chainage:.2f}m.png",
            "PNG Images (*.png)")
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches="tight",
                              facecolor="white")
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{path}")


# ------------------------------------------------------------------------------
# PolarDeformationPlotWidget
# ------------------------------------------------------------------------------

class PolarDeformationPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._angles: Optional[np.ndarray] = None; self._dmap: Optional[np.ndarray] = None
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        if _MPL_OK:
            self._fig, self._ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(4, 4))
            self._fig.patch.set_facecolor(_BG); self._canvas = FigureCanvas(self._fig); lay.addWidget(self._canvas)
        else: lay.addWidget(QtWidgets.QLabel("Matplotlib missing."))

    def update_data(self, angles: np.ndarray, dmap: np.ndarray) -> None:
        if not _MPL_OK: return
        self._angles = angles; self._dmap = dmap; self._redraw()

    def _redraw(self) -> None:
        if not _MPL_OK or self._angles is None: return
        ax = self._ax; ax.clear()
        mean_dr = np.nanmean(self._dmap, axis=0); ang = self._angles
        for i in range(len(ang)-1):
            if np.isnan(mean_dr[i]): continue
            av = abs(float(mean_dr[i]))
            col = _GRN if av < 1.0 else (_YEL if av < 3.0 else _RED)
            ax.bar(ang[i], av, width=(ang[1] - ang[0]), color=col, alpha=0.85, edgecolor="none")
        ax.set_title("Polar radial deformation dr [mm]", color=_FG, fontsize=9, pad=8)
        ax.set_facecolor(_BG); ax.tick_params(colors=_FG, labelsize=7); ax.grid(True, color=_GRID, lw=0.6, alpha=0.75)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        self._fig.tight_layout(); self._canvas.draw_idle()


# ------------------------------------------------------------------------------
# LinePlotWidget
# ------------------------------------------------------------------------------

class LinePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: Optional[np.ndarray] = None; self.title = "Time-series"; self.setMinimumHeight(220)

    def set_values(self, values: Optional[np.ndarray], title: str = "") -> None:
        self.values = None if values is None else np.asarray(values, dtype=np.float64).ravel()
        self.title = title or "Time-series"; self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        rc = self.rect().adjusted(14, 14, -14, -14)
        p.fillRect(self.rect(), QtGui.QColor("#FFFFFF"))
        p.setPen(QtGui.QPen(QtGui.QColor("#CBD5E1"), 1)); p.drawRoundedRect(rc, 6, 6)
        p.setPen(QtGui.QColor("#111827")); p.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        p.drawText(rc.adjusted(10, 6, -10, -6), QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft, self.title)
        if self.values is None or len(self.values) < 2:
            p.setFont(QtGui.QFont("Segoe UI", 9)); p.setPen(QtGui.QColor("#64748B"))
            p.drawText(rc, QtCore.Qt.AlignCenter, "Run Step 6.2 to generate chart."); return
        pr = rc.adjusted(42, 42, -18, -34)
        p.setPen(QtGui.QPen(QtGui.QColor("#E2E8F0"), 1))
        for i in range(5):
            y = pr.top() + i * pr.height() / 4.0; p.drawLine(pr.left(), int(y), pr.right(), int(y))
        vals = self.values[np.isfinite(self.values)]
        if len(vals) < 2: return
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if math.isclose(vmin, vmax): vmax = vmin + 1.0
        pts = []
        for i, v in enumerate(self.values):
            x = pr.left() + i / max(1, len(self.values) - 1) * pr.width()
            y = pr.bottom() - (float(v) - vmin) / (vmax - vmin) * pr.height()
            pts.append(QtCore.QPointF(x, y))
        p.setPen(QtGui.QPen(QtGui.QColor("#2563EB"), 2))
        for a, b in zip(pts[:-1], pts[1:]): p.drawLine(a, b)
        p.setPen(QtGui.QColor("#475569")); p.setFont(QtGui.QFont("Segoe UI", 8))
        p.drawText(pr.left(), rc.bottom() - 8, f"min {vmin:.2f}mm")
        p.drawText(pr.right() - 110, rc.bottom() - 8, f"max {vmax:.2f}mm")


# ------------------------------------------------------------------------------
# Main Window & PySide6 UI
