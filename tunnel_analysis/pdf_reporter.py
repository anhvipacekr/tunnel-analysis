"""
pdf_reporter.py - Automated PDF report generation for tunnel analysis.
Style: Engineering report similar to Leica Cyclone 3DR output.
"""
from .common import *
from .models import PipelineContext, SectionGeometry
from pathlib import Path
from datetime import datetime
import io


class TunnelPDFReporter:
    """Generate professional PDF inspection report."""

    THRESHOLDS = {
        "crown_settlement_mm":    {"caution": 10.0, "critical": 25.0},
        "lateral_convergence_mm": {"caution": 15.0, "critical": 30.0},
        "ovality_mean_pct":       {"caution":  0.5, "critical":  1.0},
        "eccentricity_mean_mm":   {"caution": 10.0, "critical": 25.0},
    }

    # colours
    C_BLUE   = (0.059, 0.298, 0.506)
    C_RED    = (0.863, 0.082, 0.082)
    C_YELLOW = (0.851, 0.467, 0.024)
    C_GREEN  = (0.094, 0.502, 0.094)
    C_LGRAY  = (0.941, 0.945, 0.953)
    C_DGRAY  = (0.278, 0.337, 0.416)
    C_WHITE  = (1.0, 1.0, 1.0)
    C_BLACK  = (0.0, 0.0, 0.0)

    def export_pdf(self, context: PipelineContext, out_path: str,
                   project_name: str = "Tunnel Analysis",
                   engineer: str = "CBNU Smart Structure Lab",
                   location: str = "Osong Test Line") -> str:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.utils import ImageReader
        except ImportError:
            raise RuntimeError("reportlab required: pip install reportlab")

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        W, H = A4  # 595 x 842 pt

        c = rl_canvas.Canvas(str(path), pagesize=A4)
        c.setTitle(f"Tunnel Inspection Report - {project_name}")
        c.setAuthor(engineer)

        # ── Page 1: Cover ──────────────────────────────────────────────────
        self._draw_cover(c, W, H, project_name, engineer, location, context)
        c.showPage()

        # ── Page 2: Summary table ──────────────────────────────────────────
        self._draw_summary(c, W, H, context, project_name)
        c.showPage()

        # ── Page 3+: Section plots (up to 12 sections per page) ───────────
        if context.sections:
            self._draw_section_plots(c, W, H, context)

        # ---- Section data table ----
        if context.sections:
            self._draw_section_table(c, W, H, context, project_name)
            c.showPage()

        # ── Last page: Warnings ────────────────────────────────────────────
        self._draw_warnings(c, W, H, context, project_name)
        c.showPage()

        c.save()
        return str(path)

    # ---------------------------------------------------------------- Cover --
    def _draw_cover(self, c, W, H, project_name, engineer, location, context):
        from reportlab.lib.units import mm
        # header bar
        c.setFillColorRGB(*self.C_BLUE)
        c.rect(0, H - 80, W, 80, fill=1, stroke=0)
        # title
        c.setFillColorRGB(*self.C_WHITE)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(30, H - 42, "TUNNEL STRUCTURAL HEALTH MONITORING")
        c.setFont("Helvetica", 13)
        c.drawString(30, H - 62, "LiDAR Point Cloud Analysis Report")
        # sub bar
        c.setFillColorRGB(*self.C_DGRAY)
        c.rect(0, H - 100, W, 20, fill=1, stroke=0)
        c.setFillColorRGB(*self.C_WHITE)
        c.setFont("Helvetica", 9)
        c.drawString(30, H - 94, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   Software: Tunnel Analysis v4.0 - CBNU Smart Structure Lab")

        # info box
        c.setFillColorRGB(*self.C_LGRAY)
        c.roundRect(30, H - 320, W - 60, 200, 8, fill=1, stroke=0)
        c.setFillColorRGB(*self.C_BLACK)
        rows = [
            ("Project",   project_name),
            ("Location",  location),
            ("Engineer",  engineer),
            ("Date",      datetime.now().strftime("%Y-%m-%d")),
            ("Profile",   context.tunnel_profile),
        ]
        scan = context.active_scan
        if scan:
            rows.append(("Scan file", str(Path(scan.path).name) if scan.path else "N/A"))
            rows.append(("Points",    f"{len(scan.points):,}"))
        rows.append(("Sections", str(len(context.sections))))
        y = H - 145
        for label, val in rows:
            c.setFont("Helvetica-Bold", 10); c.drawString(50, y, f"{label}:")
            c.setFont("Helvetica", 10);      c.drawString(180, y, str(val))
            y -= 18

        # CBNU logo text
        c.setFillColorRGB(*self.C_BLUE)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(W / 2, H - 380, "CBNU Smart Structure Lab")
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(*self.C_DGRAY)
        c.drawCentredString(W / 2, H - 398, "Chungbuk National University · Structural Health Monitoring")

        # footer
        self._draw_footer(c, W, 1, project_name)

    # --------------------------------------------------------- Summary table --
    def _draw_summary(self, c, W, H, context, project_name):
        from reportlab.lib.units import mm
        self._draw_page_header(c, W, H, "PARAMETER SUMMARY", project_name)
        params = context.parameters
        y = H - 110

        # Global parameters
        if params:
            c.setFont("Helvetica-Bold", 11)
            c.setFillColorRGB(*self.C_BLUE)
            c.drawString(30, y, "Global Parameters"); y -= 6
            c.setStrokeColorRGB(*self.C_BLUE)
            c.line(30, y, W - 30, y); y -= 14
            col_w = [180, 100, 80, 80, 80]
            headers = ["Parameter", "Value", "Unit", "Threshold", "Status"]
            self._draw_table_row(c, 30, y, col_w, headers, header=True); y -= 18
            param_defs = [
                ("crown_settlement_mm",    "Crown Settlement δv",    "mm"),
                ("lateral_convergence_mm", "Horizontal Convergence δh", "mm"),
                ("ovality_mean_pct",       "Ovality ε (mean)",       "%"),
                ("eccentricity_mean_mm",   "Eccentricity e (mean)",  "mm"),
            ]
            for key, label, unit in param_defs:
                val = params.get(key, float("nan"))
                if not isinstance(val, (int, float)): continue
                thr = self.THRESHOLDS.get(key, {})
                status, color = self._status_color(key, val)
                thr_str = f"C:{thr.get('caution','?')} / R:{thr.get('critical','?')}" if thr else "-"
                self._draw_table_row(c, 30, y, col_w,
                    [label, f"{val:.3f}", unit, thr_str, status],
                    status_color=color); y -= 16

        # Section statistics
        if context.sections:
            y -= 10
            c.setFont("Helvetica-Bold", 11)
            c.setFillColorRGB(*self.C_BLUE)
            c.drawString(30, y, "Section Statistics"); y -= 6
            c.line(30, y, W - 30, y); y -= 14
            col_w2 = [160, 70, 70, 70, 70, 80]
            headers2 = ["Parameter", "Min", "Max", "Mean", "Unit", "Status"]
            self._draw_table_row(c, 30, y, col_w2, headers2, header=True); y -= 18
            sec_params = [
                ("H1",          "Clear Height H1",    "m"),
                ("W1",          "Clear Width W1",     "m"),
                ("ovality",     "Ovality ε",          "%"),
                ("eccentricity","Eccentricity e",     "mm"),
                ("radius_fit",  "Fitted Radius R",    "m"),
            ]
            for attr, label, unit in sec_params:
                vals = [getattr(s, attr) for s in context.sections
                        if np.isfinite(getattr(s, attr, float("nan")))]
                if not vals: continue
                arr = np.array(vals)
                self._draw_table_row(c, 30, y, col_w2, [
                    label,
                    f"{arr.min():.3f}", f"{arr.max():.3f}", f"{arr.mean():.3f}",
                    unit, "-"
                ]); y -= 16

        self._draw_footer(c, W, 2, project_name)

    # ---------------------------------------------------- Section plots page --
    def _draw_section_plots(self, c, W, H, context):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        sections = context.sections
        # settlement/convergence chart
        chainages = [s.chainage for s in sections]
        settlements = [s.H1 * 1e3 if np.isfinite(s.H1) else np.nan for s in sections]
        convergences = [s.W1 * 1e3 if np.isfinite(s.W1) else np.nan for s in sections]
        ovalities = [s.ovality if np.isfinite(s.ovality) else np.nan for s in sections]

        fig, axes = plt.subplots(3, 1, figsize=(7.5, 8), facecolor="white")
        fig.suptitle("Deformation Trend Along Tunnel Axis", fontsize=11, fontweight="bold", color="#0F172A")

        for ax, vals, label, color, thr_c, thr_r in [
            (axes[0], settlements,  "Clear Height H1 (mm)", "#1D4ED8", 10.0, 25.0),
            (axes[1], convergences, "Clear Width W1 (mm)",  "#047857", 15.0, 30.0),
            (axes[2], ovalities,    "Ovality ε (%)",        "#C2410C",  0.5,  1.0),
        ]:
            ax.plot(chainages, vals, color=color, lw=1.8, marker="o", ms=3, zorder=3)
            ax.axhline(thr_c, color="#D97706", lw=1.0, ls="--", alpha=0.8, label=f"Caution {thr_c}")
            ax.axhline(thr_r, color="#DC2626", lw=1.0, ls="--", alpha=0.8, label=f"Critical {thr_r}")
            ax.set_ylabel(label, fontsize=8)
            ax.grid(True, color="#E2E8F0", lw=0.5)
            ax.set_facecolor("white")
            ax.legend(fontsize=7, loc="upper right")
            for spine in ax.spines.values(): spine.set_color("#CBD5E1")
        axes[2].set_xlabel("Chainage (m)", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        from reportlab.lib.utils import ImageReader
        self._draw_page_header(c, W, H, "DEFORMATION CHARTS", "")
        img = ImageReader(buf)
        c.drawImage(img, 30, 80, width=W - 60, height=H - 160, preserveAspectRatio=True)
        self._draw_footer(c, W, 3, "")
        c.showPage()

    # ------------------------------------------------------- Warnings page --

    def _draw_section_table(self, c, W, H, context, project_name):
        """Per-section parameters table page."""
        import numpy as _np
        self._draw_page_header(c, W, H, "SECTION DATA TABLE", project_name)
        y = H - 110
        col_w = [50, 50, 50, 50, 55, 55, 50, 45]
        headers = ["Ch(m)","H1(m)","W1(m)","Oval(%)","Ecc(mm)","R_fit(m)","Clr(m)","Status"]
        self._draw_table_row(c, 15, y, col_w, headers, header=True); y -= 16
        for sec in context.sections:
            if y < 60:
                self._draw_footer(c, W, 3, project_name)
                c.showPage()
                self._draw_page_header(c, W, H, "SECTION DATA TABLE (cont.)", project_name)
                y = H - 110
                self._draw_table_row(c, 15, y, col_w, headers, header=True); y -= 16
            clr = f"{sec.min_clearance_dist:.3f}" if _np.isfinite(sec.min_clearance_dist) else "-"
            status = "VIOL" if sec.clearance_violation else "OK"
            sc = self.C_RED if sec.clearance_violation else None
            vals = [
                f"{sec.chainage:.2f}",
                f"{sec.H1:.3f}" if _np.isfinite(sec.H1) else "-",
                f"{sec.W1:.3f}" if _np.isfinite(sec.W1) else "-",
                f"{sec.ovality:.2f}" if _np.isfinite(sec.ovality) else "-",
                f"{sec.eccentricity:.1f}" if _np.isfinite(sec.eccentricity) else "-",
                f"{sec.radius_fit:.3f}" if _np.isfinite(sec.radius_fit) else "-",
                clr, status,
            ]
            self._draw_table_row(c, 15, y, col_w, vals, status_color=sc)
            y -= 14
        self._draw_footer(c, W, 3, project_name)

    def _draw_warnings(self, c, W, H, context, project_name):
        self._draw_page_header(c, W, H, "STRUCTURAL WARNINGS", project_name)
        y = H - 110
        warnings = []
        for sec in context.sections:
            ch = sec.chainage
            checks = [
                ("crown_settlement_mm",    sec.H1 * 1e3 if np.isfinite(sec.H1) else float("nan")),
                ("lateral_convergence_mm", sec.W1 * 1e3 if np.isfinite(sec.W1) else float("nan")),
                ("ovality_mean_pct",       sec.ovality   if np.isfinite(sec.ovality) else float("nan")),
                ("eccentricity_mean_mm",   sec.eccentricity if np.isfinite(sec.eccentricity) else float("nan")),
            ]
            for key, val in checks:
                if not np.isfinite(val): continue
                thr = self.THRESHOLDS.get(key, {})
                if val >= thr.get("critical", float("inf")):
                    warnings.append((ch, key, val, "CRITICAL", "#DC2626"))
                elif val >= thr.get("caution", float("inf")):
                    warnings.append((ch, key, val, "CAUTION", "#D97706"))
            if sec.clearance_violation:
                warnings.append((ch, "clearance_violation", sec.min_clearance_dist, "CRITICAL", "#DC2626"))

        if not warnings:
            c.setFillColorRGB(*self.C_GREEN)
            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(W / 2, H / 2, "✓  No structural warnings detected.")
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(*self.C_DGRAY)
            c.drawCentredString(W / 2, H / 2 - 20, "All parameters within acceptable limits.")
        else:
            col_w = [70, 160, 70, 70, 120]
            headers = ["Chainage", "Parameter", "Value", "Status", "Action"]
            self._draw_table_row(c, 30, y, col_w, headers, header=True); y -= 18
            for ch, key, val, status, color in warnings[:30]:
                action = "Immediate inspection" if status == "CRITICAL" else "Schedule within 30d"
                r, g, b = (0.863, 0.082, 0.082) if status == "CRITICAL" else (0.851, 0.467, 0.024)
                self._draw_table_row(c, 30, y, col_w,
                    [f"{ch:.2f}m", key, f"{val:.3f}", status, action],
                    status_color=(r, g, b)); y -= 16
                if y < 60: break

        self._draw_footer(c, W, 4, project_name)

    # ─────────────────────────────────────── helpers ──────────────────────────
    def _draw_page_header(self, c, W, H, title, project_name):
        c.setFillColorRGB(*self.C_BLUE)
        c.rect(0, H - 50, W, 50, fill=1, stroke=0)
        c.setFillColorRGB(*self.C_WHITE)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, H - 32, title)
        c.setFont("Helvetica", 9)
        c.drawRightString(W - 30, H - 32, project_name)
        c.setFont("Helvetica", 8)
        c.drawRightString(W - 30, H - 44, datetime.now().strftime("%Y-%m-%d"))

    def _draw_footer(self, c, W, page_num, project_name):
        c.setFillColorRGB(*self.C_LGRAY)
        c.rect(0, 0, W, 30, fill=1, stroke=0)
        c.setFillColorRGB(*self.C_DGRAY)
        c.setFont("Helvetica", 8)
        c.drawString(30, 10, "CBNU Smart Structure Lab  |  Tunnel Analysis v4.0")
        c.drawCentredString(W / 2, 10, project_name)
        c.drawRightString(W - 30, 10, f"Page {page_num}")

    def _draw_table_row(self, c, x, y, col_widths, values, header=False, status_color=None):
        row_h = 14
        if header:
            c.setFillColorRGB(*self.C_BLUE)
            c.rect(x, y - 2, sum(col_widths), row_h, fill=1, stroke=0)
            c.setFillColorRGB(*self.C_WHITE)
            c.setFont("Helvetica-Bold", 8)
        else:
            if status_color:
                r, g, b = status_color
                c.setFillColorRGB(r, g, b, 0.15)
                c.rect(x, y - 2, sum(col_widths), row_h, fill=1, stroke=0)
            c.setFillColorRGB(*self.C_BLACK)
            c.setFont("Helvetica", 8)
        cx = x
        for i, (val, w) in enumerate(zip(values, col_widths)):
            if header:
                c.setFillColorRGB(*self.C_WHITE)
            elif status_color and i == len(values) - 2:
                r, g, b = status_color
                c.setFillColorRGB(r, g, b)
            else:
                c.setFillColorRGB(*self.C_BLACK)
            c.drawString(cx + 3, y + 2, str(val)[:22])
            cx += w
        c.setStrokeColorRGB(*self.C_LGRAY)
        c.line(x, y - 2, x + sum(col_widths), y - 2)

    def _status_color(self, key, val):
        thr = self.THRESHOLDS.get(key, {})
        if not thr or not np.isfinite(val):
            return "N/A", self.C_DGRAY
        if val >= thr.get("critical", float("inf")):
            return "CRITICAL", self.C_RED
        if val >= thr.get("caution", float("inf")):
            return "CAUTION", self.C_YELLOW
        return "OK", self.C_GREEN
