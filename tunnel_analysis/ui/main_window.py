from ..common import *
from ..models import PointCloudBundle, PipelineContext
from ..io_layer import BaseLayer
from ..preprocessing import PreprocessingLayer
from ..registration import RegistrationLayer
from ..geometry import GeometricLayer
from ..segmentation import SegmentationLayer
from ..parameters import ParameterExtractionLayer
from ..timeseries import TimeSeriesLayer
from ..digital_twin import DigitalTwinAILayer
from ..worker import PipelineWorker
from ..exporter import TunnelExporter
from ..pdf_reporter import TunnelPDFReporter
from ..ifc_exporter import TunnelIFCExporter
from ..target_detector import TargetDetector, Target
from ..rag_ai import TunnelRAGAssistant
from .widgets import CollapsibleSection, MatplotlibSectionWidget, PolarDeformationPlotWidget, LinePlotWidget

class TunnelAnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tunnel Analysis v4.0 (r1) - CBNU Smart Structure Lab")
        self.resize(1720, 1000)

        self.context   = PipelineContext()
        self.base_mod  = BaseLayer()
        self.pre_mod   = PreprocessingLayer()
        self.reg_mod   = RegistrationLayer()
        self.geo_mod   = GeometricLayer()
        self.seg_mod   = SegmentationLayer()
        self.par_mod   = ParameterExtractionLayer()
        self.ts_mod    = TimeSeriesLayer()
        self.dt_mod    = DigitalTwinAILayer()
        self.exp_mod   = TunnelExporter()
        self.pdf_mod   = TunnelPDFReporter()
        self.ifc_mod   = TunnelIFCExporter()
        self.tgt_mod   = TargetDetector()
        self._targets: List[Target] = []   # all detected targets
        self._manual_pick_mode: bool = False   # manual picking active
        self.rag_mod   = TunnelRAGAssistant()
        # Initialize RAG in background
        import threading
        threading.Thread(target=self._init_rag, daemon=True).start()

        self.plotter:        Optional[QtInteractor]   = None
        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker:        Optional[PipelineWorker] = None
        self._all_sub_btns: List[QtWidgets.QPushButton] = []
        self._station_colors = [
            "#EF4444", "#3B82F6", "#10B981", "#F59E0B",
            "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
        ]
        self._noise_pts:    Optional[np.ndarray] = None  # current noise candidates
        self._kept_pts:     Optional[np.ndarray] = None  # current kept candidates
        self._noise_panel:  Optional[QtWidgets.QWidget] = None
        self._ai_tab_idx:   int = 5
        self._section_tab_idx: int = 3

        self._build_ui()
        self._apply_theme()
        self._init_pyvista()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        self.setCentralWidget(central)
        root.addWidget(self._build_sidebar())

        right = QtWidgets.QWidget(); rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(14, 12, 14, 12); rlay.setSpacing(10)
        root.addWidget(right, 1)

        self.header = QtWidgets.QFrame(); self.header.setObjectName("Header")
        hlay = QtWidgets.QVBoxLayout(self.header); hlay.setContentsMargins(14, 10, 14, 10)
        self.task_title = QtWidgets.QLabel("Tunnel Analysis v4.0")
        self.task_title.setObjectName("TaskTitle")
        self.task_desc  = QtWidgets.QLabel("Select a structural analysis workflow from the sidebar.")
        self.task_desc.setWordWrap(True); self.task_desc.setObjectName("TaskDescription")
        hlay.addWidget(self.task_title); hlay.addWidget(self.task_desc)
        rlay.addWidget(self.header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal); rlay.addWidget(splitter, 1)

        self.vp_frame  = QtWidgets.QFrame(); self.vp_frame.setObjectName("ViewportFrame")
        self.vp_layout = QtWidgets.QVBoxLayout(self.vp_frame)
        self.vp_layout.setContentsMargins(0, 0, 0, 0); self.vp_layout.setSpacing(0)
        splitter.addWidget(self.vp_frame)

        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.setMinimumWidth(460); splitter.addWidget(self.right_tabs)
        splitter.setSizes([1100, 620])

        self.results_text = QtWidgets.QPlainTextEdit(); self.results_text.setReadOnly(True)
        self.right_tabs.addTab(self.results_text, "Results Log")

        self.meta_table = QtWidgets.QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.meta_table.horizontalHeader().setStretchLastSection(True)
        self.right_tabs.addTab(self.meta_table, "Scan Database")

        # Station tree panel (Faro SCENE style)
        self._station_panel = QtWidgets.QWidget()
        st_lay = QtWidgets.QVBoxLayout(self._station_panel)
        st_lay.setContentsMargins(0, 0, 0, 0); st_lay.setSpacing(0)

        # Toolbar
        st_tb = QtWidgets.QFrame()
        st_tb.setStyleSheet("QFrame{background:#0F4C81;padding:4px;}")
        st_tb_lay = QtWidgets.QHBoxLayout(st_tb)
        st_tb_lay.setContentsMargins(8, 4, 8, 4); st_tb_lay.setSpacing(4)
        st_title = QtWidgets.QLabel("Structure")
        st_title.setStyleSheet("color:white;font-weight:bold;font-size:10pt;background:transparent;")
        btn_add_st = QtWidgets.QPushButton("+")
        btn_add_st.setToolTip("Add scan station")
        btn_add_st.setFixedSize(24, 24)
        btn_add_st.setStyleSheet(
            "QPushButton{background:#1D4ED8;color:white;border-radius:4px;font-weight:bold;border:none;}"
            "QPushButton:hover{background:#2563EB;}")
        btn_add_st.clicked.connect(self._slot_1_3_add_scan)
        st_tb_lay.addWidget(st_title, 1)
        st_tb_lay.addWidget(btn_add_st)
        st_lay.addWidget(st_tb)

        # Tree widget
        self._station_tree = QtWidgets.QTreeWidget()
        self._station_tree.setHeaderHidden(True)
        self._station_tree.setColumnCount(1)
        self._station_tree.setStyleSheet("""
            QTreeWidget {
                border: none; background: #FAFAFA;
                font-size: 9.5pt; font-family: 'Segoe UI';
            }
            QTreeWidget::item {
                padding: 4px 2px; border-bottom: 1px solid #F1F5F9;
                min-height: 28px;
            }
            QTreeWidget::item:selected {
                background: #DBEAFE; color: #1D4ED8;
            }
            QTreeWidget::item:hover {
                background: #EFF6FF;
            }
        """)
        self._station_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._station_tree.customContextMenuRequested.connect(self._station_context_menu)
        self._station_tree.currentItemChanged.connect(self._on_station_tree_changed)
        self._station_tree.itemChanged.connect(self._on_station_item_changed)
        st_lay.addWidget(self._station_tree, 1)

        # Bottom toolbar
        st_bot = QtWidgets.QFrame()
        st_bot.setStyleSheet("QFrame{background:#F1F5F9;border-top:1px solid #E2E8F0;padding:2px;}")
        st_bot_lay = QtWidgets.QHBoxLayout(st_bot)
        st_bot_lay.setContentsMargins(6, 2, 6, 2); st_bot_lay.setSpacing(4)
        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.setStyleSheet(
            "QPushButton{background:#FEE2E2;color:#DC2626;border:1px solid #FCA5A5;"
            "border-radius:4px;padding:3px 8px;font-weight:600;font-size:8.5pt;}"
            "QPushButton:hover{background:#FECACA;}")
        btn_clear.clicked.connect(self._clear_all_stations)
        st_bot_lay.addStretch()
        st_bot_lay.addWidget(btn_clear)
        st_lay.addWidget(st_bot)

        self.right_tabs.addTab(self._station_panel, "Stations")

        # Target Manager panel (Faro SCENE style)
        self._target_panel = QtWidgets.QWidget()
        tgt_lay = QtWidgets.QVBoxLayout(self._target_panel)
        tgt_lay.setContentsMargins(0, 0, 0, 0); tgt_lay.setSpacing(0)

        # Header toolbar
        tgt_tb = QtWidgets.QFrame()
        tgt_tb.setStyleSheet("QFrame{background:#065F46;padding:4px;}")
        tgt_tb_lay = QtWidgets.QHBoxLayout(tgt_tb)
        tgt_tb_lay.setContentsMargins(8, 4, 8, 4); tgt_tb_lay.setSpacing(4)
        tgt_title = QtWidgets.QLabel("Target Manager")
        tgt_title.setStyleSheet("color:white;font-weight:bold;font-size:10pt;background:transparent;")
        btn_detect = QtWidgets.QPushButton("Auto Detect")
        btn_detect.setStyleSheet(
            "QPushButton{background:#047857;color:white;border-radius:4px;"
            "padding:3px 10px;font-weight:700;border:none;font-size:9pt;}"
            "QPushButton:hover{background:#059669;}")
        btn_manual = QtWidgets.QPushButton("+ Manual")
        btn_manual.setStyleSheet(
            "QPushButton{background:#1D4ED8;color:white;border-radius:4px;"
            "padding:3px 10px;font-weight:700;border:none;font-size:9pt;}"
            "QPushButton:hover{background:#2563EB;}")
        btn_match = QtWidgets.QPushButton("Auto Match")
        btn_match.setStyleSheet(
            "QPushButton{background:#7C3AED;color:white;border-radius:4px;"
            "padding:3px 10px;font-weight:700;border:none;font-size:9pt;}"
            "QPushButton:hover{background:#6D28D9;}")
        btn_reg = QtWidgets.QPushButton("Register")
        btn_reg.setStyleSheet(
            "QPushButton{background:#DC2626;color:white;border-radius:4px;"
            "padding:3px 10px;font-weight:700;border:none;font-size:9pt;}"
            "QPushButton:hover{background:#B91C1C;}")
        btn_detect.clicked.connect(self._slot_target_detect)
        btn_manual.clicked.connect(self._slot_target_manual)
        btn_match.clicked.connect(self._slot_target_match)
        btn_reg.clicked.connect(self._slot_target_register)
        tgt_tb_lay.addWidget(tgt_title, 1)
        tgt_tb_lay.addWidget(btn_detect)
        tgt_tb_lay.addWidget(btn_manual)
        tgt_tb_lay.addWidget(btn_match)
        tgt_tb_lay.addWidget(btn_reg)
        tgt_lay.addWidget(tgt_tb)

        # Target table
        self._target_table = QtWidgets.QTableWidget(0, 7)
        self._target_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Scan", "X", "Y", "Z", "Conf"])
        self._target_table.horizontalHeader().setStretchLastSection(True)
        self._target_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._target_table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self._target_table.setStyleSheet(
            "QTableWidget{border:none;background:#FAFAFA;font-size:9pt;}"
            "QTableWidget::item{padding:3px 6px;}"
            "QTableWidget::item:selected{background:#DBEAFE;color:#1D4ED8;}"
            "QHeaderView::section{background:#065F46;color:white;padding:4px;"
            "font-weight:600;font-size:8.5pt;border:none;}")
        self._target_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._target_table.customContextMenuRequested.connect(self._target_context_menu)
        self._target_table.cellClicked.connect(self._on_target_selected)
        tgt_lay.addWidget(self._target_table, 1)

        # Status bar
        self._tgt_status = QtWidgets.QLabel("No targets detected.")
        self._tgt_status.setStyleSheet(
            "color:#065F46;font-size:8.5pt;padding:4px 8px;"
            "background:#ECFDF5;border-top:1px solid #A7F3D0;")
        tgt_lay.addWidget(self._tgt_status)
        self.right_tabs.addTab(self._target_panel, "Targets")
        self._station_visibility = {}  # idx -> bool

        self.ts_plot = LinePlotWidget()
        self.right_tabs.addTab(self.ts_plot, "Time-Series Plot")

        self.section_widget = MatplotlibSectionWidget()
        self.right_tabs.addTab(self.section_widget, "2D Cross-Section")

        self.polar_plot = PolarDeformationPlotWidget()
        self.right_tabs.addTab(self.polar_plot, "Polar Deformation")

        ai_panel = QtWidgets.QWidget(); ai_lay = QtWidgets.QVBoxLayout(ai_panel)
        ai_lay.setContentsMargins(8, 8, 8, 8); ai_lay.setSpacing(6)
        self.ai_prompt = QtWidgets.QPlainTextEdit()
        self.ai_prompt.setPlaceholderText("Enter a structural engineering question for the local AI assistant (Llama 3)...")
        self.ai_prompt.setMaximumHeight(90)
        self.ai_send = QtWidgets.QPushButton("Query AI Assistant")
        self.ai_send.clicked.connect(self._slot_7_2_query_ai)
        self.ai_resp = QtWidgets.QPlainTextEdit(); self.ai_resp.setReadOnly(True)
        ai_lay.addWidget(QtWidgets.QLabel("Engineering query:")); ai_lay.addWidget(self.ai_prompt)
        ai_lay.addWidget(self.ai_send)
        ai_lay.addWidget(QtWidgets.QLabel("AI analysis report:")); ai_lay.addWidget(self.ai_resp, 1)
        self.right_tabs.addTab(ai_panel, "AI Engineering Assistant")

        self.sb_pts  = QtWidgets.QLabel("Points: --")
        self.sb_rmse = QtWidgets.QLabel("RMSE: --")
        self.sb_msg  = QtWidgets.QLabel("Ready")
        self.sb_prog = QtWidgets.QProgressBar(); self.sb_prog.setRange(0, 100)
        self.statusBar().addWidget(self.sb_pts)
        self.statusBar().addWidget(self.sb_rmse)
        self.statusBar().addWidget(self.sb_msg, 1)
        self.statusBar().addPermanentWidget(self.sb_prog)

    def _build_sidebar(self) -> QtWidgets.QFrame:
        sb = QtWidgets.QFrame(); sb.setObjectName("Sidebar"); sb.setFixedWidth(375)
        out = QtWidgets.QVBoxLayout(sb); out.setContentsMargins(10, 14, 10, 14); out.setSpacing(6)

        t1 = QtWidgets.QLabel("TUNNEL ANALYSIS"); t1.setObjectName("ProductTitle")
        t2 = QtWidgets.QLabel("v4.0 r1 - CBNU Smart Structure Lab"); t2.setObjectName("LabSubtitle")
        out.addWidget(t1); out.addWidget(t2)
        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setObjectName("Separator"); out.addWidget(sep)

        pf_frame = QtWidgets.QGroupBox("Tunnel Profile Type")
        pf_frame.setStyleSheet("QGroupBox{color:#334155;border:1px solid #CBD5E1;border-radius:5px;margin-top:6px;padding:4px;}")
        pf_lay = QtWidgets.QHBoxLayout(pf_frame)
        self._profile_combo = QtWidgets.QComboBox()
        self._profile_combo.addItems(TUNNEL_PROFILES)
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)
        pf_lay.addWidget(self._profile_combo); out.addWidget(pf_frame)

        vl_frame = QtWidgets.QGroupBox("Vehicle Clearance Limit (m)")
        vl_frame.setStyleSheet("QGroupBox{color:#334155;border:1px solid #CBD5E1;border-radius:5px;margin-top:6px;padding:4px;}")
        vl_lay = QtWidgets.QFormLayout(vl_frame)
        self._sp_vl_w = QtWidgets.QDoubleSpinBox(); self._sp_vl_w.setValue(VL_BOX_W)
        self._sp_vl_h = QtWidgets.QDoubleSpinBox(); self._sp_vl_h.setValue(VL_BOX_H)
        self._sp_vl_r = QtWidgets.QDoubleSpinBox(); self._sp_vl_r.setValue(VL_CIR_R)
        vl_lay.addRow("Half clear width W:", self._sp_vl_w)
        vl_lay.addRow("Clear height H:", self._sp_vl_h)
        vl_lay.addRow("Circular clearance radius R:", self._sp_vl_r)
        out.addWidget(vl_frame)

        # ── Auto Pipeline button ──────────────────────────────────────
        auto_btn = QtWidgets.QPushButton("AUTO PIPELINE  (1-click full analysis)")
        auto_btn.setObjectName("AutoPipelineBtn")
        auto_btn.setMinimumHeight(42)
        auto_btn.setStyleSheet("""
            QPushButton#AutoPipelineBtn {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #0F4C81, stop:1 #1D4ED8);
                color: white; font-size: 11pt; font-weight: 800;
                border-radius: 8px; border: none; padding: 8px 12px;
                letter-spacing: 0.5px;
            }
            QPushButton#AutoPipelineBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1D4ED8, stop:1 #2563EB);
            }
            QPushButton#AutoPipelineBtn:disabled {
                background: #94A3B8; color: #E2E8F0;
            }
        """)
        auto_btn.clicked.connect(self._slot_auto_pipeline)
        out.addWidget(auto_btn)
        self._auto_btn = auto_btn
        reset_btn = QtWidgets.QPushButton("Reset Pipeline")
        reset_btn.setMinimumHeight(30)
        reset_btn.setStyleSheet(
            "QPushButton{background:#FEE2E2;color:#DC2626;border:1px solid #FCA5A5;"
            "border-radius:6px;padding:4px;font-weight:600;font-size:9pt;}"
            "QPushButton:hover{background:#FECACA;}")
        reset_btn.clicked.connect(self._slot_reset_pipeline)
        out.addWidget(reset_btn)

        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        out.addWidget(scroll, 1)
        sc = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(sc)
        sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(4); scroll.setWidget(sc)

        SECTIONS = [
            (1, "LiDAR data acquisition", "Base", [
                ("1.1  Import LAS / PLY data", self._slot_1_1_import),
                ("1.2  Initialize 3D viewport", self._slot_1_2_viewport),
                ("1.3  Add scan station (+)", self._slot_1_3_add_scan),
                ("1.4  Register & merge all stations", self._slot_1_4_merge),
                ("1.5  Rough alignment (manual)", self._slot_1_5_rough),
                ("1.6  Chain register & merge", self._slot_1_6_chain),
                ("1.7  Registration error heatmap", self._slot_1_7_reg_error),
            ]),
            (2, "Preprocessing and noise filtering", "Pre.", [
                ("2.1  Voxel downsampling", self._slot_2_1_voxel),
                ("2.2  Statistical outlier removal", self._slot_2_2_sor),
                ("2.3  Extract tunnel lining shell", self._slot_2_3_lining),
                ("2.4  Semantic noise removal (PDF 3.2)", self._slot_2_4_semantic),
            ]),
            (3, "Registration and synchronization", "Reg.", [
                ("3.1  Anchor translation", self._slot_3_1_anchor),
                ("3.2  Fine surface ICP", self._slot_3_2_icp),
                ("3.3  Calculate RMSE", self._slot_3_3_rmse),
            ]),
            (4, "Geometric coordinate system", "Geo.", [
                ("4.1  Extract PCA centerline", self._slot_4_1_centerline),
                ("4.2  Iterative centerline refinement", self._slot_4_2_iterative),
                ("4.3  Smooth B-Spline centerline", self._slot_4_3_bspline),
                ("4.3b B-Spline C2 centerline (PDF 3.4)", self._slot_4_3b_bspline),
                ("4.4  Generate gravity-aligned N-B sections", self._slot_4_4_frenet),
                ("4.5  Detect ring seams", self._slot_4_5_seams),
                ("4.5b Intensity ring seam detection (PDF 3.3)", self._slot_4_5b_intensity_seams),
            ]),
            (5, "Parameter extraction", "Param.", [
                ("5.1  Crown settlement dv", self._slot_5_1_settlement),
                ("5.2  Horizontal convergence dh", self._slot_5_2_convergence),
                ("5.3  3D deformation heatmap", self._slot_5_3_heatmap),
                ("5.3b Hausdorff heatmap T0→Tn (PDF 3.5)", self._slot_5_3b_hausdorff),
                ("5.4  Polar radial deformation dr", self._slot_5_4_polar),
                ("5.5  Ovality epsilon", self._slot_5_5_ovality),
                ("5.6  Section eccentricity e", self._slot_5_6_eccentricity),
                ("5.7  Plot 2D Technical Section", self._slot_5_7_sections),
                ("5.8  Clearance 3D violation map", self._slot_5_8_clearance_3d),
            ]),
            (6, "Time-series analysis", "T-S", [
                ("6.1  Load T0 and Tn epochs", self._slot_6_1_epochs),
                ("6.2  Plot deformation trend", self._slot_6_2_plot),
            ]),
            (7, "BIM and AI", "BIM/AI", [
                ("7.1  Export IFC package", self._slot_7_1_ifc),
                ("7.2  Query structural AI assistant", self._slot_7_2_query_ai),
            ]),
        ]
        for step, title_s, tag, buttons in SECTIONS:
            sec = CollapsibleSection(title_s, step, tag)
            for label, slot in buttons:
                btn = sec.add_sub_button(label, slot); self._all_sub_btns.append(btn)
            sl.addWidget(sec)

        sl.addStretch()
        self.pt_label   = QtWidgets.QLabel("Points: --")
        self.rmse_label = QtWidgets.QLabel("RMSE: --")
        out.addWidget(self.pt_label); out.addWidget(self.rmse_label)
        return sb

    def _on_profile_changed(self, text: str) -> None:
        self.context.tunnel_profile = text

    def _init_pyvista(self) -> None:
        while self.vp_layout.count():
            item = self.vp_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if pv is None: self._vp_msg("PyVista is not installed."); return
        try:
            self.plotter = QtInteractor(self.vp_frame); self.plotter.set_background("#F8FAFC")
            self.vp_layout.addWidget(self.plotter, 1); self.plotter.add_axes(color="#111827")
            self.plotter.show_bounds(color="#94A3B8", grid="front", location="outer", font_size=8)
            self.plotter.render()
        except Exception as exc: self.plotter = None; self._vp_msg(f"Failed to initialize PyVista: {exc}")

    def _vp_msg(self, msg: str) -> None:
        lbl = QtWidgets.QLabel(msg); lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setWordWrap(True); lbl.setObjectName("ViewportMessage")
        self.vp_layout.addWidget(lbl, 1)

    def _start_worker(self, key: str, cb: Callable[[], object]) -> None:
        if self.worker_thread is not None: self._log("A workflow task is already running."); return
        self._btns_enabled(False); self.sb_prog.setValue(10); self.sb_msg.setText(f"Running task: {key} ...")
        self.worker_thread = QtCore.QThread(self)
        self.worker = PipelineWorker(key, cb); self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished); self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker_thread.quit); self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater); self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker); self.worker_thread.start()

    @QtCore.Slot()
    def _clear_worker(self) -> None:
        self.worker_thread = None; self.worker = None; self._btns_enabled(True); self.sb_prog.setValue(0)

    def _btns_enabled(self, en: bool) -> None:
        for b in self._all_sub_btns: b.setEnabled(en)

    @QtCore.Slot(str, object)
    def _on_finished(self, key: str, result: object) -> None:
        self.sb_prog.setValue(100); self.sb_msg.setText(f"Task completed: {key}"); self._dispatch(key, result); self._check_auto_pipeline(key)

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
            self.ts_plot.set_values(series, "Deformation Trend Chart Across Chainage Line (mm)")
            self.right_tabs.setCurrentIndex(2)

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


class _RoughAlignDialog(QtWidgets.QDialog):
    def __init__(self, context, reg_mod, parent=None, plotter=None):
        super().__init__(parent)
        self.setWindowTitle("Rough Alignment")
        self.setMinimumWidth(480)
        self.context = context; self.reg_mod = reg_mod
        self.plotter = plotter; self.offset = [0.0,0.0,0.0]
        self.rotation = [0.0,0.0,0.0]; self.aligned_pts = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10); lay.setContentsMargins(16,16,16,16)
        hdr = QtWidgets.QLabel("Adjust scan station position before ICP")
        hdr.setStyleSheet("color:#0F172A;font-weight:bold;font-size:10pt;")
        lay.addWidget(hdr)
        st_lay = QtWidgets.QHBoxLayout()
        st_lay.addWidget(QtWidgets.QLabel("Active station:"))
        self._station_combo = QtWidgets.QComboBox()
        for i, sc in enumerate(self.context.scans):
            name = "Station " + str(i+1)
            if sc.path:
                import pathlib
                name += " - " + pathlib.Path(sc.path).name
            self._station_combo.addItem(name)
        self._station_combo.setCurrentIndex(len(self.context.scans)-1)
        st_lay.addWidget(self._station_combo, 1); lay.addLayout(st_lay)
        grp = QtWidgets.QGroupBox("Translation (m) & Rotation (deg)")
        grp.setStyleSheet("QGroupBox{font-weight:600;color:#0F4C81;border:1px solid #CBD5E1;border-radius:6px;margin-top:8px;padding:8px;}")
        form = QtWidgets.QFormLayout(grp)
        self._sliders = {}
        params = [("dx","dX (m)",-20,20,0),("dy","dY (m)",-20,20,0),("dz","dZ (m)",-20,20,0),
                  ("rx","Rot X",-180,180,0),("ry","Rot Y",-180,180,0),("rz","Rot Z",-180,180,0)]
        for key,label,mn,mx,val in params:
            row = QtWidgets.QHBoxLayout()
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(int(mn*100),int(mx*100)); slider.setValue(int(val*100))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(mn,mx); spin.setValue(val); spin.setFixedWidth(90)
            slider.valueChanged.connect(lambda v,s=spin: s.setValue(v/100.0))
            spin.valueChanged.connect(lambda v,sl=slider: sl.setValue(int(v*100)))
            spin.valueChanged.connect(self._on_param_changed)
            row.addWidget(slider,1); row.addWidget(spin)
            self._sliders[key] = (slider,spin)
            form.addRow(label, row)
        lay.addWidget(grp)
        self._rmse_lbl = QtWidgets.QLabel("RMSE: --")
        self._rmse_lbl.setStyleSheet("color:#0F4C81;font-weight:bold;font-size:10pt;padding:6px;background:#EFF6FF;border-radius:4px;")
        lay.addWidget(self._rmse_lbl)
        btn_lay = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_icp   = QtWidgets.QPushButton("Run ICP")
        btn_ok    = QtWidgets.QPushButton("Apply & Close")
        btn_cancel= QtWidgets.QPushButton("Cancel")
        for btn,color in [(btn_reset,"#64748B"),(btn_icp,"#7C3AED"),(btn_ok,"#047857"),(btn_cancel,"#DC2626")]:
            btn.setStyleSheet(f"QPushButton{{background:{color};color:white;border-radius:5px;padding:7px 16px;font-weight:700;border:none;}}")
        btn_reset.clicked.connect(self._reset); btn_icp.clicked.connect(self._run_icp)
        btn_ok.clicked.connect(self.accept); btn_cancel.clicked.connect(self.reject)
        btn_lay.addWidget(btn_reset); btn_lay.addWidget(btn_icp)
        btn_lay.addStretch(); btn_lay.addWidget(btn_ok); btn_lay.addWidget(btn_cancel)
        lay.addLayout(btn_lay)
        self._on_param_changed()

    def _get_params(self):
        return ([self._sliders[k][1].value() for k in ["dx","dy","dz"]],
                [self._sliders[k][1].value() for k in ["rx","ry","rz"]])

    def _on_param_changed(self):
        offset, rotation = self._get_params()
        self.offset = offset; self.rotation = rotation
        idx = self._station_combo.currentIndex()
        if idx < 0 or idx >= len(self.context.scans): return
        pts = validate_xyz(self.context.scans[idx].points)
        self.aligned_pts = self.reg_mod.apply_manual_transform(pts, tuple(offset), tuple(rotation))
        if self.aligned_pts is not None and len(self.context.scans) > 0:
            ref_idx = max(0, idx-1)
            ref = validate_xyz(self.context.scans[ref_idx].points)
            rmse = self.reg_mod._rmse(self.aligned_pts, ref)
            color = "#047857" if rmse < 2.0 else "#D97706" if rmse < 5.0 else "#DC2626"
            status = "GOOD" if rmse < 2.0 else "CAUTION" if rmse < 5.0 else "POOR"
            self._rmse_lbl.setText(f"RMSE vs Station {ref_idx+1}: {rmse:.3f} mm  [{status}]")
            self._rmse_lbl.setStyleSheet(f"color:{color};font-weight:bold;font-size:10pt;padding:6px;background:#F8FAFC;border-radius:4px;border:1px solid {color};")

    def _run_icp(self):
        if self.aligned_pts is None: return
        idx = self._station_combo.currentIndex()
        ref_idx = max(0, idx-1)
        ref = validate_xyz(self.context.scans[ref_idx].points)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            reg, rmse = self.reg_mod._icp(self.aligned_pts, ref)
            self.aligned_pts = reg
            color = "#047857" if rmse < 2.0 else "#D97706" if rmse < 5.0 else "#DC2626"
            status = "GOOD" if rmse < 2.0 else "CAUTION" if rmse < 5.0 else "POOR"
            self._rmse_lbl.setText(f"After ICP: {rmse:.3f} mm  [{status}]")
            self._rmse_lbl.setStyleSheet(f"color:{color};font-weight:bold;font-size:10pt;padding:6px;background:#F8FAFC;border-radius:4px;border:1px solid {color};")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _reset(self):
        for key, (slider, spin) in self._sliders.items():
            spin.setValue(0.0)


class _TargetDetectDialog(QtWidgets.QDialog):
    def __init__(self, bundle, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Target Detection Settings")
        self.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10); lay.setContentsMargins(16,16,16,16)
        has_int = bundle.intensity is not None and float(bundle.intensity.max()) > 0
        n_pts = len(bundle.points)
        info = QtWidgets.QLabel("Points: " + str(n_pts) + chr(10) + "Has intensity/color: " + str(has_int))
        info.setStyleSheet("background:#F0FDF4;border:1px solid #86EFAC;border-radius:4px;padding:6px;color:#166534;font-size:9pt;")
        lay.addWidget(info)
        if not has_int:
            warn = QtWidgets.QLabel("No intensity/color data - sphere and flat detection only.")
            warn.setStyleSheet("background:#FEF3C7;border:1px solid #FCD34D;border-radius:4px;padding:6px;color:#92400E;font-size:9pt;")
            lay.addWidget(warn)
        grp = QtWidgets.QGroupBox("Detection Types")
        grp.setStyleSheet("QGroupBox{font-weight:600;color:#065F46;border:1px solid #A7F3D0;border-radius:6px;margin-top:8px;padding:8px;}")
        g_lay = QtWidgets.QVBoxLayout(grp)
        self._chk_sphere = QtWidgets.QCheckBox("Sphere targets (RANSAC sphere fitting)")
        self._chk_flat   = QtWidgets.QCheckBox("Flat / Checkerboard targets (plane + FFT)")
        self._chk_int    = QtWidgets.QCheckBox("Intensity / Color targets (high-reflectance)")
        self._chk_sphere.setChecked(True)
        self._chk_flat.setChecked(True)
        self._chk_int.setChecked(has_int)
        self._chk_int.setEnabled(has_int)
        for chk in [self._chk_sphere, self._chk_flat, self._chk_int]:
            g_lay.addWidget(chk)
        lay.addWidget(grp)
        prm = QtWidgets.QGroupBox("Parameters")
        prm.setStyleSheet("QGroupBox{font-weight:600;color:#065F46;border:1px solid #A7F3D0;border-radius:6px;margin-top:8px;padding:8px;}")
        p_lay = QtWidgets.QFormLayout(prm)
        self._sp_r_min = QtWidgets.QDoubleSpinBox()
        self._sp_r_min.setRange(0.01,0.5); self._sp_r_min.setValue(0.05); self._sp_r_min.setSuffix(" m")
        self._sp_r_max = QtWidgets.QDoubleSpinBox()
        self._sp_r_max.setRange(0.05,1.0); self._sp_r_max.setValue(0.25); self._sp_r_max.setSuffix(" m")
        self._sp_cell_min = QtWidgets.QDoubleSpinBox()
        self._sp_cell_min.setRange(0.02,0.5); self._sp_cell_min.setValue(0.03); self._sp_cell_min.setSuffix(" m")
        self._sp_cell_max = QtWidgets.QDoubleSpinBox()
        self._sp_cell_max.setRange(0.05,1.0); self._sp_cell_max.setValue(0.50); self._sp_cell_max.setSuffix(" m")
        self._sp_contrast = QtWidgets.QDoubleSpinBox()
        self._sp_contrast.setRange(1.1,10.0); self._sp_contrast.setValue(1.3); self._sp_contrast.setSingleStep(0.1)
        self._sp_int_pct = QtWidgets.QDoubleSpinBox()
        self._sp_int_pct.setRange(80.0,99.9); self._sp_int_pct.setValue(97.0); self._sp_int_pct.setSuffix(" %")
        self._sp_min_pts = QtWidgets.QSpinBox()
        self._sp_min_pts.setRange(5,500); self._sp_min_pts.setValue(20); self._sp_min_pts.setSuffix(" pts")
        p_lay.addRow("Sphere radius min:", self._sp_r_min)
        p_lay.addRow("Sphere radius max:", self._sp_r_max)
        p_lay.addRow("Checker cell min:", self._sp_cell_min)
        p_lay.addRow("Checker cell max:", self._sp_cell_max)
        p_lay.addRow("Min contrast ratio:", self._sp_contrast)
        p_lay.addRow("Intensity percentile:", self._sp_int_pct)
        p_lay.addRow("Min cluster points:", self._sp_min_pts)
        lay.addWidget(prm)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_params(self):
        return {
            "detect_sphere":        self._chk_sphere.isChecked(),
            "detect_flat":          self._chk_flat.isChecked(),
            "detect_intensity":     self._chk_int.isChecked(),
            "sphere_radius_range":  (self._sp_r_min.value(), self._sp_r_max.value()),
            "intensity_percentile": self._sp_int_pct.value(),
            "min_cluster_pts":      self._sp_min_pts.value(),
            "cell_size_range":      (self._sp_cell_min.value(), self._sp_cell_max.value()),
            "min_contrast_ratio":   self._sp_contrast.value(),
        }


    def closeEvent(self, event) -> None:
        """Clean up timers and threads before closing."""
        try:
            if hasattr(self, "_anim_timer") and self.section_widget:
                self.section_widget._anim_timer.stop()
        except Exception: pass
        try:
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait(1000)
        except Exception: pass
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Tunnel Analysis v4.0")
    win = TunnelAnalysisWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
