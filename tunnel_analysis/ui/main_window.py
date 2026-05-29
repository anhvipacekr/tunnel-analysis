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

class TunnelAnalysisWindow(DispatchMixin, SlotsMixin, StationMixin, RenderMixin, QtWidgets.QMainWindow):

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
                ("1.1  Import LAS / PLY data", self._slot_1_1_import, "Tai du lieu diem tu file LAS/LAZ/PLY vao he thong"),
                ("1.2  Initialize 3D viewport", self._slot_1_2_viewport, "Khoi tao cua so hien thi 3D PyVista"),
                ("1.3  Add scan station", self._slot_1_3_add_scan, "Them tram scan moi vao danh sach"),
                ("1.4  Register and merge all stations", self._slot_1_4_merge, "Dang ky va gop tat ca tram scan thanh 1 dam may diem"),
                ("1.5  Rough alignment (manual)", self._slot_1_5_rough, "Can chinh thu cong bang offset va goc xoay"),
                ("1.6  Chain register and merge", self._slot_1_6_chain, "Dang ky day chuyen S1->S2->S3 theo thu tu"),
                ("1.7  Registration error heatmap", self._slot_1_7_reg_error, "Hien thi ban do sai so dang ky giua cac tram"),
            ]),
            (2, "Preprocessing and noise filtering", "Pre.", [
                ("2.1  Voxel downsampling", self._slot_2_1_voxel, "Giam mat do diem bang luoi voxel, giu hinh dang ham"),
                ("2.2  Statistical outlier removal", self._slot_2_2_sor, "Loai diem nhieu bang thong ke median+MAD theo tung mat cat"),
                ("2.3  Extract tunnel lining shell", self._slot_2_3_lining, "Trich xuat be mat vo ham, loai bo cap dien va den"),
                ("2.4  Semantic noise removal (PDF 3.2)", self._slot_2_4_semantic, "Phan loai va loai bo: cap=linearity cao, den=sphericity cao, nguoi=chieu cao 1.2-2.2m"),
            ]),
            (3, "Registration and synchronization", "Reg.", [
                ("3.1  Anchor translation", self._slot_3_1_anchor, "Dich chuyen diem dua tren diem neo cuong do cao nhat"),
                ("3.2  Fine surface ICP", self._slot_3_2_icp, "Tinh chinh dang ky bang ICP point-to-plane 2 buoc"),
                ("3.3  Calculate RMSE", self._slot_3_3_rmse, "Tinh sai so RMSE giua dam may diem hien tai va tham chieu"),
            ]),
            (4, "Geometric coordinate system", "Geo.", [
                ("4.1  Extract PCA centerline", self._slot_4_1_centerline, "Trich xuat duong tam ham bang PCA sliding window"),
                ("4.2  Iterative centerline refinement", self._slot_4_2_iterative, "Tinh chinh duong tam bang RANSAC circle per section (Yi 2020)"),
                ("4.3  Smooth B-Spline centerline", self._slot_4_3_bspline, "Lam min duong tam bang B-Spline bac 3"),
                ("4.3b B-Spline C2 centerline (PDF 3.4)", self._slot_4_3b_bspline, "Duong tam B-Spline lien tuc C2, khong co goc gap"),
                ("4.4  Generate Frenet N-B sections", self._slot_4_4_frenet, "Tao he toa do Frenet-Serret T/N/B can bang theo trong luc"),
                ("4.5  Detect ring seams", self._slot_4_5_seams, "Phat hien duong noi vong be tong bang phan cum"),
                ("4.5b Intensity ring seam detection (PDF 3.3)", self._slot_4_5b_intensity_seams, "Phat hien duong noi vong bang dao ham cuong do laser"),
            ]),
            (5, "Parameter extraction", "Param.", [
                ("5.1  Crown settlement dv", self._slot_5_1_settlement, "Do lun dinh ham: dich chuyen dung tai diem cao nhat"),
                ("5.2  Horizontal convergence dh", self._slot_5_2_convergence, "Do hoi tu ngang: tong dich chuyen 2 tuong vao trong"),
                ("5.3  3D deformation heatmap", self._slot_5_3_heatmap, "Ban do nhiet bien dang 3D: xanh<1mm vang 1-3mm do>3mm"),
                ("5.3b Hausdorff heatmap T0->Tn (PDF 3.5)", self._slot_5_3b_hausdorff, "Khoang cach Hausdorff T0->Tn: do bien dang theo thoi gian"),
                ("5.4  Polar radial deformation dr", self._slot_5_4_polar, "Bien dang huong kinh dr(theta) theo tung goc quanh mat cat"),
                ("5.5  Ovality epsilon", self._slot_5_5_ovality, "Do oval hoa epsilon=(a-b)/a*100%: muc do bien dang hinh dang"),
                ("5.6  Section eccentricity e", self._slot_5_6_eccentricity, "Do lech tam e: khoang cach tam do vs tam thiet ke"),
                ("5.7  Plot 2D Technical Section", self._slot_5_7_sections, "Ve mat cat ky thuat 2D theo mat phang Frenet N-B"),
                ("5.8  Clearance 3D violation map", self._slot_5_8_clearance_3d, "Ban do 3D vi pham gioi han khoang thong xe"),
            ]),
            (6, "Time-series analysis", "T-S", [
                ("6.1  Load T0 and Tn epochs", self._slot_6_1_epochs, "Tai 2 file diem: T0=tham chieu, Tn=do lan sau"),
                ("6.2  Plot deformation trend", self._slot_6_2_plot, "Ve bieu do xu huong bien dang cloud-to-cloud T0->Tn"),
            ]),
            (7, "BIM and AI", "BIM/AI", [
                ("7.1  Export IFC package", self._slot_7_1_ifc, "Xuat mo hinh IFC4 voi solid geometry Wall/Slab/Space"),
                ("7.2  Query structural AI assistant", self._slot_7_2_query_ai, "Hoi AI ve tinh trang ham, nguong an toan, khuyen nghi"),
            ]),
        ]
        for step, title_s, tag, buttons in SECTIONS:
            sec = CollapsibleSection(title_s, step, tag)
            for label, slot, tip in buttons:
                btn = sec.add_sub_button(label, slot, tip); self._all_sub_btns.append(btn)
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



from .dispatch import DispatchMixin
from .slots import SlotsMixin
from .station import StationMixin
from .render import RenderMixin

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
