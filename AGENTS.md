# AGENTS.md — Tunnel Analysis v4.0
# CBNU Smart Structure Lab
# Reference document: Final 12.05.2026.pdf (Proposal Lidar)

## ROLE
You are an expert Python developer continuing development of "Tunnel Analysis v4.0" — a LiDAR-based structural health monitoring desktop application for railway tunnels at CBNU Smart Structure Lab.

**Your primary reference is the PDF proposal at:**
`C:\Users\ssl\Desktop\Proposal Lidar\Final 12.05.2026.pdf`

Always read this PDF first before making any changes. Every feature must align with the PDF specification.

---

## EXISTING CODEBASE (DO NOT REWRITE FROM SCRATCH)

**Repository:** https://github.com/anhvipacekr/tunnel-analysis
**Working directory:** `C:\Users\ssl\Desktop\Code Python\Test AI tunnel\`
**Python venv:** `C:\Users\ssl\Desktop\Code Python\data python cusor\.venv311\Scripts\python.exe`
**Run app:** `python run_tunnel_analysis.py`

### Module structure (already implemented):
```
tunnel_analysis/
  common.py          # shared imports, constants, utilities
  models.py          # PointCloudBundle, SectionGeometry, PipelineContext
  io_layer.py        # LAS/PLY reader, RGB->luminance fallback
  preprocessing.py   # voxel, SOR (median+MAD), lining extraction (4-pass)
  registration.py    # anchor, ICP, chain registration, rough alignment
  geometry.py        # PCA centerline, B-spline C2, Frenet-Serret frames
  segmentation.py    # ring segmentation, intensity seam detection
  parameters.py      # settlement, convergence, ovality, eccentricity, polar, Hausdorff
  timeseries.py      # T0/Tn epoch comparison
  digital_twin.py    # Ollama LLM hook
  exporter.py        # CSV, Excel (5 sheets)
  pdf_reporter.py    # PDF report (reportlab)
  ifc_exporter.py    # IFC4 export (ifcopenshell)
  web_dashboard.py   # Dash web dashboard
  rag_ai.py          # ChromaDB + sentence-transformers RAG
  target_detector.py # sphere/flat/checkerboard/intensity/faro/manual targets
  worker.py          # QThread PipelineWorker
  ui/
    main_window.py   # TunnelAnalysisWindow (main GUI)
    widgets.py       # 2D section, polar plot, line plot
```

---

## PDF SPECIFICATION (PRIMARY REFERENCE)

### System Architecture (PDF §3.1)
5-layer pipeline — Physical → Digital → Decision:
```
Layer 1 (Base):  LiDAR/SLAM → 3D coords + intensity
Layer 2 (Pre.):  Statistical filter + voxel + lining extraction
Layer 3 (Geo.):  Ring segmentation + B-spline centerline + Frenet frames
Layer 4 (BIM):   Parameter extraction + IFC + clearance analysis
Layer 5 (AI):    RAG + Local LLM + natural language interface
```

### Layer 2 — Preprocessing (PDF §3.2)
- Partition tunnel into 1m sections along axis
- Per section: fit plane (LSQ), compute signed distance per point
- Statistical filter: μ ± kσ dynamic threshold → remove outliers
- Voxel downsampling: uniform density, remove sampling bias
- Extract tunnel lining surface only (remove cables, lights, people)

### Layer 3 — Centerline & Registration (PDF §3.3, §3.4)
- Ring seam detection: intensity 1st derivative → valley = seam boundary
- ICP registration: kd-tree correspondence → argmin_{R,t} Σ||p_i - (R·q_i + t)||²
- B-spline centerline: sliding window curvature → B-spline C² fit (no kinks)
- Frenet-Serret: T=r'(s)/|r'(s)|, N=T'(s)/|T'(s)|, B=T×N
- N-B plane = cross-section cutting plane (eliminates perspective distortion)

### Layer 4 — Parameters (PDF §3.5)
Extract per section:
| Parameter | Formula | Physical meaning |
|-----------|---------|-----------------|
| δv Crown settlement | ΔZ_max vs T0 | Vertical load deformation |
| δh Convergence | ΔX_left + ΔX_right vs T0 | Lateral ground pressure |
| ε Ovality | (a-b)/a × 100% | Shape distortion |
| e Eccentricity | |C_meas - C_design| | Center offset |
| Δr Polar deformation | Δr(θ) per angle bin | Local deformation |

Hausdorff heatmap T0→Tn:
- Green: stable (<1mm)
- Yellow: caution (1-3mm)
- Red: critical (>3mm)

### Layer 4 — Clearance & BIM (PDF §3.6)
- Vehicle clearance envelope inside point cloud
- Collision detection → flag violation points
- Quantify violation distance
- Convert to IFC4 solid: Wall/Slab/Space classification
- Export to 5D lifecycle management software

### Layer 5 — AI Assistant (PDF §3.7)
- Convert settlement/convergence/heatmap to structured text
- Vectorize → local AI knowledge database (RAG)
- Load safety standards as judgment criteria
- Natural language Q&A with engineer
- Generate detailed reports + risk warnings + work orders

---

## SAFETY THRESHOLDS (Korean Railway Standards)
```python
THRESHOLDS = {
    "crown_settlement_mm":    {"caution": 10.0,  "critical": 25.0},
    "lateral_convergence_mm": {"caution": 15.0,  "critical": 30.0},
    "ovality_mean_pct":       {"caution":  0.5,  "critical":  1.0},
    "eccentricity_mean_mm":   {"caution": 10.0,  "critical": 25.0},
    "hausdorff_mm":           {"green":   1.0,   "yellow":    3.0},
}
```

---

## IMPLEMENTATION SCHEDULE (PDF §4)
- Phase 1 (→ Sep 2026): Data collection, sync, preprocessing
- Phase 2 (Sep→Oct 2026): Parameter extraction module
- Phase 3 (Oct→Nov 2026): Clearance analysis + deformation warning
- Phase 4 (Nov→Dec 2026): Digital twin + AI integration + web dashboard

---

## CURRENT STATUS & KNOWN GAPS

### Already implemented (✅):
- All 5 layers basic pipeline
- B-spline C2 centerline + Frenet frames
- Intensity ring seam detection
- Settlement/convergence with T0 comparison
- Ovality, eccentricity, polar deformation
- Hausdorff heatmap T0→Tn
- 2D cross-section engineering drawing (zoom/pan, T0 overlay, animation)
- CSV/Excel/PDF/IFC4 export
- Web dashboard (Dash)
- RAG AI (ChromaDB + sentence-transformers)
- Auto Pipeline 1-click
- Station tree UI (Faro SCENE style)
- Manual target picking + auto-refine
- Chain registration + rough alignment dialog
- Noise removal with interactive review panel

### Needs improvement (🔶) — prioritize per PDF:
1. **Preprocessing (PDF §3.2)**: Semantic object removal (cables/lights/people) — currently only radius band filter
2. **Settlement/Convergence (PDF §3.5)**: Must compare vs T0 reference scan — partially done
3. **IFC export (PDF §3.6)**: Currently metadata only — needs mesh/solid geometry conversion
4. **AI RAG (PDF §3.7)**: Needs safety standards database + work order generation
5. **Web dashboard (PDF §4 Phase 4)**: Needs more interactive features

### Not yet done (❌):
- Semantic object classification (cables vs lights vs people)
- IFC solid geometry (Wall/Slab/Space objects)
- Work order generation from AI
- Predictive maintenance module

---

## CODING RULES

1. **Always read existing code before modifying** — use `grep` or read file first
2. **Never rewrite working modules** — extend, don't replace
3. **PDF is the spec** — every feature must map to a PDF section
4. **Test after every change** — run `python -m py_compile` then regression test
5. **Commit after major features**:
   ```
   git add tunnel_analysis/
   git commit -m "feat: description (PDF §X.X)"
   git push
   ```
6. **Sync both locations** after changes:
   - `C:\Users\ssl\Desktop\Code Python\Test AI tunnel\tunnel_analysis\`
   - `C:\Users\ssl\Desktop\Code Python\data python cusor\tunnel_analysis\`
7. **No inline comments** unless explicitly requested
8. **Threading**: all heavy tasks via `PipelineWorker` (QThread)
9. **Error handling**: `QMessageBox` for user-facing errors, log to `results_text`
10. **String literals**: avoid multiline f-strings in PowerShell patches — use `chr(10)` for newlines

---

## REGRESSION TEST
Run before committing:
```powershell
$py = "C:\Users\ssl\Desktop\Code Python\data python cusor\.venv311\Scripts\python.exe"
& $py "$env:TEMP\final_full_regression.py"
# Expected: ALL 18 TESTS PASSED
```

---

## PRIORITY ORDER FOR NEXT IMPROVEMENTS
Based on PDF implementation schedule:

1. **Phase 1 priority**: Improve preprocessing — semantic noise removal (PDF §3.2)
2. **Phase 2 priority**: Improve parameter accuracy — proper T0 comparison for all metrics (PDF §3.5)
3. **Phase 3 priority**: Clearance analysis visualization + warning system (PDF §3.6)
4. **Phase 4 priority**: IFC solid export + AI work order generation (PDF §3.6, §3.7)
