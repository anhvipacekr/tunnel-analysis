"""
rag_ai.py - RAG-enhanced local AI assistant with tunnel safety standards.
Per PDF section 3.7: vector DB + local LLM + safety standards knowledge base.
"""
from .common import *
from .models import PipelineContext
from pathlib import Path
import json


# ── Korean/International tunnel safety standards knowledge base ────────────
SAFETY_STANDARDS = [
    # Crown settlement
    "Crown settlement threshold: caution >10mm, critical >25mm. "
    "Per KR C-08080 (Korean Railway Safety Standards) and ITA guidelines.",

    "Crown settlement (delta_v) is measured as vertical displacement at tunnel crown. "
    "Immediate inspection required when delta_v exceeds 25mm. "
    "Monitoring frequency should increase when delta_v exceeds 10mm.",

    # Convergence
    "Horizontal convergence threshold: caution >15mm, critical >30mm. "
    "Convergence (delta_h) = sum of inward wall displacement on both sides.",

    "When horizontal convergence exceeds 30mm, tunnel may require emergency shoring. "
    "Per NATM guidelines, convergence rate >2mm/day requires immediate action.",

    # Ovality
    "Ovality (epsilon) threshold: caution >0.5%, critical >1.0%. "
    "Ovality = (a-b)/a * 100% where a=major axis, b=minor axis of fitted ellipse.",

    "High ovality indicates uneven ground pressure. "
    "Ovality >1% in circular tunnels suggests potential lining distress.",

    # Eccentricity
    "Eccentricity (e) threshold: caution >10mm, critical >25mm. "
    "Eccentricity = distance between measured center and design center.",

    "Large eccentricity may indicate differential settlement or construction error. "
    "Per KDS 27 25 00 (Korean Design Standard for Tunnels).",

    # Clearance
    "Vehicle clearance violation is a critical safety issue requiring immediate action. "
    "Korean Railway Act Article 26: minimum clearance must be maintained at all times.",

    "Clearance envelope for standard Korean railway: width 3.0m, height 4.5m for box section. "
    "Circle tunnel: minimum radius 3.0m from track centerline.",

    # Heatmap
    "Hausdorff distance heatmap color coding: "
    "green = stable (<1mm), yellow = caution (1-3mm), red = critical (>3mm). "
    "Red zones require priority inspection and possible repair.",

    # General
    "LiDAR-based tunnel inspection should be performed at minimum annually. "
    "High-risk tunnels (age >30 years, heavy traffic) require semi-annual inspection.",

    "Point cloud registration RMSE should be <2mm for reliable deformation analysis. "
    "ICP convergence criteria: relative fitness <1e-6, relative RMSE <1e-6.",

    "Frenet-Serret coordinate system ensures orthogonal cross-section extraction. "
    "Non-orthogonal sections cause apparent ovality errors up to 15%.",

    "B-spline centerline with C2 continuity eliminates kink artifacts in curved tunnels. "
    "Sliding window curvature detection identifies direction change points automatically.",

    "Ring seam detection using intensity derivative: "
    "concrete ring joints show intensity drop of 30-60% compared to lining surface. "
    "Typical ring spacing: 1.0-1.5m for precast concrete segments.",

    "Statistical outlier removal (SOR): partition tunnel into 1m sections, "
    "compute radial deviation per section, remove points outside mu +/- 2.5*sigma.",

    "Voxel downsampling grid size recommendation: 0.05m for high-density scans, "
    "0.02m for precision analysis, 0.10m for quick preview.",
]


class TunnelRAGAssistant:
    """RAG-enhanced AI assistant with tunnel safety knowledge base."""

    OLLAMA_URL   = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3"
    _TIMEOUT     = (5.0, 120.0)
    _DB_PATH     = str(Path.home() / ".tunnel_analysis" / "chroma_db")

    def __init__(self):
        self._collection = None
        self._embedder   = None
        self._ready      = False

    def initialize(self) -> str:
        """Initialize ChromaDB + sentence-transformers embedder."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return "RAG dependencies missing: pip install chromadb sentence-transformers"

        try:
            Path(self._DB_PATH).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self._DB_PATH)
            self._collection = client.get_or_create_collection(
                name="tunnel_safety",
                metadata={"hnsw:space": "cosine"})

            # Load embedder (small, fast model)
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Index safety standards if not already indexed
            existing = self._collection.count()
            if existing < len(SAFETY_STANDARDS):
                ids  = [f"std_{i}" for i in range(len(SAFETY_STANDARDS))]
                embs = self._embedder.encode(SAFETY_STANDARDS).tolist()
                self._collection.upsert(
                    ids=ids,
                    embeddings=embs,
                    documents=SAFETY_STANDARDS)

            self._ready = True
            return f"RAG initialized: {self._collection.count()} safety standards indexed."
        except Exception as e:
            return f"RAG initialization failed: {e}"

    def query(self, prompt: str, context: PipelineContext,
              n_results: int = 5) -> str:
        """Query local LLM with RAG context from safety standards."""
        # Build context string from parameters
        params_str = self._build_params_str(context)

        # RAG retrieval
        rag_context = ""
        if self._ready and self._embedder and self._collection:
            try:
                q_emb = self._embedder.encode([prompt]).tolist()
                results = self._collection.query(
                    query_embeddings=q_emb, n_results=n_results)
                docs = results.get("documents", [[]])[0]
                if docs:
                    rag_context = "\n".join(f"- {d}" for d in docs)
            except Exception as e:
                rag_context = f"(RAG retrieval failed: {e})"

        # Build full prompt
        system_prompt = f"""You are a licensed structural engineer specialising in tunnel SHM.
Answer based on the measurement data and safety standards provided.
Be concise, quantitative, and actionable.

=== TUNNEL MEASUREMENT DATA ===
{params_str}

=== RELEVANT SAFETY STANDARDS ===
{rag_context if rag_context else "(Safety standards not available - answer from general knowledge)"}

=== ENGINEER QUERY ===
{prompt}

Provide:
1. Assessment of current tunnel condition
2. Parameters exceeding thresholds (if any)
3. Recommended actions with priority
4. Locations requiring immediate attention"""

        # Query Ollama
        try:
            import requests
            payload = {
                "model": self.OLLAMA_MODEL,
                "prompt": system_prompt,
                "stream": False,
                "options": {"temperature": 0.15, "num_predict": 1500}
            }
            r = requests.post(self.OLLAMA_URL, json=payload, timeout=self._TIMEOUT)
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "").strip()
            if not text:
                return "[EMPTY RESPONSE]\n" + json.dumps(data, indent=2)[:400]
            m  = data.get("model", "unknown")
            n  = data.get("eval_count", "?")
            es = data.get("eval_duration", 0) / 1e9
            rag_note = f"RAG: {n_results} standards retrieved" if self._ready else "RAG: not initialized"
            return f"{text}\n\n{'─'*52}\nModel: {m} | Tokens: {n} | Eval: {es:.1f}s | {rag_note}"
        except Exception as e:
            return (f"[CONNECTION ERROR] {e}\n\n"
                    f"Start Ollama: ollama serve\n"
                    f"Pull model: ollama pull {self.OLLAMA_MODEL}\n\n"
                    f"--- Offline Analysis ---\n"
                    f"{self._offline_analysis(context)}")

    def _build_params_str(self, context: PipelineContext) -> str:
        lines = []
        p = context.parameters
        if p:
            for k, v in p.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    lines.append(f"  {k}: {v:.3f}")
        if context.sections:
            n_viol = sum(1 for s in context.sections if s.clearance_violation)
            ov_vals = [s.ovality for s in context.sections if np.isfinite(s.ovality)]
            ec_vals = [s.eccentricity for s in context.sections if np.isfinite(s.eccentricity)]
            lines.append(f"  sections_count: {len(context.sections)}")
            lines.append(f"  clearance_violations: {n_viol}")
            if ov_vals:
                lines.append(f"  ovality_max_pct: {max(ov_vals):.3f}")
            if ec_vals:
                lines.append(f"  eccentricity_max_mm: {max(ec_vals):.3f}")
        scan = context.active_scan
        if scan:
            lines.append(f"  scan_points: {len(scan.points):,}")
            lines.append(f"  scan_file: {scan.path or 'N/A'}")
        return "\n".join(lines) if lines else "  (no parameters extracted yet)"

    def _offline_analysis(self, context: PipelineContext) -> str:
        """Offline rule-based analysis when Ollama is not available."""
        THRESHOLDS = {
            "crown_settlement_mm":    ("Crown Settlement",    10.0, 25.0, "mm"),
            "lateral_convergence_mm": ("Convergence",         15.0, 30.0, "mm"),
            "ovality_mean_pct":       ("Ovality",              0.5,  1.0, "%"),
            "eccentricity_mean_mm":   ("Eccentricity",        10.0, 25.0, "mm"),
        }
        p = context.parameters or {}
        lines = ["OFFLINE RULE-BASED ASSESSMENT", "=" * 40]
        overall = "STABLE"
        for key, (label, c_thr, r_thr, unit) in THRESHOLDS.items():
            val = p.get(key)
            if not isinstance(val, (int, float)) or not np.isfinite(float(val)):
                continue
            if val >= r_thr:
                lines.append(f"[CRITICAL] {label} = {val:.2f}{unit} (threshold: {r_thr}{unit})")
                overall = "CRITICAL"
            elif val >= c_thr:
                lines.append(f"[CAUTION]  {label} = {val:.2f}{unit} (threshold: {c_thr}{unit})")
                if overall == "STABLE": overall = "CAUTION"
            else:
                lines.append(f"[OK]        {label} = {val:.2f}{unit}")
        if context.sections:
            n_viol = sum(1 for s in context.sections if s.clearance_violation)
            if n_viol:
                lines.append(f"[CRITICAL] {n_viol} clearance violation(s) detected")
                overall = "CRITICAL"
        lines.append(f"\nOVERALL STATUS: {overall}")
        if overall == "CRITICAL":
            lines.append("ACTION: Immediate inspection and engineering assessment required.")
        elif overall == "CAUTION":
            lines.append("ACTION: Schedule detailed inspection within 30 days.")
        else:
            lines.append("ACTION: Continue routine monitoring schedule.")
        return "\n".join(lines)
