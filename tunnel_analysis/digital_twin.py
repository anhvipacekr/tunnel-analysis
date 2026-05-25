from .common import *
from .models import PipelineContext
# ------------------------------------------------------------------------------
# DigitalTwinAILayer
# ------------------------------------------------------------------------------

class DigitalTwinAILayer:
    OLLAMA_URL   = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3"
    _TIMEOUT     = (5.0, 120.0)

    def export_ifc(self, context: PipelineContext) -> Dict[str, object]:
        scan = context.active_scan
        return {"ifc_schema":"IFC4","status":"ready_for_ifcopenshell_hook",
                "scan_path": scan.path if scan else None,
                "point_count": int(len(scan.points)) if scan else 0,
                "centerline_points": 0 if context.centerline is None else int(len(context.centerline)),
                "frenet_frames": int(len(context.frenet_frames)),
                "parameters": context.parameters, "local_llm_status":"ready_for_ollama_hook"}

    def query_local_ai(self, prompt: str, context: PipelineContext) -> str:
        try: import requests
        except ImportError: return "[ERROR] pip install requests"
        sys_p = self._sys(context)
        payload = {"model":self.OLLAMA_MODEL,
                   "prompt":f"{sys_p}\n\nEngineer query: {prompt}",
                   "stream":False,"options":{"temperature":0.2,"num_predict":1024}}
        try:
            r = requests.post(self.OLLAMA_URL, json=payload, timeout=self._TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            return (f"[CONNECTION ERROR] {e}\n\n"
                    f"Start Ollama: ollama serve\nPull: ollama pull {self.OLLAMA_MODEL}")
        try: data = r.json()
        except ValueError: return "[PARSE ERROR]\n" + r.text[:600]
        text = data.get("response","").strip()
        if not text: return "[EMPTY]\n" + json.dumps(data,indent=2)[:400]
        m=data.get("model", "unknown"); n=data.get("eval_count", "unknown")
        es=data.get("eval_duration",0)/1e9; ls=data.get("load_duration",0)/1e9
        return f"{text}\n\n{'-'*52}\nModel:{m}|Tokens:{n}|Eval:{es:.1f}s|Load:{ls:.1f}s"

    def _sys(self, context: PipelineContext) -> str:
        p = context.parameters
        lines = ("\n".join(f"  - {k}: {v:.3f}" for k,v in p.items()
                           if isinstance(v,(int,float))) if p else "  (none)")
        scan = context.active_scan
        info = (f"Scan: {scan.path}\nPoints: {len(scan.points):,}" if scan else "not loaded")
        return (
            "You are a licensed structural engineer specialising in tunnel SHM.\n"
            "Thresholds: Crown settlement >10mm=caution|>25mm=critical; "
            "Lateral convergence >15mm=caution|>30mm=critical; "
            "Ovality >0.5%=caution|>1.0%=critical.\n"
            f"--- Scan ---\n{info}\n--- Parameters ---\n{lines}\n--------------------------------"
        )


