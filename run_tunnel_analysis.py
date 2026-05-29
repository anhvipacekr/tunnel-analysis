import os
import warnings
import logging

# Suppress VTK/PyVista warning spam
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"
os.environ["PYVISTA_OFF_SCREEN"] = "false"

# Use HuggingFace offline mode to suppress token warning
# Models are cached locally after first download
import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Prevent recursive WARNING:root: loop in PyVista VTK error handler
logging.getLogger("root").setLevel(logging.CRITICAL)
logging.getLogger("pyvista").setLevel(logging.ERROR)
logging.getLogger("vtk").setLevel(logging.ERROR)

# Disable all root logger handlers to stop the loop
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(logging.NullHandler())

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*VTK.*")
warnings.filterwarnings("ignore", message=".*pyvista.*")

try:
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass
try:
    import vtkmodules.vtkRenderingCore as _vtk_rc
    _vtk_rc.vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass

from tunnel_analysis.main import main

if __name__ == "__main__":
    raise SystemExit(main())
