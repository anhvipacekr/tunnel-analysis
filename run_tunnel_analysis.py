import os
import warnings
# Suppress VTK output window warnings
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"
try:
    import vtkmodules.vtkRenderingCore as _vtk_rc
    _vtk_rc.vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass
try:
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning)

from tunnel_analysis.main import main

if __name__ == "__main__":
    raise SystemExit(main())
