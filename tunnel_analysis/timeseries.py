from .common import *
from .models import PointCloudBundle, PipelineContext
from .io_layer import BaseLayer
# ------------------------------------------------------------------------------
# TimeSeriesLayer
# ------------------------------------------------------------------------------

class TimeSeriesLayer:
    def load_epochs(self, p0: str, pn: str) -> Tuple[PointCloudBundle, PointCloudBundle]:
        bl = BaseLayer(); return bl.load_scan(p0), bl.load_scan(pn)

    def plot_deformation(self, context: PipelineContext) -> np.ndarray:
        pts = context.working_points
        if pts is None: raise RuntimeError("Load epochs first.")
        pts = validate_xyz(pts)
        sc  = (pts[:,2] - np.median(pts[:,2]))*1e3
        ord_= np.argsort(pts[:,0])
        return np.array([float(np.nanmean(c)) for c in np.array_split(sc[ord_],120) if len(c)>0])


