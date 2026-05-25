"""
ifc_exporter.py - Real IFC4 export using ifcopenshell 0.8+.
Per PDF section 3.6.
"""
from .common import *
from .models import PipelineContext, SectionGeometry
from pathlib import Path
from datetime import datetime


class TunnelIFCExporter:
    """Export tunnel analysis results to IFC4 format."""

    def export_ifc(self, context: PipelineContext, out_path: str,
                   project_name: str = "Tunnel Analysis",
                   engineer: str = "CBNU Smart Structure Lab") -> str:
        try:
            import ifcopenshell
            import ifcopenshell.api
        except ImportError:
            raise RuntimeError("ifcopenshell required: pip install ifcopenshell")

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ifc = ifcopenshell.file(schema="IFC4")

        # Project
        project = ifcopenshell.api.run("root.create_entity", ifc,
                                        ifc_class="IfcProject", name=project_name)
        ifcopenshell.api.run("unit.assign_unit", ifc)
        ctx3d = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
        body_ctx = ifcopenshell.api.run("context.add_context", ifc,
                                         context_type="Model",
                                         context_identifier="Body",
                                         target_view="MODEL_VIEW",
                                         parent=ctx3d)

        # Hierarchy: Project > Site > Building > Storey
        site = ifcopenshell.api.run("root.create_entity", ifc,
                                     ifc_class="IfcSite", name="Osong Test Site")
        building = ifcopenshell.api.run("root.create_entity", ifc,
                                         ifc_class="IfcBuilding", name="Tunnel Structure")
        storey = ifcopenshell.api.run("root.create_entity", ifc,
                                       ifc_class="IfcBuildingStorey", name="Tunnel Level")

        # API 0.8+: products is a list
        ifcopenshell.api.run("aggregate.assign_object", ifc,
                              products=[site], relating_object=project)
        ifcopenshell.api.run("aggregate.assign_object", ifc,
                              products=[building], relating_object=site)
        ifcopenshell.api.run("aggregate.assign_object", ifc,
                              products=[storey], relating_object=building)

        # Centerline as IfcAnnotation
        cl = context.centerline
        if cl is not None and len(cl) >= 2:
            cl_entity = ifcopenshell.api.run("root.create_entity", ifc,
                                              ifc_class="IfcAnnotation",
                                              name="Tunnel Centerline")
            pts_3d = [ifc.createIfcCartesianPoint(
                (float(p[0]), float(p[1]), float(p[2]))) for p in cl]
            polyline = ifc.createIfcPolyline(pts_3d)
            shape = ifc.createIfcShapeRepresentation(
                body_ctx, "Axis", "Curve3D", [polyline])
            prod_def = ifc.createIfcProductDefinitionShape(None, None, [shape])
            cl_entity.Representation = prod_def
            ifcopenshell.api.run("spatial.assign_container", ifc,
                                  products=[cl_entity], relating_structure=storey)

        # Sections as IfcBuildingElementProxy
        for i, sec in enumerate(context.sections):
            if sec.center_3d is None:
                continue
            name = f"Section_{i+1:03d}_Ch{sec.chainage:.2f}m"
            elem = ifcopenshell.api.run("root.create_entity", ifc,
                                         ifc_class="IfcBuildingElementProxy",
                                         name=name)
            # Property set
            pset = ifcopenshell.api.run("pset.add_pset", ifc,
                                         product=elem,
                                         name="TunnelSectionProperties")
            props = {}
            if np.isfinite(sec.chainage):     props["Chainage_m"]       = float(sec.chainage)
            if np.isfinite(sec.H1):           props["ClearHeight_H1_m"] = float(sec.H1)
            if np.isfinite(sec.W1):           props["ClearWidth_W1_m"]  = float(sec.W1)
            if np.isfinite(sec.ovality):      props["Ovality_pct"]      = float(sec.ovality)
            if np.isfinite(sec.eccentricity): props["Eccentricity_mm"]  = float(sec.eccentricity)
            if np.isfinite(sec.radius_fit):   props["RadiusFit_m"]      = float(sec.radius_fit)
            props["ClearanceViolation"] = bool(sec.clearance_violation)
            if np.isfinite(sec.min_clearance_dist):
                props["MinClearance_m"] = float(sec.min_clearance_dist)
            ifcopenshell.api.run("pset.edit_pset", ifc, pset=pset, properties=props)
            ifcopenshell.api.run("spatial.assign_container", ifc,
                                  products=[elem], relating_structure=storey)

        # Global parameters on project
        params = context.parameters
        if params:
            pset_g = ifcopenshell.api.run("pset.add_pset", ifc,
                                           product=project,
                                           name="TunnelGlobalParameters")
            clean = {k: float(v) for k, v in params.items()
                     if isinstance(v, (int, float)) and np.isfinite(float(v))}
            if clean:
                ifcopenshell.api.run("pset.edit_pset", ifc, pset=pset_g, properties=clean)

        ifc.write(str(path))
        return str(path)
