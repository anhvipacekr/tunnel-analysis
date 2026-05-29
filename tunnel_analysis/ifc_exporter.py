"""
ifc_exporter.py - IFC4 export with solid geometry per PDF section 3.6.
Converts tunnel sections to IfcExtrudedAreaSolid (Wall/Slab/Space classification).
"""
from .common import *
from .models import PipelineContext, SectionGeometry
from pathlib import Path
from datetime import datetime


class TunnelIFCExporter:

    def export_ifc(self, context: PipelineContext, out_path: str,
                   project_name: str = "Tunnel Analysis",
                   engineer: str = "CBNU Smart Structure Lab") -> str:
        try:
            import ifcopenshell
            import ifcopenshell.api
            import ifcopenshell.util.element
        except ImportError:
            raise RuntimeError("ifcopenshell required: pip install ifcopenshell")

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ifc = ifcopenshell.file(schema="IFC4")

        # Project hierarchy
        project = ifcopenshell.api.run("root.create_entity", ifc,
                                        ifc_class="IfcProject", name=project_name)
        ifcopenshell.api.run("unit.assign_unit", ifc)
        ctx3d = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
        body_ctx = ifcopenshell.api.run("context.add_context", ifc,
                                         context_type="Model",
                                         context_identifier="Body",
                                         target_view="MODEL_VIEW",
                                         parent=ctx3d)

        site     = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite",     name="Osong Test Site")
        building = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Tunnel Structure")
        storey   = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcBuildingStorey", name="Tunnel Level")

        ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site],     relating_object=project)
        ifcopenshell.api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)
        ifcopenshell.api.run("aggregate.assign_object", ifc, products=[storey],   relating_object=building)

        # Centerline as IfcAnnotation
        cl = context.centerline
        if cl is not None and len(cl) >= 2:
            cl_entity = ifcopenshell.api.run("root.create_entity", ifc,
                                              ifc_class="IfcAnnotation", name="Tunnel Centerline")
            pts_3d  = [ifc.createIfcCartesianPoint((float(p[0]), float(p[1]), float(p[2]))) for p in cl]
            polyline = ifc.createIfcPolyline(pts_3d)
            shape    = ifc.createIfcShapeRepresentation(body_ctx, "Axis", "Curve3D", [polyline])
            prod_def = ifc.createIfcProductDefinitionShape(None, None, [shape])
            cl_entity.Representation = prod_def
            ifcopenshell.api.run("spatial.assign_container", ifc,
                                  products=[cl_entity], relating_structure=storey)

        # Sections as solid geometry
        profile  = context.tunnel_profile or "Circle"
        sections = context.sections or []

        for i, sec in enumerate(sections):
            if sec.center_3d is None:
                continue

            name = f"TunnelSection_{i+1:03d}_Ch{sec.chainage:.2f}m"

            # Classify element type per PDF 3.6
            if sec.clearance_violation:
                ifc_class = "IfcWall"
            elif np.isfinite(sec.ovality) and sec.ovality > 0.5:
                ifc_class = "IfcSlab"
            else:
                ifc_class = "IfcSpace"

            elem = ifcopenshell.api.run("root.create_entity", ifc,
                                         ifc_class=ifc_class, name=name)

            # Build solid geometry
            solid = self._make_section_solid(ifc, body_ctx, sec, profile)
            if solid is not None:
                shape    = ifc.createIfcShapeRepresentation(body_ctx, "Body", "SweptSolid", [solid])
                prod_def = ifc.createIfcProductDefinitionShape(None, None, [shape])
                elem.Representation = prod_def

            # Place element at section center
            origin = ifc.createIfcCartesianPoint((
                float(sec.center_3d[0]),
                float(sec.center_3d[1]),
                float(sec.center_3d[2]),
            ))
            axis    = ifc.createIfcDirection((0.0, 0.0, 1.0))
            ref_dir = ifc.createIfcDirection((1.0, 0.0, 0.0))
            placement = ifc.createIfcAxis2Placement3D(origin, axis, ref_dir)
            local_pl  = ifc.createIfcLocalPlacement(None, placement)
            elem.ObjectPlacement = local_pl

            # Property set
            pset = ifcopenshell.api.run("pset.add_pset", ifc, product=elem,
                                         name="TunnelSectionProperties")
            props = {}
            if np.isfinite(sec.chainage):     props["Chainage_m"]        = float(sec.chainage)
            if np.isfinite(sec.H1):           props["ClearHeight_H1_m"]  = float(sec.H1)
            if np.isfinite(sec.W1):           props["ClearWidth_W1_m"]   = float(sec.W1)
            if np.isfinite(sec.ovality):      props["Ovality_pct"]       = float(sec.ovality)
            if np.isfinite(sec.eccentricity): props["Eccentricity_mm"]   = float(sec.eccentricity)
            if np.isfinite(sec.radius_fit):   props["RadiusFit_m"]       = float(sec.radius_fit)
            props["ClearanceViolation"] = bool(sec.clearance_violation)
            props["ElementClass"]       = ifc_class
            if np.isfinite(sec.min_clearance_dist):
                props["MinClearance_m"] = float(sec.min_clearance_dist)
            ifcopenshell.api.run("pset.edit_pset", ifc, pset=pset, properties=props)
            ifcopenshell.api.run("spatial.assign_container", ifc,
                                  products=[elem], relating_structure=storey)

        # Global parameters on project
        params = context.parameters
        if params:
            pset_g = ifcopenshell.api.run("pset.add_pset", ifc, product=project,
                                           name="TunnelGlobalParameters")
            clean = {k: float(v) for k, v in params.items()
                     if isinstance(v, (int, float)) and np.isfinite(float(v))}
            if clean:
                ifcopenshell.api.run("pset.edit_pset", ifc, pset=pset_g, properties=clean)

        ifc.write(str(path))
        return str(path)

    def _make_section_solid(self, ifc, body_ctx, sec: "SectionGeometry",
                             profile: str) -> object:
        """Create IfcExtrudedAreaSolid for a tunnel cross-section.

        Circle profile  -> IfcCircleHollowProfileDef (lining ring)
        Box profile     -> IfcRectangleHollowProfileDef
        Extrusion depth = 1 ring width (1.2 m default per Korean standard)
        """
        try:
            extrusion_depth = 1.2

            origin_2d  = ifc.createIfcCartesianPoint((0.0, 0.0))
            axis_2d    = ifc.createIfcAxis2Placement2D(origin_2d, None)

            if profile == "Circle":
                r = float(sec.radius_fit) if np.isfinite(sec.radius_fit) else float(sec.W1 / 2.0) if np.isfinite(sec.W1) else 3.0
                r = float(np.clip(r, 1.0, 15.0))
                thickness = max(0.25, r * 0.08)
                prof = ifc.createIfcCircleHollowProfileDef(
                    "AREA", None, axis_2d, r, thickness)
            else:
                w = float(sec.W1) if np.isfinite(sec.W1) else 6.0
                h = float(sec.H1) if np.isfinite(sec.H1) else 4.5
                w = float(np.clip(w, 1.0, 20.0))
                h = float(np.clip(h, 1.0, 20.0))
                thickness = 0.35
                prof = ifc.createIfcRectangleHollowProfileDef(
                    "AREA", None, axis_2d, w, h, thickness, None, None)

            origin_3d  = ifc.createIfcCartesianPoint((0.0, 0.0, 0.0))
            z_axis     = ifc.createIfcDirection((0.0, 0.0, 1.0))
            x_axis     = ifc.createIfcDirection((1.0, 0.0, 0.0))
            placement  = ifc.createIfcAxis2Placement3D(origin_3d, z_axis, x_axis)
            direction  = ifc.createIfcDirection((0.0, 0.0, 1.0))

            solid = ifc.createIfcExtrudedAreaSolid(prof, placement, direction, extrusion_depth)
            return solid

        except Exception:
            return None
