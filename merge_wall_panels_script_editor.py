# -*- coding: utf-8 -*-
"""
Merge Wall Panels — Script Editor (BBox mode)
Each panel's AABB U/V extents are projected onto a single shared plane
at the global face position in the FACE_NORMAL direction.
All quads are vertex-welded and triangulated into a single UsdGeom.Mesh.
Holes are intentionally omitted — output is a solid wall plane.

Usage:
    1. Select a component prim in USD Composer
    2. Set FACE_NORMAL to match the wall's inward-facing direction
    3. Run script

Config:
    FACE_NORMAL        — "X+" / "X-" / "Y+" / "Y-" / "Z+" / "Z-"
    FILTER_MODE        — "all_mesh" / "name" / "metadata"
    WALL_CATEGORIES    — Category values to match (FILTER_MODE="metadata")
    WALL_FAMILY_NAMES  — FamilyName values to match (FILTER_MODE="metadata")
    RESULT_PRIM_NAME   — Output mesh prim name
    ORIGINAL_ACTION    — "deactivate" / "delete" / "none"
    PLANE_OFFSET       — Additional offset along FACE_NORMAL direction
    MIN_PANEL_EXTENT   — Minimum panel size along FACE_NORMAL axis (filters edge-on panels)
"""

import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, Vt

# ── Config ───────────────────────────────────────────────────────────────────
FACE_NORMAL        = "X+"   # "X+" / "X-" / "Y+" / "Y-" / "Z+" / "Z-"

FILTER_MODE        = "metadata"

WALL_CATEGORIES    = {"Curtain Panels", "Walls", "Floors"}
WALL_FAMILY_NAMES  = {"System Panel", "Access Floor Panel", "Basic Wall", "Floor"}
ATTR_CATEGORY      = "omni:hoops:metadata:Other:Category"
ATTR_FAMILY_NAME   = "omni:hoops:metadata:Other:tn__FamilyName_mA"

PANEL_MESH_NAMES   = {"polySurface1"}

RESULT_PRIM_NAME   = "WallPlane_Merged"
ORIGINAL_ACTION    = "deactivate"   # "deactivate" / "delete" / "none"
PLANE_OFFSET       = 0.0
WELD_TOLERANCE     = 1e-3
MIN_PANEL_EXTENT   = 0.01  # panels thinner than this along FACE_NORMAL are skipped
# ─────────────────────────────────────────────────────────────────────────────

_NORMALS = {
    "X+": ( 0, 1, 2, +1),  # axis_idx, u_idx, v_idx, sign
    "X-": ( 0, 1, 2, -1),
    "Y+": ( 1, 0, 2, +1),
    "Y-": ( 1, 0, 2, -1),
    "Z+": ( 2, 0, 1, +1),
    "Z-": ( 2, 0, 1, -1),
}


def _get_attr(prim, attr_name):
    attr = prim.GetAttribute(attr_name)
    if attr and attr.HasValue():
        v = attr.Get()
        return str(v) if v is not None else None
    return None


def _is_panel(prim):
    cat = _get_attr(prim, ATTR_CATEGORY)
    fam = _get_attr(prim, ATTR_FAMILY_NAME)
    return (cat and cat in WALL_CATEGORIES) or (fam and fam in WALL_FAMILY_NAMES)


def _collect_panel_meshes(root_prim):
    result = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        if FILTER_MODE == "all_mesh":
            result.append(prim)
        elif FILTER_MODE == "name":
            if prim.GetName() in PANEL_MESH_NAMES:
                result.append(prim)
        elif FILTER_MODE == "metadata":
            if _is_panel(prim):
                result.append(prim)
                continue
            cur = prim.GetParent()
            while cur and cur.IsValid() and cur.GetPath() != root_prim.GetPath():
                if _is_panel(cur):
                    result.append(prim)
                    break
                cur = cur.GetParent()
    return result


def _get_world_bbox(mesh_prim):
    """Return world AABB as (min_pt, max_pt) numpy arrays, or (None, None)."""
    mesh = UsdGeom.Mesh(mesh_prim)
    pts_attr = mesh.GetPointsAttr()
    if not (pts_attr and pts_attr.HasValue()):
        return None, None

    pts = np.array(pts_attr.Get(), dtype=np.float64)
    mat = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    m   = np.array(mat)
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ m   # USD row-vector convention
    pts = pts[:, :3]

    return pts.min(axis=0), pts.max(axis=0)


def _build_face_quad(mn, mx, axis_idx, u_idx, v_idx, sign, face_val):
    """Build a CCW face quad at face_val using the panel's U/V AABB extents."""
    u0, u1 = float(mn[u_idx]), float(mx[u_idx])
    v0, v1 = float(mn[v_idx]), float(mx[v_idx])

    def _pt(u, v):
        p = [0.0, 0.0, 0.0]
        p[axis_idx] = face_val
        p[u_idx]    = u
        p[v_idx]    = v
        return p

    if sign > 0:
        quad = [_pt(u0, v0), _pt(u1, v0), _pt(u1, v1), _pt(u0, v1)]
    else:
        quad = [_pt(u1, v0), _pt(u0, v0), _pt(u0, v1), _pt(u1, v1)]

    return np.array(quad, dtype=np.float64)


def _merge_vertices(pts_list, tol=WELD_TOLERANCE):
    all_pts = np.vstack(pts_list)
    mapping = np.arange(len(all_pts))
    unique  = []
    seen    = {}
    for i, p in enumerate(all_pts):
        key = tuple(np.round(p / tol).astype(int))
        if key in seen:
            mapping[i] = seen[key]
        else:
            seen[key] = len(unique)
            mapping[i] = len(unique)
            unique.append(p)
    return np.array(unique), mapping


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    ctx   = omni.usd.get_context()
    stage = ctx.get_stage()

    if FACE_NORMAL not in _NORMALS:
        print(f"[MergeWallPanels] ERROR: Invalid FACE_NORMAL '{FACE_NORMAL}'. "
              f"Use: {list(_NORMALS.keys())}")
        return

    axis_idx, u_idx, v_idx, sign = _NORMALS[FACE_NORMAL]

    selection = ctx.get_selection().get_selected_prim_paths()
    if not selection:
        print("[MergeWallPanels] Please select a component prim first.")
        return

    for root_path in selection:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            print(f"[MergeWallPanels] Invalid prim: {root_path}")
            continue

        print(f"[MergeWallPanels] Processing: {root_path} | FACE_NORMAL={FACE_NORMAL}")

        panel_meshes = _collect_panel_meshes(root_prim)
        if not panel_meshes:
            print(f"[MergeWallPanels] No panel meshes found under: {root_path}")
            continue

        print(f"[MergeWallPanels] Found {len(panel_meshes)} panel mesh(es)")

        # Step 1: collect world AABBs, filter edge-on panels
        bboxes       = []
        valid_meshes = []
        skipped      = 0
        for m in panel_meshes:
            mn, mx = _get_world_bbox(m)
            if mn is None:
                continue
            extent_along_normal = float(mx[axis_idx] - mn[axis_idx])
            if extent_along_normal < MIN_PANEL_EXTENT:
                skipped += 1
                continue
            bboxes.append((mn, mx))
            valid_meshes.append(m)

        if skipped:
            print(f"[MergeWallPanels] Skipped {skipped} edge-on panel(s) "
                  f"(extent < {MIN_PANEL_EXTENT})")

        if not bboxes:
            print("[MergeWallPanels] No valid mesh data found.")
            continue

        # Step 2: compute single shared face plane (global extreme along FACE_NORMAL)
        all_mn = np.array([b[0] for b in bboxes])
        all_mx = np.array([b[1] for b in bboxes])
        if sign > 0:
            global_face_val = float(all_mx[:, axis_idx].max())
        else:
            global_face_val = float(all_mn[:, axis_idx].min())
        global_face_val += PLANE_OFFSET * sign

        print(f"[MergeWallPanels] Shared plane at axis[{axis_idx}] = {global_face_val:.4f}")

        # Step 3: build quads projected onto the shared plane
        quads = [_build_face_quad(mn, mx, axis_idx, u_idx, v_idx, sign, global_face_val)
                 for mn, mx in bboxes]

        # Step 4: weld vertices
        merged_pts, global_mapping = _merge_vertices(quads)
        print(f"[MergeWallPanels] BBox quads: {len(quads)}, Welded vertices: {len(merged_pts)}")

        # Step 5: triangulate each quad (2 triangles, CCW)
        all_triangles = []
        for qi in range(len(quads)):
            base = qi * 4
            v0 = int(global_mapping[base + 0])
            v1 = int(global_mapping[base + 1])
            v2 = int(global_mapping[base + 2])
            v3 = int(global_mapping[base + 3])
            all_triangles.append((v0, v1, v2))
            all_triangles.append((v0, v2, v3))

        print(f"[MergeWallPanels] Triangles: {len(all_triangles)}")

        # Step 6: compact vertices
        used_set   = sorted({v for tri in all_triangles for v in tri})
        vert_remap = {old: new for new, old in enumerate(used_set)}
        compact_pts   = merged_pts[used_set]
        remapped_tris = [(vert_remap[a], vert_remap[b], vert_remap[c])
                         for a, b, c in all_triangles]

        # Step 7: create USD Mesh
        result_path = f"{root_path}/{RESULT_PRIM_NAME}"
        if stage.GetPrimAtPath(result_path):
            stage.RemovePrim(result_path)

        result_mesh = UsdGeom.Mesh.Define(stage, result_path)
        result_mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*p) for p in compact_pts]))
        result_mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray([3] * len(remapped_tris)))
        result_mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray([v for tri in remapped_tris for v in tri]))
        result_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        result_mesh.CreateDoubleSidedAttr().Set(True)

        lo = Gf.Vec3f(*compact_pts.min(axis=0))
        hi = Gf.Vec3f(*compact_pts.max(axis=0))
        result_mesh.GetExtentAttr().Set(Vt.Vec3fArray([lo, hi]))

        if ORIGINAL_ACTION == "deactivate":
            for m in valid_meshes:
                m.SetActive(False)
            print(f"[MergeWallPanels] Deactivated {len(valid_meshes)} original panel(s)")
        elif ORIGINAL_ACTION == "delete":
            for m in valid_meshes:
                stage.RemovePrim(m.GetPath())
            print(f"[MergeWallPanels] Deleted {len(valid_meshes)} original panel(s)")
        else:
            print(f"[MergeWallPanels] Original panels unchanged (ORIGINAL_ACTION='none')")

        stage.GetRootLayer().Save()
        print(f"[MergeWallPanels] Done -> {result_path}")
        print(f"  Vertices: {len(compact_pts)}, Triangles: {len(remapped_tris)}")


run()
