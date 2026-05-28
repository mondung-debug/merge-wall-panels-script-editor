# -*- coding: utf-8 -*-
"""
Merge Wall Panels — Script Editor
Merges wall panel tiles into a single lightweight UsdGeom.Mesh by
selecting faces whose normal matches FACE_NORMAL and triangulating them.
Works with walls that have uneven depths (protrusions/recesses).

Usage:
    1. Select a component prim in USD Composer
    2. Set FACE_NORMAL to match the wall's outward direction
    3. Run script

Config:
    FACE_NORMAL        — Outward face direction: "Z+" "Z-" "X+" "X-" "Y+" "Y-"
    NORMAL_THRESHOLD   — Dot product threshold for face selection (0.9 = within ~25 deg)
    FILTER_MODE        — "all_mesh" / "name" / "metadata"
    RESULT_PRIM_NAME   — Output mesh prim name
    DEACTIVATE_ORIGINAL — Deactivate original panels after merge
    OFFSET             — Offset along FACE_NORMAL direction (0 = original position)
"""

import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, Vt, Sdf
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
# FACE_NORMAL: outward direction of the surface to extract
#   "Z+"  — ceiling/top face  (same as floor tiles but from above)
#   "Z-"  — floor/bottom face
#   "X+"  — wall facing +X
#   "X-"  — wall facing -X
#   "Y+"  — wall facing +Y
#   "Y-"  — wall facing -Y
FACE_NORMAL        = "X+"

NORMAL_THRESHOLD   = 0.85   # min dot product with target normal (cos ~32 deg)
FILTER_MODE        = "all_mesh"

FLOOR_CATEGORIES   = {"Walls", "Curtain Panels", "Curtain Wall Panels"}
FLOOR_FAMILY_NAMES = {"Basic Wall", "System Panel", "Curtain Wall Panel"}
ATTR_CATEGORY      = "omni:hoops:metadata:Other:Category"
ATTR_FAMILY_NAME   = "omni:hoops:metadata:Other:tn__FamilyName_mA"

PANEL_MESH_NAMES   = {"polySurface1"}

RESULT_PRIM_NAME   = "WallPlane_Merged"
DEACTIVATE_ORIGINAL = True
OFFSET             = 0.0    # offset along FACE_NORMAL direction
WELD_TOLERANCE     = 1e-3
# ─────────────────────────────────────────────────────────────────────────────

_NORMAL_DIRECTIONS = {
    "X+": np.array([ 1.0, 0.0, 0.0]),
    "X-": np.array([-1.0, 0.0, 0.0]),
    "Y+": np.array([ 0.0, 1.0, 0.0]),
    "Y-": np.array([ 0.0,-1.0, 0.0]),
    "Z+": np.array([ 0.0, 0.0, 1.0]),
    "Z-": np.array([ 0.0, 0.0,-1.0]),
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
    return (cat and cat in FLOOR_CATEGORIES) or (fam and fam in FLOOR_FAMILY_NAMES)


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


def _world_mesh(mesh_prim):
    mesh = UsdGeom.Mesh(mesh_prim)
    pts_attr = mesh.GetPointsAttr()
    fvc_attr = mesh.GetFaceVertexCountsAttr()
    fvi_attr = mesh.GetFaceVertexIndicesAttr()
    if not (pts_attr and pts_attr.HasValue() and
            fvc_attr and fvc_attr.HasValue() and
            fvi_attr and fvi_attr.HasValue()):
        return None, None, None
    pts = np.array(pts_attr.Get(), dtype=np.float64)
    fvc = list(fvc_attr.Get())
    fvi = list(fvi_attr.Get())
    mat = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    m   = np.array(mat)
    pts = np.hstack([pts, np.ones((len(pts), 1))]) @ m   # USD row-vector convention
    return pts[:, :3], fvc, fvi


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


def _face_normal(pts, face_verts):
    """Compute unit face normal using Newell's method (robust for N-gons)."""
    n = np.zeros(3)
    nv = len(face_verts)
    for i in range(nv):
        cur = pts[face_verts[i]]
        nxt = pts[face_verts[(i + 1) % nv]]
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
    length = np.linalg.norm(n)
    if length < 1e-10:
        return np.zeros(3)
    return n / length


def _project_to_2d(pts3d, face_normal_key):
    """Project 3D points to 2D plane perpendicular to FACE_NORMAL."""
    ax = face_normal_key[0]  # 'X', 'Y', or 'Z'
    if ax == "X":
        return pts3d[:, [1, 2]]   # YZ plane
    elif ax == "Y":
        return pts3d[:, [0, 2]]   # XZ plane
    else:
        return pts3d[:, [0, 1]]   # XY plane


def _ray_cast_2d(pt, poly_pts):
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly_pts)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly_pts[i][0]), float(poly_pts[i][1])
        xj, yj = float(poly_pts[j][0]), float(poly_pts[j][1])
        if (yi > y) != (yj > y):
            denom = yj - yi
            if abs(denom) > 1e-300 and x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def _triangulate_delaunay(face_global, pts2d):
    """Delaunay triangulation filtered by point-in-polygon."""
    try:
        from scipy.spatial import Delaunay as _Delaunay
    except ImportError:
        return []
    if len(face_global) < 3:
        return []
    local_pts = pts2d[face_global]
    try:
        tri = _Delaunay(local_pts)
    except Exception:
        return []
    result = []
    for simp in tri.simplices:
        centroid = local_pts[simp].mean(axis=0)
        if _ray_cast_2d(centroid, local_pts):
            result.append((face_global[simp[0]], face_global[simp[1]], face_global[simp[2]]))
    return result


def _triangulate_target_faces(meshes_data, global_mapping, offset_list,
                               pts2d, target_normal, threshold):
    """Select faces whose normal aligns with target_normal, then triangulate."""
    all_triangles = []
    face_count = 0

    for (pts_w, fvc, fvi), off in zip(meshes_data, offset_list):
        if pts_w is None:
            continue

        vi_idx = 0
        for fc in fvc:
            face_local = [fvi[vi_idx + k] for k in range(fc)]

            if fc >= 3:
                n = _face_normal(pts_w, face_local)
                if np.dot(n, target_normal) >= threshold:
                    face_global = [global_mapping[off + v] for v in face_local]
                    face_count += 1

                    if fc == 3:
                        all_triangles.append(tuple(face_global))
                    elif fc == 4:
                        all_triangles.append((face_global[0], face_global[1], face_global[2]))
                        all_triangles.append((face_global[0], face_global[2], face_global[3]))
                    else:
                        tris = _triangulate_delaunay(face_global, pts2d)
                        if not tris:
                            tris = [(face_global[0], face_global[i], face_global[i+1])
                                    for i in range(1, fc - 1)]
                        all_triangles.extend(tris)

            vi_idx += fc

    print(f"[MergeWallPanels] Target faces found: {face_count}")
    return all_triangles


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    ctx   = omni.usd.get_context()
    stage = ctx.get_stage()

    if FACE_NORMAL not in _NORMAL_DIRECTIONS:
        print(f"[MergeWallPanels] ERROR: Invalid FACE_NORMAL '{FACE_NORMAL}'. "
              f"Use one of: {list(_NORMAL_DIRECTIONS.keys())}")
        return

    target_normal = _NORMAL_DIRECTIONS[FACE_NORMAL]

    selection = ctx.get_selection().get_selected_prim_paths()
    if not selection:
        print("[MergeWallPanels] Please select a component prim first.")
        return

    for root_path in selection:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            print(f"[MergeWallPanels] Invalid prim: {root_path}")
            continue

        print(f"[MergeWallPanels] Processing: {root_path}")
        print(f"[MergeWallPanels] Target face normal: {FACE_NORMAL}")

        panel_meshes = _collect_panel_meshes(root_prim)
        if not panel_meshes:
            print(f"[MergeWallPanels] No panel meshes found under: {root_path}")
            continue

        print(f"[MergeWallPanels] Found {len(panel_meshes)} panel mesh(es)")

        meshes_data = [_world_mesh(m) for m in panel_meshes]

        valid_pts = [d[0] for d in meshes_data if d[0] is not None]
        if not valid_pts:
            print("[MergeWallPanels] No valid mesh data found.")
            continue

        merged_pts, global_mapping = _merge_vertices(valid_pts)
        print(f"[MergeWallPanels] Merged vertices: {len(merged_pts)}")

        offsets = []
        off = 0
        for d in meshes_data:
            offsets.append(off)
            if d[0] is not None:
                off += len(d[0])

        # Project to 2D plane perpendicular to FACE_NORMAL
        pts2d = _project_to_2d(merged_pts, FACE_NORMAL)

        all_triangles = _triangulate_target_faces(
            meshes_data, global_mapping, offsets,
            pts2d, target_normal, NORMAL_THRESHOLD)

        if not all_triangles:
            print(f"[MergeWallPanels] ERROR: No triangles produced. "
                  f"Check FACE_NORMAL='{FACE_NORMAL}' or NORMAL_THRESHOLD={NORMAL_THRESHOLD}.")
            continue

        print(f"[MergeWallPanels] Triangles: {len(all_triangles)}")

        # Compact: only include vertices referenced by triangles
        used_set   = sorted({v for tri in all_triangles for v in tri})
        vert_remap = {old: new for new, old in enumerate(used_set)}
        compact_pts = merged_pts[used_set]
        remapped_tris = [(vert_remap[a], vert_remap[b], vert_remap[c])
                         for a, b, c in all_triangles]

        # Apply offset along FACE_NORMAL
        pts3d_result = compact_pts.copy()
        if OFFSET != 0.0:
            pts3d_result += target_normal * OFFSET

        # Create USD Mesh
        result_path = f"{root_path}/{RESULT_PRIM_NAME}"
        if stage.GetPrimAtPath(result_path):
            stage.RemovePrim(result_path)

        result_mesh = UsdGeom.Mesh.Define(stage, result_path)
        result_mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts3d_result]))
        result_mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray([3] * len(remapped_tris)))
        result_mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray([v for tri in remapped_tris for v in tri]))
        result_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        result_mesh.CreateDoubleSidedAttr().Set(True)

        lo = Gf.Vec3f(*pts3d_result.min(axis=0))
        hi = Gf.Vec3f(*pts3d_result.max(axis=0))
        result_mesh.GetExtentAttr().Set(Vt.Vec3fArray([lo, hi]))

        if DEACTIVATE_ORIGINAL:
            for m in panel_meshes:
                m.SetActive(False)
            print(f"[MergeWallPanels] Deactivated {len(panel_meshes)} original panel(s)")

        stage.GetRootLayer().Save()
        print(f"[MergeWallPanels] Done -> {result_path}")
        print(f"  Vertices: {len(pts3d_result)}, Triangles: {len(remapped_tris)}")


run()
