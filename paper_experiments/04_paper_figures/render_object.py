"""Textured renders of an ObjectFolder mesh, optionally with touch locations marked.

The teaser and the method figure only ever show the object through the tactile
sensor's own narrow window. A picture of the whole object, with its real texture,
gives the reader something to anchor on -- and marking where the reference and
query touches sit turns it into a small map of what the figure is doing.

    python render_object.py --obj 951 --views 6
    python render_object.py --obj 951 --mark_ref 7 --mark_query_ply <shifted>/contact_points.ply

Output: log/paper_job04_paper_figures/object_renders/{obj}_view{k}.png plus a
contact sheet of all the views.
"""
import argparse
import os

import cv2
import numpy as np
import trimesh

# Taxim's renderer needs GLX on this machine, not EGL -- leave PYOPENGL_PLATFORM
# alone (the _run.sh scripts set egl, which fails here).
import pyrender

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OBJ_DIR = f"{ROOT}/Taxim/data/ObjectFolder"
QUERY_PTS = f"{ROOT}/Taxim/results/object_folder_touch_query"
REF_PTS = f"{ROOT}/Taxim/results/object_folder_touch"
OUT = f"{ROOT}/log/paper_job04_paper_figures/object_renders"

C_REF = (0.12, 0.44, 0.71)      # same blue as the "given" row of the teaser
C_QRY = (0.76, 0.27, 0.16)      # same red as the "predicted" row


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """Camera-to-world matrix for a camera at `eye` looking at `target`."""
    f = target - eye
    f = f / np.linalg.norm(f)
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye
    return m


def texture_path(obj):
    """The albedo map of an ObjectFolder object, wherever the .mtl thinks it is."""
    mtl = f"{OBJ_DIR}/{obj}/model.mtl"
    name = None
    if os.path.exists(mtl):
        for line in open(mtl):
            if line.strip().startswith("map_Kd"):
                name = line.split(None, 1)[1].strip()
    if name is None:
        return None
    for cand in (f"{OBJ_DIR}/{obj}/{name}", f"{OBJ_DIR}/{obj}/textures/{name}"):
        if os.path.exists(cand):
            return cand
    return None


def marker(point, radius, color):
    s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    s.apply_translation(point)
    s.visual.vertex_colors = np.tile(
        (np.array([*color, 1.0]) * 255).astype(np.uint8), (len(s.vertices), 1))
    return pyrender.Mesh.from_trimesh(s, smooth=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=int, default=951)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--elev", type=float, default=12.0, help="camera elevation, degrees")
    ap.add_argument("--start_az", type=float, default=0.0)
    ap.add_argument("--size", type=int, nargs=2, default=[900, 900])
    ap.add_argument("--margin", type=float, default=1.12,
                    help="how much room to leave around the object (1.0 = tight fit)")
    ap.add_argument("--untextured", action="store_true",
                    help="render plain grey instead of the object's albedo")
    ap.add_argument("--mark_ref", type=int, nargs="*", default=None,
                    help="reference touch indices to mark (from the object's own point set)")
    ap.add_argument("--mark_query", type=int, nargs="*", default=None,
                    help="query touch indices to mark")
    ap.add_argument("--mark_query_ply", default=None,
                    help="mark points of this .ply instead (e.g. moved query points)")
    ap.add_argument("--mark_query_ply_idx", type=int, nargs="*", default=None,
                    help="which points of --mark_query_ply to mark (default: all)")
    ap.add_argument("--marker_scale", type=float, default=0.012,
                    help="marker radius, as a fraction of the object's bounding-box diagonal")
    ap.add_argument("--focus_ply", default=None,
                    help="centre the camera on a point of this .ply instead of the whole "
                         "object (use with --focus_idx and --focus_radius)")
    ap.add_argument("--focus_idx", type=int, default=0)
    ap.add_argument("--focus_radius", type=float, default=0.25,
                    help="with --focus_ply: how much of the object to keep in frame, "
                         "as a fraction of its radius")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    loaded = trimesh.load(f"{OBJ_DIR}/{args.obj}/model.obj", process=False)
    geoms = (list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene)
             else [loaded])
    # ObjectFolder's .mtl names its maps without the textures/ directory they
    # actually live in, so trimesh silently falls back to a flat grey. Attach the
    # albedo by hand instead.
    albedo = texture_path(args.obj)
    if albedo and not args.untextured:
        from PIL import Image
        img = Image.open(albedo).convert("RGB")
        for g in geoms:
            uv = getattr(g.visual, "uv", None)
            if uv is None:
                print(f"  (no texture coordinates on {g.metadata.get('name', 'mesh')})")
                continue
            g.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
        print("textured with", os.path.basename(albedo))
    bounds = np.vstack([g.bounds for g in geoms])
    centre = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    diag = float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0)))
    radius = diag / 2

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.45] * 3)
    for g in geoms:
        scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))

    # touch markers
    def points_of(ply, idxs):
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(ply).points)
        return pts if idxs is None else pts[list(idxs)]

    rad = args.marker_scale * diag
    if args.mark_ref:
        for p in points_of(f"{REF_PTS}/{args.obj}/picked_points_fps.ply", args.mark_ref):
            scene.add(marker(p, rad, C_REF))
    if args.mark_query:
        for p in points_of(f"{QUERY_PTS}/{args.obj}/picked_points_query.ply", args.mark_query):
            scene.add(marker(p, rad, C_QRY))
    if args.mark_query_ply:
        for p in points_of(args.mark_query_ply, args.mark_query_ply_idx):
            scene.add(marker(p, rad, C_QRY))

    if args.focus_ply:
        import open3d as o3d
        centre = np.asarray(o3d.io.read_point_cloud(args.focus_ply).points)[args.focus_idx]
        radius = radius * args.focus_radius
        print(f"framing on {centre.round(3)}, {args.focus_radius:.0%} of the object radius")

    yfov = np.deg2rad(35.0)
    cam = pyrender.PerspectiveCamera(yfov=yfov)
    cam_node = scene.add(cam, pose=np.eye(4))
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    key_node = scene.add(key, pose=np.eye(4))

    r = pyrender.OffscreenRenderer(args.size[0], args.size[1])
    # far enough that the object's bounding sphere fits the (square) frame
    dist = args.margin * radius / np.sin(yfov / 2)
    shots = []
    for k in range(args.views):
        az = np.deg2rad(args.start_az + 360.0 * k / args.views)
        el = np.deg2rad(args.elev)
        eye = centre + dist * np.array([np.cos(el) * np.sin(az),
                                        np.sin(el),
                                        np.cos(el) * np.cos(az)])
        pose = look_at(eye, centre)
        scene.set_pose(cam_node, pose)
        scene.set_pose(key_node, pose)      # light follows the camera
        colour, _ = r.render(scene)
        path = f"{args.out}/{args.obj}_view{k}.png"
        cv2.imwrite(path, cv2.cvtColor(colour, cv2.COLOR_RGB2BGR))
        shots.append(colour)
        print("wrote", path)
    r.delete()

    thumbs = [cv2.cvtColor(cv2.resize(s, (300, 300)), cv2.COLOR_RGB2BGR) for s in shots]
    rows = [np.hstack(thumbs[i:i + 3]) for i in range(0, len(thumbs), 3)]
    w = max(x.shape[1] for x in rows)
    sheet = np.vstack([np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0)),
                              constant_values=255) for x in rows])
    cv2.imwrite(f"{args.out}/{args.obj}_sheet.png", sheet)
    print("wrote", f"{args.out}/{args.obj}_sheet.png")


if __name__ == "__main__":
    main()
