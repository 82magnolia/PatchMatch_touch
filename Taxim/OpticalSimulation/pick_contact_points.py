import argparse
import os

import numpy as np
import open3d as o3d
import pyrender
import pyglet.window.key
import pyglet.window.mouse
import trimesh
from pyrender.renderer import TextAlign


class PointPickerViewer(pyrender.Viewer):
    """Interactive viewer for picking 3D contact points on a mesh.

    Controls:
      Left drag         : Rotate around centroid
      Right drag/Scroll : Zoom
      Middle drag       : Pan
      Shift + Click     : Pick point on mesh surface
      S                 : Save picked points and quit
      Q                 : Save picked points and quit
      [Stop Selection]  : On-screen button — save and quit
    """

    # Screen-space button region (pixels from bottom-left)
    BUTTON_X = 10
    BUTTON_Y = 10
    BUTTON_W = 170
    BUTTON_H = 32

    def __init__(self, scene, tr_mesh, intersector, output_ply, centroid_offset):
        # Store all attributes BEFORE super().__init__, which starts the event loop
        self.tr_mesh = tr_mesh
        self.intersector = intersector
        self.output_ply = output_ply
        self.centroid_offset = centroid_offset  # added back when saving
        self._picked_points = []
        self._marker_nodes = []

        print("\nControls:")
        print("  Left drag          : Rotate around centroid")
        print("  Right drag / Scroll: Zoom")
        print("  Middle drag        : Pan")
        print("  Shift + Click      : Pick point on mesh")
        print("  S / Q              : Save & quit")
        print("  On-screen button   : Stop Selection (save & quit)\n")

        super().__init__(
            scene,
            viewer_flags={
                "use_raymond_lighting": True,
                "window_title": "Pick Contact Points",
            },
        )

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------

    def on_mouse_press(self, x, y, buttons, modifiers):
        # "Stop Selection" button
        if (self.BUTTON_X <= x <= self.BUTTON_X + self.BUTTON_W and
                self.BUTTON_Y <= y <= self.BUTTON_Y + self.BUTTON_H):
            self._save_and_exit()
            return

        # Shift + left-click → pick a surface point
        if (buttons == pyglet.window.mouse.LEFT and
                (modifiers & pyglet.window.key.MOD_SHIFT)):
            self._pick_point(x, y)
            return

        # All other clicks: default rotation / pan / zoom
        super().on_mouse_press(x, y, buttons, modifiers)

    def on_key_press(self, symbol, modifiers):
        if symbol in (pyglet.window.key.S, pyglet.window.key.Q):
            self._save_and_exit()
            return
        super().on_key_press(symbol, modifiers)

    def on_draw(self):
        super().on_draw()
        if self._renderer is None:
            return
        # "Stop Selection" button text
        self._renderer.render_text(
            "[ Stop Selection ]",
            self.BUTTON_X + self.BUTTON_W // 2,
            self.BUTTON_Y + 8,
            font_pt=16,
            color=np.array([1.0, 0.35, 0.35, 1.0]),
            align=TextAlign.BOTTOM_CENTER,
        )
        # Picked-point counter
        self._renderer.render_text(
            f"Points: {len(self._picked_points)}",
            self.BUTTON_X + self.BUTTON_W + 12,
            self.BUTTON_Y + 8,
            font_pt=16,
            color=np.array([1.0, 1.0, 0.3, 1.0]),
            align=TextAlign.BOTTOM_LEFT,
        )

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def _unproject_ray(self, x, y):
        """Convert screen pixel (x, y) to a world-space ray (origin, direction).

        pyglet and OpenGL share the convention that y=0 is at the bottom of the
        window, so no y-flip is needed here.
        """
        w, h = self._viewport_size
        ndc_x = (2.0 * x / w) - 1.0
        ndc_y = (2.0 * y / h) - 1.0

        cam_node = self._camera_node
        proj = cam_node.camera.get_projection_matrix(w, h)
        proj_inv = np.linalg.inv(proj)

        # NDC → view space
        # proj_inv @ [ndc_x, ndc_y, -1, 1] followed by perspective divide gives
        # the view-space position of the near-plane point. The direction from the
        # camera (at view-space origin) to that point is p_view[:3].
        p_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
        p_view = proj_inv @ p_ndc
        p_view /= p_view[3]   # perspective divide → p_view[:3] is the direction
        p_view[3] = 0.0       # w=0: only rotation applied below, no translation

        # View → world
        cam_pose = cam_node.matrix  # camera-to-world (4×4)
        direction = (cam_pose @ p_view)[:3]
        direction /= np.linalg.norm(direction)
        origin = cam_pose[:3, 3]
        return origin, direction

    def _pick_point(self, x, y):
        origin, direction = self._unproject_ray(x, y)
        locations, _, _ = self.intersector.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
        )
        if len(locations) == 0:
            return
        # Take the closest intersection to the camera
        dists = np.linalg.norm(locations - origin, axis=1)
        pt = locations[np.argmin(dists)]
        self._picked_points.append(pt)
        self._add_sphere_marker(pt)
        print(f"[{len(self._picked_points)}] Picked: {pt}")

    def _add_sphere_marker(self, pt):
        r = self._scene.scale * 0.015
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=r)
        sphere.apply_translation(pt)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 0.15, 0.15, 1.0])
        node = self._scene.add(
            pyrender.Mesh.from_trimesh(sphere, material=mat, smooth=False))
        self._marker_nodes.append(node)

    # ------------------------------------------------------------------
    # Save & exit
    # ------------------------------------------------------------------

    def _save_and_exit(self):
        if self._picked_points:
            pcd = o3d.geometry.PointCloud()
            pts = np.array(self._picked_points) + self.centroid_offset
            pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(self.output_ply, pcd)
            print(f"Saved {len(self._picked_points)} point(s) → {self.output_ply}")
        else:
            print("No points were picked.")
        self.close()


def main():
    parser = argparse.ArgumentParser(
        description="Interactively pick contact points on a .obj mesh and save to .ply")
    parser.add_argument("--obj_path", required=True, help="Path to .obj file")
    parser.add_argument("--output_ply", default="./picked_points.ply",
                        help="Output .ply file path (default: ./picked_points.ply)")
    args = parser.parse_args()

    tr_mesh = trimesh.load(args.obj_path, force="mesh", process=False)
    if not isinstance(tr_mesh, trimesh.Trimesh):
        raise ValueError("Could not load a single Trimesh from the .obj file.")

    # Center mesh at origin so rotation orbits the centroid
    centroid = tr_mesh.centroid.copy()
    tr_mesh.apply_translation(-centroid)

    scene = pyrender.Scene(
        bg_color=[0.15, 0.15, 0.15, 1.0],
        ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(tr_mesh, smooth=False))

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(tr_mesh)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_ply)), exist_ok=True)

    # Launching the Viewer blocks until the window is closed
    PointPickerViewer(scene, tr_mesh, intersector, args.output_ply, centroid)


if __name__ == "__main__":
    main()
