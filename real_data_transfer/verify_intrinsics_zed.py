"""
Intrinsics verification for ZED 2i.

Two checks side by side in one cv2 window:

  Left panel — Reprojection check
    Green dots : 2D corners detected by ARuCO
    Red  dots  : 3D corners (from estimatePoseSingleMarkers) re-projected back
                 to 2D with the same camera_matrix.
    If they overlap, intrinsics are self-consistent.

  Right panel — Depth vs pose check
    For each detected marker, reads the ZED depth at the marker centre pixel
    and prints the ARuCO tvec Z.  If intrinsics and depth scale are both
    correct these two Z values should agree within a few mm.

Press 'q' to quit.
"""

import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from camera_utils import ZEDCamera

ARUCO_DICT = cv2.aruco.DICT_4X4_50
DISPLAY_W, DISPLAY_H = 640, 360


def main():
    parser = argparse.ArgumentParser(description="ZED intrinsics verifier")
    parser.add_argument("--marker_size", type=float, default=0.092,
                        help="Printed ARuCO marker side length in metres (measure with a ruler)")
    parser.add_argument("--depth_mode",
                        choices=["performance", "quality", "ultra", "neural", "neural_plus"],
                        default="neural_plus")
    args = parser.parse_args()

    MARKER_SIZE = args.marker_size
    cam = ZEDCamera(depth_mode=args.depth_mode)
    intr = cam.intrinsics
    camera_matrix = intr["camera_matrix"]
    dist_coeffs   = intr["dist_coeffs"]   # zeros for ZED (pre-rectified)

    print(f"Intrinsics: fx={intr['fx']:.1f}  fy={intr['fy']:.1f}  "
          f"cx={intr['cx']:.1f}  cy={intr['cy']:.1f}  "
          f"{intr['width']}x{intr['height']}")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params     = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector   = cv2.aruco.ArucoDetector(dictionary, params)

    try:
        while True:
            if not cam.grab():
                continue

            color_bgr = cam.color_bgr.copy()
            depth_mm  = cam.depth_mm

            gray    = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            left  = color_bgr.copy()   # reprojection panel
            right = color_bgr.copy()   # depth-vs-pose panel

            if ids is not None and len(ids) > 0:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_SIZE, camera_matrix, dist_coeffs
                )

                h_m = MARKER_SIZE / 2.0
                local_corners_3d = np.array([
                    [-h_m,  h_m, 0],
                    [ h_m,  h_m, 0],
                    [ h_m, -h_m, 0],
                    [-h_m, -h_m, 0],
                ], dtype=np.float32)

                for i, mid in enumerate(ids.flatten()):
                    rvec = rvecs[i].flatten()
                    tvec = tvecs[i].flatten()

                    # ── Left panel: reprojection check ────────────────────
                    # Detected 2D corners (green)
                    det_corners = corners[i].reshape(4, 2).astype(np.int32)
                    for pt in det_corners:
                        cv2.circle(left, tuple(pt), 5, (0, 255, 0), -1)

                    # Reprojected corners from 3D pose (red)
                    proj, _ = cv2.projectPoints(
                        local_corners_3d, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    proj = proj.reshape(4, 2).astype(np.int32)
                    for pt in proj:
                        cv2.circle(left, tuple(pt), 3, (0, 0, 255), -1)

                    # Axis overlay
                    cv2.drawFrameAxes(left, camera_matrix, dist_coeffs,
                                      rvec, tvec, MARKER_SIZE * 0.5)

                    # ── Right panel: depth vs pose check ──────────────────
                    cx_px, cy_px = det_corners.mean(axis=0).astype(int)
                    cx_px = np.clip(cx_px, 0, depth_mm.shape[1] - 1)
                    cy_px = np.clip(cy_px, 0, depth_mm.shape[0] - 1)

                    depth_z_mm = int(depth_mm[cy_px, cx_px])
                    aruco_z_mm = int(tvec[2] * 1000)
                    aruco_dist_mm = int(np.linalg.norm(tvec) * 1000)

                    color = (0, 255, 0) if abs(depth_z_mm - aruco_z_mm) < 20 else (0, 100, 255)
                    for j, line in enumerate([
                        f"ID{mid}",
                        f"depth   = {depth_z_mm} mm",
                        f"ARuCO_Z = {aruco_z_mm} mm",
                        f"ARuCO_d = {aruco_dist_mm} mm",
                    ]):
                        cv2.putText(right, line,
                                    (cx_px - 80, cy_px - 10 + j * 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(right, (cx_px, cy_px), 6, color, -1)

            cv2.putText(left, "Green=detected  Red=reprojected",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(right, "depth at centre vs ARuCO tvec Z",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            left_s  = cv2.resize(left,  (DISPLAY_W, DISPLAY_H))
            right_s = cv2.resize(right, (DISPLAY_W, DISPLAY_H))
            cv2.imshow("Intrinsics verification  (q=quit)", np.hstack([left_s, right_s]))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
