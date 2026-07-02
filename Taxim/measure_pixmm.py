import cv2
import numpy as np
import sys

image_path = sys.argv[1]
real_length_mm = float(sys.argv[2])

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(image_path)

points = []

def mouse_callback(event, x, y, flags, param):
    global points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked point {len(points)}: x={x}, y={y}")

        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        if len(points) == 2:
            p1 = np.array(points[0])
            p2 = np.array(points[1])
            pixel_length = np.linalg.norm(p2 - p1)
            pixmm = real_length_mm / pixel_length

            print("\nResult")
            print(f"pixel length = {pixel_length:.3f} px")
            print(f"real length   = {real_length_mm:.3f} mm")
            print(f"pixmm         = {pixmm:.6f} mm/px")
            print(f"px per mm     = {1.0 / pixmm:.3f} px/mm")

            cv2.line(img, points[0], points[1], (0, 255, 0), 2)

        cv2.imshow("measure", img)

cv2.imshow("measure", img)
cv2.setMouseCallback("measure", mouse_callback)

print("Click two endpoints of a known physical length.")
print("Press ESC to quit.")

while True:
    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyAllWindows()
