import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def show_depth_map(npy_path):
    if not os.path.exists(npy_path):
        return

    depth_raw = np.load(npy_path)
    
    depth_2d = np.squeeze(depth_raw)
    
    print(f"min: {depth_2d.min():.4f}m ~ max: {depth_2d.max():.4f}m")

    # Normalization
    d_min = depth_2d.min()
    d_max = depth_2d.max()
    
    if d_max - d_min == 0:
        depth_norm = np.zeros_like(depth_2d)
    else:
        depth_norm = (depth_2d - d_min) / (d_max - d_min)
    
    depth = (depth_norm * 255).astype(np.uint8)
    depth_mask = (depth==0).astype(np.uint8)
    depth_inpaint = cv2.inpaint(depth, depth_mask, 3, cv2.INPAINT_TELEA)
    return depth, depth_inpaint
  

file_path1 = "../TaRF/nerfstudio_modules/outputs/touch_estimation_input_cache/depth/40_50.npy"
file_path2 = "../TaRF/nerfstudio_modules/outputs/touch_estimation_input_cache/depth/0_40.npy"
file_path3 = "../TaRF/nerfstudio_modules/outputs/touch_estimation_input_cache/rgb/40_50.png"
file_path4 = "../TaRF/nerfstudio_modules/outputs/touch_estimation_input_cache/rgb/0_40.png"

img1 = cv2.imread(file_path3)
img2 = cv2.imread(file_path4)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

file1_depth, file1_depth_inpaint = show_depth_map(file_path1)
file2_depth, file2_depth_inpaint = show_depth_map(file_path2)

plt.subplot(2,2,1)
plt.title("Depth 40_50 View")
plt.imshow(file1_depth, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Depth 0_40 View")
plt.imshow(file2_depth, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("RGB 40_50 View")
plt.imshow(img1, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("RGB 0_40 View")
plt.imshow(img2, cmap='gray')
plt.axis('off')

plt.show()
