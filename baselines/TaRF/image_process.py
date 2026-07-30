import pyrealsense2 as rs
import numpy as np
import cv2
import os

def inpaint(depth_img):
    d_min = depth_img.min()
    d_max = depth_img.max()
    
    if d_max - d_min == 0:
        depth_norm = np.zeros_like(depth_img)
    else:
        depth_norm = (depth_img - d_min) / (d_max - d_min)
    
    depth_uint = (depth_norm * 255).astype(np.uint8)
    mask = (depth_uint==0).astype(np.uint8)
    depth_inpaint_u = cv2.inpaint(depth_uint, mask, 3, cv2.INPAINT_TELEA)
    depth_inpaint = (depth_inpaint_u.astype(np.float32) / 255.0) * (d_max - d_min) + d_min

    return depth_inpaint

def save_images(color_img, depth_img, output_dir, crop_40_50, crop_0_40, res, cx, cy):

        os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)

        def process_crop(img, crop_size, is_depth=False, shift = 0.0):
            x1 = cx - crop_size // 2
            y1 = cy - crop_size // 2
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            
            if is_depth: 
                img = inpaint(img)

            cropped = img[y1:y2, x1:x2]
            
            if is_depth:                
                resized = cv2.resize(cropped, (res, res), interpolation=cv2.INTER_CUBIC)
                resized = resized.astype(np.float32) * 0.001    # mm (uint16) -> meter (float32)
                if shift != 0.0:
                    resized = resized - shift
                
                resized = np.clip(resized, 0.0, 5.0)    # cliping (0 ~ 5m) - 모델 요구사항
                resized = resized[:, :, np.newaxis]     # TaRF input depth shape : (H, W, 1)
            else:
                resized = cv2.resize(cropped, (res, res), interpolation=cv2.INTER_CUBIC)
            return resized

        # rgb, depth image 생성
        rgb_40_50 = process_crop(color_img, crop_40_50)         # 40_50.png
        rgb_0_40 = process_crop(color_img, crop_0_40)           # 0_40.png

        depth_40_50 = process_crop(depth_img, crop_40_50, is_depth=True)                    # 40_50.npy
        depth_0_40 = process_crop(depth_img, crop_0_40, is_depth=True, shift=0.4)           # 0_40.npy

        cv2.imwrite(os.path.join(output_dir, 'rgb', '40_50.png'), rgb_40_50)
        cv2.imwrite(os.path.join(output_dir, 'rgb', '0_40.png'), rgb_0_40)

        np.save(os.path.join(output_dir, 'depth', '40_50.npy'), depth_40_50)
        np.save(os.path.join(output_dir, 'depth', '0_40.npy'), depth_0_40)

        print(f"[Saved!]")
        print(f"Path: {output_dir}")

def capture(output_dir):    
    # Realsense D435 RGB FOV : 69.4 * 42.5 / Depth FOV : 86 * 57 (aligned to RGB)
    # Center Distance = 0.45 * tan(50/2) / tan(42.5/2) ≈ 54
    
    # Crop Center of RGB image for 0_40.png(zoom)
    # Context : 45cm, Fov 50 >>> Zoom : 5cm, Fov 40.86
    # Zoom Scale = (0.45 * tan(50/2)) / (0.05 * tan(40.86/2)) ≈ 11.3
    ZOOM_FACTOR = 10.3     
    
    CROP_40_50 = 720 # Realsense 1280*720, 정사각형 이미지 크기 조정
    CROP_0_40 = int(CROP_40_50 / ZOOM_FACTOR)

    pipeline = rs.pipeline()
    config = rs.config()

    W, H = 1280, 720
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        print(f"Set Center Distance about 54cm")
        for _ in range(30):
            pipeline.wait_for_frames()
            
        while True:

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                return

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            h, w, _ = color_image.shape
            cx, cy = w // 2, h // 2  
            center_dist = aligned_depth_frame.get_distance(cx, cy)

            display_image = color_image.copy()
            cv2.line(display_image, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
            cv2.line(display_image, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

            cv2.putText(display_image, f"Depth: {center_dist:.3f} m", (cx + 30, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('RealSense Capture - Press SPACE to Capture', display_image)

            key = cv2.waitKey(1)
            
            if key & 0xFF == ord(' '):
                print(f"[Captured] Center Distance: {center_dist:.4f} m")
                
                # TaRF input Depth Shape (480,480,1)
                save_images(color_image, depth_image, output_dir, 
                                 CROP_40_50, CROP_0_40, 480, cx, cy)
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SAVE_DIR = "../TaRF/nerfstudio_modules/outputs/touch_estimation_input_cache" 
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    capture(SAVE_DIR)
