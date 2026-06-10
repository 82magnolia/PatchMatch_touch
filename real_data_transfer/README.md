# real_data_transfer

Scripts for capturing and processing data from physical sensors:
- **Intel RealSense D435i** — RGB-D camera
- **GelSight Mini** — tactile sensor

## Setup

> **Note:** Use a separate conda environment for real-world data collection. The main `pm_touch` env is reserved for running transfer and evaluation.

### 1. Create and activate the data-collection environment

```bash
conda create -n pm_real python=3.10
conda activate pm_real
```

### 2. Install dependencies

```bash
pip install pyrealsense2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything-2.git  # SAM 2
pip install opencv-contrib-python   # includes ARuCO support
pip install numpy open3d
```

> If you see a pyrealsense2 driver version mismatch at runtime, install the matching wheel explicitly:
> `pip install pyrealsense2==<version>` where `<version>` matches the output of `rs-enumerate-devices --version`.

### 3. Connect the RealSense D435i

Plug the camera into a **USB 3** port (blue tab). USB 2 ports will cause the depth stream to fail or produce degraded framerates.

Verify the camera is detected:

```bash
rs-enumerate-devices
```

You should see an entry like `Intel RealSense D435I` with a serial number.

---

## Scripts

### `visualize_realsense.py` — Real-time RGB-D viewer

Streams color and depth from the RealSense D435i and displays them in two windows:

| Window | Contents |
|--------|----------|
| **RGB (left) \| Depth (right)** | cv2 window with color frame and JET-colorized depth side by side |
| **Point Cloud** | open3d window with interactive colored 3D point cloud |

**Run:**

```bash
python real_data_transfer/visualize_realsense.py
```

**Controls:**

| Input | Action |
|-------|--------|
| Left-drag (point cloud window) | Rotate |
| Scroll wheel | Zoom in / out |
| Right-drag | Pan |
| `q` (RGB-D window) | Quit |
| Close point cloud window | Quit |

---

### `capture_turntable.py` — Turntable capture with SAM 2 + ARuCO pose tracking

Streams RGB-D with ARuCO marker axes overlaid. On each capture, the user clicks a point to prompt SAM 2 segmentation, then saves full and masked RGB/depth along with pose information.

**Run:**

```bash
python real_data_transfer/capture_turntable.py \
    --save_dir real_data_transfer/captures \
    --marker_size 0.05        # ARuCO marker side length in metres
```

**Controls:**

| Input | Action |
|-------|--------|
| `c` (main window) | Freeze frame and enter capture mode |
| `q` (main window) | Quit |
| Click left panel (capture mode) | Set SAM 2 point prompt |
| `Enter` (capture mode) | Run SAM 2 inference |
| `r` (capture mode) | Re-select point |
| `s` (capture mode) | Save capture |
| `Esc` (capture mode) | Cancel, return to live stream |

**Output** (per capture, zero-indexed):

| File | Content |
|------|---------|
| `NNN_rgb.png` | Full color frame |
| `NNN_depth.npy` | Raw depth (uint16, mm) |
| `NNN_depth_vis.png` | JET-colorized depth |
| `NNN_mask.png` | Binary SAM 2 mask |
| `NNN_rgb_masked.png` | Color frame with mask applied |
| `NNN_depth_masked.npy` | Depth with mask applied |
| `poses.json` | ARuCO-derived poses for all captures |

**`poses.json` structure:**
- `T_marker_in_cam` — 4×4 marker-to-camera transform for this capture
- `T_relative` — 4×4 transform relative to the first capture (pick 0); `null` for pick 0 itself
