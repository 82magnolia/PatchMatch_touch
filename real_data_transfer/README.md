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
pip install segment-anything         # SAM (Meta)
pip install opencv-contrib-python    # includes ARuCO support
pip install numpy open3d matplotlib
```

> If you see a pyrealsense2 driver version mismatch at runtime, install the matching wheel explicitly:
> `pip install pyrealsense2==<version>` where `<version>` matches the output of `rs-enumerate-devices --version`.

### 3a. Connect the RealSense D435i

Plug the camera into a **USB 3** port (blue tab). USB 2 ports will cause the depth stream to fail or produce degraded framerates.

Verify the camera is detected:

```bash
rs-enumerate-devices
```

You should see an entry like `Intel RealSense D435I` with a serial number.

### 3b. (Optional) Connect the ZED 2i

Install the ZED SDK (includes drivers) from [stereolabs.com/developers](https://www.stereolabs.com/developers/), then install the Python package:

```bash
pip install pyzed
```

Verify the camera is detected:

```bash
python -c "import pyzed.sl as sl; c = sl.Camera(); print(c.open(sl.InitParameters()))"
```

You should see `SUCCESS`.

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

### `visualize_zed.py` — Real-time RGB-D viewer (ZED 2i)

Streams color and depth from the StereoLabs ZED 2i and displays them in two windows:

| Window | Contents |
|--------|----------|
| **RGB (left) \| Depth (right)** | cv2 window with color frame and JET-colorized depth side by side |
| **Point Cloud** | open3d window with interactive colored 3D point cloud |

**Dependencies** (`pm_real` env):

```bash
pip install pyzed   # or install via the ZED SDK installer
```

**Run:**

```bash
# Default (neural_plus — best quality)
python real_data_transfer/visualize_zed.py

# Override depth mode
python real_data_transfer/visualize_zed.py --depth_mode ultra
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--depth_mode` | `neural_plus` | ZED depth estimation mode: `performance`, `quality`, `ultra`, `neural`, `neural_plus` |

**Controls:**

| Input | Action |
|-------|--------|
| Left-drag (point cloud window) | Rotate |
| Scroll wheel | Zoom in / out |
| Right-drag | Pan |
| `q` (RGB-D window) | Quit |
| Close point cloud window | Quit |

> Depth is clipped to `0.3–3.0 m`. Point cloud is subsampled `[::4]` for real-time performance. `neural_plus` gives the best quality on textureless or specular surfaces; fall back to `ultra` if the neural module is not installed.

---

### `gen_aruco_pdf.py` — Print ARuCO marker sheet

Generates a PDF of N ARuCO markers (DICT_4X4_50) at a specified physical size, laid out in a grid on A4 pages.  Print and cut out the markers to attach to the turntable.

**Run:**

```bash
# 5 markers at 5 cm (defaults)
python real_data_transfer/gen_aruco_pdf.py

# Custom: 4 markers at 3.5 cm, IDs 0–3, saved to log/captures/
python real_data_transfer/gen_aruco_pdf.py \
    --n 4 --marker_size 0.092 --log_dir log/captures
```

Output filename encodes the ID range, e.g. `log/aruco_00_to_04.pdf`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | `5` | Number of markers |
| `--marker_size` | `0.05` | Printed side length in metres |
| `--quiet_zone` | `0.25` | Quiet-zone width as fraction of marker size |
| `--start_id` | `0` | First marker ID (max ID is 49 for DICT_4X4_50) |
| `--margin_mm` | `15` | Page margin in mm |
| `--gap_mm` | `6` | Gap between marker cells in mm |
| `--log_dir` | `log/` | Output directory |

> After printing, measure the black square with a ruler and pass the measured value as `--marker_size` to `calibrate_board.py` and `capture_turntable.py`.

---

### `calibrate_board.py` — One-time ARuCO board calibration

Establishes the 3D layout of all ARuCO markers on the turntable surface by observing them from multiple viewpoints and computing their positions relative to the lowest-ID (origin) marker.  Run this once per physical turntable setup; the result is used by `capture_turntable.py --board_config` for more accurate joint pose estimation.

**Run:**

```bash
# ZED 2i (default)
python real_data_transfer/calibrate_board.py \
    --log_dir log/captures \
    --marker_size 0.092

# RealSense D435i
python real_data_transfer/calibrate_board.py \
    --camera realsense \
    --log_dir log/captures \
    --marker_size 0.092
# Saves log/captures/board_config.json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--log_dir` | `log/captures` | Directory to save `board_config.json` |
| `--marker_size` | `0.092` | ARuCO marker side length in metres |
| `--camera` | `zed` | Camera to use: `zed` or `realsense` |
| `--depth_mode` | `neural_plus` | ZED depth mode: `performance`, `quality`, `ultra`, `neural`, `neural_plus` (ignored for RealSense) |

**Controls:**

| Input | Action |
|-------|--------|
| `c` | Capture current frame (accumulate detections) |
| `b` | Compute board layout from all accumulated frames and save |
| `q` | Quit |

**Tips:**
- Aim for 20+ frames.
- Vary the viewpoint by slightly tilting or shifting the camera (or rotating the turntable to different positions) so markers are seen from multiple angles.
- A good calibration prints `Z-spread < 5 mm`. Larger values mean more viewpoint diversity is needed.

---

### `capture_turntable.py` — Turntable capture with SAM + ARuCO pose tracking

Streams RGB-D with ARuCO marker axes overlaid. On each capture, the user clicks a point on the frozen frame to prompt SAM segmentation, then saves full and masked RGB/depth along with pose information.

#### SAM checkpoint

Download the ViT-B checkpoint (default) before first use:

```bash
# ViT-B (~375 MB, default)
wget -P log/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L (~1.2 GB)
wget -P log/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-H (~2.4 GB, best quality)
wget -P log/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**Run:**

```bash
# ZED 2i, board mode (recommended):
python real_data_transfer/capture_turntable.py \
    --board_config log/captures/board_config.json \
    --marker_size 0.092

# RealSense D435i, board mode:
python real_data_transfer/capture_turntable.py \
    --camera realsense \
    --board_config log/captures/board_config.json \
    --marker_size 0.092

# Default mode (per-marker independent poses, ZED):
python real_data_transfer/capture_turntable.py \
    --marker_size 0.092

# Saves to log/captures/ by default. Override with --log_dir log/my_session
# SAM defaults to log/sam_vit_b_01ec64.pth (vit_b). Override with --sam_checkpoint / --sam_model_type
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--log_dir` | `log/captures` | Directory to save capture outputs |
| `--marker_size` | `0.092` | ARuCO marker side length in metres (overridden by `board_config`) |
| `--board_config` | — | Path to `board_config.json` from `calibrate_board.py` (enables joint board pose) |
| `--sam_checkpoint` | `log/sam_vit_b_01ec64.pth` | Path to SAM checkpoint |
| `--sam_model_type` | `vit_b` | SAM model type: `vit_h`, `vit_l`, `vit_b` |
| `--camera` | `zed` | Camera to use: `zed` or `realsense` |
| `--depth_mode` | `neural_plus` | ZED depth mode: `performance`, `quality`, `ultra`, `neural`, `neural_plus` (ignored for RealSense) |

**Controls:**

| Input | Action |
|-------|--------|
| `c` (main window) | Freeze frame and enter capture mode |
| `q` (main window) | Quit |
| Drag left panel (capture mode) | Draw bounding box for SAM prompt |
| `Enter` (capture mode) | Run SAM inference |
| `r` (capture mode) | Redraw bounding box |
| `s` (capture mode) | Save capture |
| `Esc` (capture mode) | Cancel, return to live stream |

**Output** (per capture, zero-indexed):

| File | Content |
|------|---------|
| `NNN_rgb.png` | Full color frame |
| `NNN_depth.npy` | Raw depth (uint16, mm) |
| `NNN_depth_vis.png` | JET-colorized depth |
| `NNN_mask.png` | Binary SAM mask |
| `NNN_rgb_masked.png` | Color frame with mask applied |
| `NNN_depth_masked.npy` | Depth with mask applied |
| `poses.json` | ARuCO-derived poses for all captures |

**Session-level files** (written once per session):

| File | Content |
|------|---------|
| `intrinsics.json` | Color camera intrinsics (fx, fy, cx, cy, width, height) |
| `poses.json` | ARuCO-derived poses for all captures (see below) |

**`poses.json` structure (per entry):**
- `marker_poses` — dict mapping each detected marker ID → 4×4 `T_marker_in_cam`
- `co_visible_marker_ids` — list of marker IDs visible in both this capture and pick 0; these are the markers used to compute `T_relative`
- `T_relative` — 4×4 transform relative to pick 0, averaged over all co-visible markers (SVD-re-orthogonalised mean R, mean t); `null` at pick 0 or when no co-visible markers exist
- `T_world_from_cam` — 4×4 camera-to-world transform in the board/marker coordinate frame; `null` when no pose is available

> **Note:** the more ARuCO markers remain co-visible across captures, the more robust the relative pose estimate. If no co-visible markers are found, a warning is printed and `T_relative` is `null`.

---

### `tsdf_fusion.py` — TSDF mesh reconstruction from captures

Fuses the masked depth maps from `capture_turntable.py` into a single colored
triangle mesh using Open3D's Scalable TSDF Volume.  No camera connection needed —
reads `intrinsics.json` and `poses.json` written by `capture_turntable.py`.
Compatible with captures from both RealSense D435i and ZED 2i.

**Run:**

```bash
python real_data_transfer/tsdf_fusion.py \
    --capture_dir log/captures \
    --output      log/captures/mesh.ply
```

After integration the mesh is shown in an interactive Open3D window; close it to exit.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--capture_dir` | `log/captures` | Directory written by `capture_turntable.py` |
| `--output` | `<capture_dir>/mesh.ply` | Output mesh path (`.ply` or `.obj`) |
| `--voxel_size` | `0.002` | TSDF voxel size in metres (2 mm). Decrease for finer detail |
| `--sdf_trunc` | `4 × voxel_size` | SDF truncation distance in metres |
| `--max_depth` | `0.8` | Maximum depth to integrate in metres |
| `--depth_scale` | `1000.0` | Raw depth unit → metres divisor (1000 for uint16 mm) |
| `--no_mask` | off | Use full (unmasked) depth/color instead of SAM-masked files |
| `--fx/fy/cx/cy/width/height` | — | Manual intrinsic override when `intrinsics.json` is absent |

**Tips:**
- Start with the default `--voxel_size 0.002` (2 mm). For small objects or fine surface texture, try `0.001` (1 mm) — memory and compute scale roughly as voxel_size⁻³.
- Turntable captures work best with ≥ 8 evenly-spaced angular positions (45° steps) to avoid holes on the back face.
- If the mesh has floating fragments, open it in MeshLab and use *Filters → Cleaning → Remove Isolated Pieces* to discard small disconnected components.

---

### `mapanything_reconstruct.py` — MapAnything feed-forward 3D reconstruction

Runs [MapAnything](https://map-anything.github.io/) (a feed-forward metric 3D reconstruction transformer from Meta / CMU) on the masked RGB images and ARuCO-based camera poses produced by `capture_turntable.py`.  Unlike TSDF fusion, MapAnything does **not** use the RealSense depth stream — it infers dense depth entirely from the RGB images, guided by the known metric poses and intrinsics.  This can produce cleaner geometry on textureless or specular surfaces where the structured-light depth sensor struggles.

#### Additional setup (one-time, in `pm_real` env)

MapAnything ships with a pinned `opencv-python-headless` that conflicts with the `opencv-contrib-python` already in `pm_real`.  Install it without its conflicting dep:

```bash
conda activate pm_real
# Install the mapanything package without overwriting opencv-contrib-python
pip install -e real_data_transfer/map-anything --no-deps
# Install remaining mapanything dependencies
pip install huggingface_hub hydra-core natsort orjson pillow-heif plyfile \
    python-box safetensors tensorboard tqdm trimesh \
    jaxtyping termcolor timm torchmetrics minio
pip install "uniception==0.1.7" --no-deps
```

> **Expected warnings:** pip's resolver will print warnings about `opencv-python-headless` (not installed) and `rerun-sdk`/`torchaudio` (not installed). All three are harmless: `opencv-contrib-python` already in `pm_real` provides a superset of `opencv-python-headless`, and `rerun-sdk`/`torchaudio` are only needed for visualization utilities that this script does not use.

The first run will download the model weights from HuggingFace (~1–2 GB) and cache them in `~/.cache/huggingface/`.

#### Run

```bash
python real_data_transfer/mapanything_reconstruct.py \
    --capture_dir log/captures
```

Outputs are written to `log/captures/mapanything_out/` by default.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--capture_dir` | required | Directory written by `capture_turntable.py` |
| `--output_dir` | `<capture_dir>/mapanything_out` | Where to save outputs |
| `--model` | `facebook/map-anything` | HuggingFace model ID or local path |
| `--apache` | off | Use `facebook/map-anything-apache` (Apache 2.0 license) |
| `--no_mask` | off | Use unmasked `_rgb.png` instead of `_rgb_masked.png` |
| `--save_conf` | off | Also save per-frame confidence maps as `{idx}_conf.npy` |
| `--max_pts` | `500000` | Downsample combined point cloud to this many points (0 = no limit) |

**Output files:**

| File | Content |
|------|---------|
| `NNN_depth_ma.npy` | Per-frame float32 Z-depth in metres (original capture resolution) |
| `NNN_depth_ma_vis.png` | JET-colorized depth visualization |
| `NNN_conf.npy` | Per-frame confidence map (`--save_conf` only) |
| `pointcloud.ply` | Combined colored point cloud from all frames (masked, world frame) |
| `poses_ma.json` | Refined camera poses and intrinsics from MapAnything |

**Tips:**
- Capture at least 8 viewpoints spread around the object for good coverage.
- The board-mode poses from `capture_turntable.py --board_config` give the most accurate metric scale; prefer them over independent-marker mode.
- For objects where the RealSense depth is reliable (diffuse, non-transparent surfaces), `tsdf_fusion.py` can be faster and produce smoother meshes. Use MapAnything when depth is noisy or absent.

---

### `mapanything_vis.py` — Visualize MapAnything reconstruction

Interactive Open3D viewer showing the point cloud, input camera poses (blue frustums), and MapAnything-refined camera poses (red frustums) together.

**Run:**

```bash
python real_data_transfer/mapanything_vis.py --capture_dir log/captures
# recon_dir defaults to log/captures/mapanything_out
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--capture_dir` | required | Capture directory containing `poses.json` and `intrinsics.json` |
| `--recon_dir` | `<capture_dir>/mapanything_out` | MapAnything output directory |
| `--frustum_depth` | `0.05` | Frustum display depth in metres |
| `--frustum_axis_size` | `0.02` | Camera axis cross size in metres |
| `--no_pointcloud` | off | Skip loading the point cloud (faster startup for pose inspection) |

**Controls:** left-drag = rotate · scroll = zoom · right-drag = pan · `q` = quit
