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

### `visualize_zed_normal_sim.py` — GelSight Mini normal map simulator

Streams ZED 2i RGB + surface normals side-by-side (identical layout to `visualize_zed_normals.py`).  On `c`, freezes the frame and opens a capture window where you can:

1. **Segment** the object of interest with SAM (bounding-box prompt)
2. **Pick** a contact point on the segmented surface with a left-click
3. **View** an orthographic projection of the surface normal map cropped to the GelSight Mini sensor FoV (18.6 × 14.3 mm), with depth holes filled by inpainting

The picked contact point is shown as a red dot on both the RGB and normal panels.  The orthographic projection opens in a separate window and annotates the estimated depth and inpaint method used.

**Dependencies** (additional, in `pm_real` env):

```bash
pip install scipy   # optional — required only for --inpaint_method nearest
```

SAM checkpoint must be downloaded first (see `capture_turntable.py` section).

**Run:**

```bash
# Default (ZED neural_plus depth, SAM ViT-B, TELEA inpainting, 20 dpm output)
python real_data_transfer/visualize_zed_normal_sim.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth

# Higher-quality inpainting
python real_data_transfer/visualize_zed_normal_sim.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --inpaint_method ns

# Fastest inpainting (nearest-neighbour via scipy EDT)
python real_data_transfer/visualize_zed_normal_sim.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --inpaint_method nearest

# Finer output resolution (40 dots per mm → 744×572 px projection)
python real_data_transfer/visualize_zed_normal_sim.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --ortho_dpm 40
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--depth_mode` | `neural_plus` | ZED depth estimation mode: `performance`, `quality`, `ultra`, `neural`, `neural_plus` |
| `--confidence` | `95` | ZED depth confidence threshold 0–100 (lower accepts noisier pixels) |
| `--sam_checkpoint` | `log/sam_vit_b_01ec64.pth` | Path to SAM checkpoint |
| `--sam_model_type` | `vit_b` | SAM model: `vit_b`, `vit_l`, `vit_h` |
| `--inpaint_method` | `telea` | Hole inpainting: `telea` (fast, good), `ns` (slower, slightly better), `nearest` (fastest, scipy required) |
| `--ortho_dpm` | `20` | Output dots-per-mm for orthographic projection (20 dpm → 372×286 px at GelSight FoV) |

**Controls (main window):**

| Input | Action |
|-------|--------|
| `c` | Freeze frame, enter capture mode |
| `q` | Quit |

**Controls (capture window):**

| Input | Action |
|-------|--------|
| Drag left panel | Draw SAM bounding box |
| `Enter` | Run SAM segmentation |
| `r` | Redraw box / re-segment |
| Left-click on object | Pick contact point and compute projection |
| `p` | Re-pick contact point |
| `Esc` | Close capture window, return to live stream |

**Inpainting methods:**

| Method | Flag | Speed | Notes |
|--------|------|-------|-------|
| TELEA (Criminisi) | `telea` | fast | Default; good quality for small holes |
| Navier-Stokes | `ns` | slower | Slightly smoother results |
| Nearest-neighbour | `nearest` | fastest | Requires `scipy`; preserves sharpness, no smoothing |

> The orthographic projection is centred at the picked pixel and covers `18.6 × 14.3 mm` at the object's measured depth.  Depth holes within the crop are inpainted before rendering.  Pixels outside the SAM mask are blacked out.

---

### `visualize_zed_fs.py` — FoundationStereo depth/normal viewer (ZED 2i)

Streams ZED RGB + ZED surface normals side-by-side in a live window. Press `c` to capture a rectified stereo pair and run **FoundationStereo** or **Fast-FoundationStereo** inference. Results appear in a second window (RGB | FS depth | FS normals) and an interactive Open3D point cloud viewer.

**Additional dependencies** (in `pm_real` env, on top of the base setup):

```bash
conda activate pm_real
pip install timm einops omegaconf imageio
# open3d should already be installed; if not:
pip install open3d
```

No `setup.py` for either repo — they are imported directly via `sys.path` from `real_data_transfer/FoundationStereo/` and `real_data_transfer/Fast-FoundationStereo/`.

**Download checkpoints:**

- **Fast-FoundationStereo** — download from the [Fast-FoundationStereo releases](https://github.com/NVlabs/Fast-FoundationStereo) and place the checkpoint and its accompanying `cfg.yaml` under `real_data_transfer/Fast-FoundationStereo/weights/`.
- **FoundationStereo** — download from [HuggingFace](https://huggingface.co/nvidia/FoundationStereo) and place the checkpoint and its accompanying `cfg.yaml` under `real_data_transfer/FoundationStereo/pretrained_models/`.

Both directories are git-ignored.

**Run:**

```bash
# Fast-FoundationStereo (default, ~50 ms/frame @ 8 iters on RTX 3090)
python real_data_transfer/visualize_zed_fs.py \
    --model_dir real_data_transfer/Fast-FoundationStereo/weights/model_best_bp2_serialize.pth \
    --model_type fast_foundation_stereo

# FoundationStereo (higher quality, slower)
python real_data_transfer/visualize_zed_fs.py \
    --model_dir real_data_transfer/FoundationStereo/pretrained_models/model_best_bp2.pth \
    --model_type foundation_stereo \
    --valid_iters 32

# FoundationStereo on <11 GB GPU: scale down to fit in memory
python real_data_transfer/visualize_zed_fs.py \
    --model_dir real_data_transfer/FoundationStereo/pretrained_models/model_best_bp2.pth \
    --model_type foundation_stereo \
    --valid_iters 32 --scale 0.9

# FoundationStereo small model (lighter weight, downloadable from FoundationStereo GitHub)
python real_data_transfer/visualize_zed_fs.py \
    --model_dir real_data_transfer/FoundationStereo/pretrained_models/small/model_best_bp2.pth \
    --model_type foundation_stereo \
    --valid_iters 32

# Halve resolution for faster inference
python real_data_transfer/visualize_zed_fs.py \
    --model_dir real_data_transfer/Fast-FoundationStereo/weights/model_best_bp2_serialize.pth \
    --scale 0.5
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | *(required)* | Path to model checkpoint `.pth` |
| `--model_type` | `fast_foundation_stereo` | `foundation_stereo` or `fast_foundation_stereo` |
| `--valid_iters` | 8 / 32 | Refinement iterations (auto-set per model type) |
| `--max_disp` | `192` | Max disparity for Fast-FoundationStereo |
| `--scale` | `1.0` | Image downscale factor for inference (≤1) |
| `--depth_mode` | `neural_plus` | ZED depth mode for live normal display |
| `--zed_confidence` | `95` | ZED depth confidence 0–100 |
| `--z_near` | `0.1` | Near depth clip in metres |
| `--z_far` | `3.0` | Far depth clip in metres |

**Controls:**

| Input | Action |
|-------|--------|
| `c` | Capture stereo frame and run FoundationStereo inference |
| `q` | Quit |
| Mouse (Open3D window) | Rotate / zoom / pan point cloud |

> The ZED already outputs rectified stereo images, so no additional rectification is needed. Baseline is read automatically from ZED calibration. FS normals are computed from the depth map via finite-difference cross-product of the xyz map.

---

### `capture_gelsight.py` — Real GelSight tactile capture for PatchMatch pipeline

Produces `{idx}_normal.jpg/.npz`, `{idx}_color.jpg`, `{idx}_shadow.mp4` per touch location — the same file layout read by `main_retrieval_transfer_accel.py` (with `--scale` omitted). Combines ZED 2i depth/normals with a GelSight Mini tactile sensor.

Requires ARuCO marker ID=6 (DICT_4X4_50, 37 mm default) attached to the flat back face of `gsmini_holder.stl`. Holder geometry is hardcoded from the STL (30 mm height). GelSight camera is opened via OpenCV VideoCapture.

**Two-stage workflow:**

**Stage 1 — Object Selection** (run once per view):
1. ZED live RGB + normals stream appears
2. Press `c` → freeze frame → drag SAM bounding box → `Enter` → `y` to confirm
3. Object mask + ZED data cached; GelSight blank (no-contact) frame saved automatically

**Stage 2 — Touch Recording** (repeat for each contact point):
1. Dashboard window shows live ARuCO tracking, object cache, GelSight live feed, orthographic normals/RGB/height/contact-mask panels, and a final row of `--render_scale` normal/RGB previews
2. Press `r` to start recording GelSight frames into buffer
3. Press `s` to stop → frames trimmed (contact detection vs blank) → resampled to `--num_frames` → saved
4. Press `a` to abort recording without saving
5. Press `t` to capture a new view after rotating the turntable — see below
6. Press `q` to quit

**Multi-view turntable workflow:**

After capturing touches at one view, rotate the turntable and press `t`:
1. A live ZED RGB + normals window opens showing the current camera view
2. Physically rotate the turntable to the desired angle
3. Press `c` → optionally runs FS inference (if `--geometry_mode` is set) → SAM bounding-box prompt opens to re-segment the object in the new view
4. The dashboard cache panel updates with the new view's normals and depth (expressed in the current camera frame — each view is an independent measurement); Stage 2 resumes — press `r`/`s` to record more touches
5. Press `t` again to rotate further; repeat as desired; all saves go to the same `--save_dir`
6. Press `Esc` in the tracking window to cancel without updating the cache

**Run:**

```bash
# Default (neural_plus depth, SAM ViT-B, TELEA inpainting, 37mm marker, 50 frames)
python real_data_transfer/capture_gelsight.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --save_dir log/gelsight_captures/session_01

# Custom marker size
python real_data_transfer/capture_gelsight.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --marker_size 0.035 \
    --save_dir log/gelsight_captures/session_01

# If GelSight is not device 0
python real_data_transfer/capture_gelsight.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --gelsight_device 2 \
    --save_dir log/gelsight_captures/session_01

# Use FoundationStereo for higher-quality depth/normals on specular surfaces
python real_data_transfer/capture_gelsight.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --geometry_mode foundation_stereo \
    --fs_model_dir real_data_transfer/FoundationStereo/pretrained_models/model_best_bp2.pth \
    --fs_scale 0.9 \
    --save_dir log/gelsight_captures/session_fs \
    --gelsight_device 2

# Use Fast-FoundationStereo (faster, slightly lower quality)
python real_data_transfer/capture_gelsight.py \
    --sam_checkpoint log/sam_vit_b_01ec64.pth \
    --geometry_mode fast_foundation_stereo \
    --fs_model_dir real_data_transfer/Fast-FoundationStereo/weights/model_best_bp2_serialize.pth \
    --save_dir log/gelsight_captures/session_fast_fs \
    --gelsight_device 2
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--depth_mode` | `neural_plus` | ZED depth mode |
| `--zed_confidence` | `95` | ZED depth confidence threshold 0–100 |
| `--sam_checkpoint` | `log/sam_vit_b_01ec64.pth` | SAM checkpoint path |
| `--sam_model_type` | `vit_b` | SAM model variant |
| `--gelsight_device` | `0` | OpenCV VideoCapture device index or path |
| `--marker_size` | `0.037` | ARuCO marker physical size in metres |
| `--num_frames` | `50` | Resampled frame count (matches `gen_contact_query.sh`) |
| `--contact_threshold` | `0.05` | Mean L2 diff vs blank for contact trimming |
| `--inpaint_method` | `telea` | Normal-map hole inpainting: `telea`, `ns`, `nearest` |
| `--render_scale` | `1` | One or more FoV multipliers for normal/RGB orthographic renders, each still output at 320×240 |
| `--save_dir` | `log/gelsight_captures` | Output directory |
| `--debug_sensor_align` | off | Add a dashboard row with the ortho normal map and RGB crop each blended 50/50 over the live GelSight frame, side-by-side. Useful for verifying that the sensor pose is correctly aligned before recording. |
| `--geometry_mode` | `zed` | Geometry source: `zed` (ZED built-in), `foundation_stereo`, or `fast_foundation_stereo`. FS modes run stereo inference on the captured pair at Stage 1 and use the resulting depth/normals for all ortho projections and render masks. |
| `--fs_model_dir` | — | Path to FS checkpoint `.pth`. Required when `--geometry_mode != zed`. |
| `--fs_valid_iters` | 8 / 32 | FS refinement iterations (auto-set per model type). |
| `--fs_max_disp` | `192` | Max disparity for Fast-FoundationStereo. |
| `--fs_scale` | `1.0` | Image downscale factor for FS inference (≤1). Use `0.9` for <11 GB GPU. |

**Output files per touch location** (in `--save_dir`):

| File | Description |
|------|-------------|
| `blank_frame.jpg` | GelSight no-contact frame (saved once in Stage 1) |
| `object_cache.npz` | Cached ZED color/normals/depth/xyz/mask (saved once) |
| `{idx}_normal.jpg` | Orthographic normal colormap at GelSight FoV |
| `{idx}_normal.npz` | Raw float32 normals (H×W×3), key `"normal"` |
| `{idx}_color.jpg` | Orthographic RGB at GelSight FoV |
| `{idx}_scale{scale}_normal.jpg/.npz` | Multi-scale normal render for each `--render_scale` entry |
| `{idx}_scale{scale}_color.jpg` | Multi-scale RGB render for each `--render_scale` entry |
| `{idx}_shadow.mp4` | Trimmed + resampled GelSight tactile video |
| `{idx}_render_mask.mp4` | Per-frame contact mask video |
| `{idx}_shadow_render_mask.mp4` | Side-by-side shadow + render mask |
| `{idx}_contact_data.npz` | `height_map_0`, `valid_depth_remap`, `mask_crop`, `sensor_z_0`, `tvec_0` at contact start (for post-hoc re-rendering) |
| `{idx}_pose_contact.npz` | `rvecs`/`tvecs` (N×3) for each frame in the contact window, NaN where marker was missing |
| `{idx}_diffs.npz` | `diffs`, `smooth_diffs` from the contact detection curve (for re-trimming) |
| `{idx}_meta.json` | Contact pixel, ARuCO pose, frame counts, `cs_idx`, `ce_idx`, `peak_idx`, `trim_threshold` |

**Downstream usage with PatchMatch pipeline:**

```bash
# Run transfer (omit --scale to match scale-free filenames)
python main_retrieval_transfer_accel.py \
    --query_dir log/gelsight_captures/session_01 \
    --ref_dir   Taxim/results/gen_contact_full \
    --retrieval_pkl log/touch_retrieval/results.pkl \
    --modality raw_normal \
    --video_type shadow \
    --save_dir log/transfer_real \
    --em --em_iters 3
```

---

### `render_masks.py` — Post-hoc contact mask re-rendering

Re-generates `{idx}_render_mask.mp4` and `{idx}_shadow_render_mask.mp4` from the raw geometry saved by `capture_gelsight.py`, without needing the ZED or GelSight connected. Useful for adjusting the contact threshold or re-trimming the contact window after the fact.

**Required saved files** (written automatically by `capture_gelsight.py`):

| File | Used for |
|------|---------|
| `{idx}_contact_data.npz` | Height map, depth validity, mask, sensor axes at contact start |
| `{idx}_pose_contact.npz` | Sensor pose sequence over the contact window |
| `{idx}_shadow.mp4` | Resampled GelSight frames for the side-by-side video |
| `{idx}_diffs.npz` | Diff curve (only needed when re-trimming with `--peak_ratio`) |

**Run:**

```bash
# Re-render touch 3 with a looser contact threshold (+1 mm), output alongside originals
python real_data_transfer/render_masks.py \
    --data_dir log/gelsight_captures/session_01 \
    --touch_idx 3 \
    --render_mask_thres 0.001

# Write re-rendered videos to a separate directory
python real_data_transfer/render_masks.py \
    --data_dir log/gelsight_captures/session_01 \
    --output_dir log/remask/session_01 \
    --touch_idx 3 \
    --render_mask_thres 0.001

# Re-render with a tighter threshold (-0.5 mm)
python real_data_transfer/render_masks.py \
    --data_dir log/gelsight_captures/session_01 \
    --touch_idx 3 \
    --render_mask_thres -0.0005

# Re-trim the contact window AND re-render (change peak detection sensitivity)
python real_data_transfer/render_masks.py \
    --data_dir log/gelsight_captures/session_01 \
    --touch_idx 3 \
    --peak_ratio 0.3 \
    --render_mask_thres 0.001

# Batch: re-render all touches, write to a separate directory
python real_data_transfer/render_masks.py \
    --data_dir log/gelsight_captures/session_01 \
    --output_dir log/remask/session_01 \
    --render_mask_thres 0.0005

# Dry run: check which touches have the required data without writing files
python real_data_transfer/render_masks.py \
    --data_dir log/gelsight_captures/session_01 \
    --dry_run
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `log/gelsight_captures` | Directory written by `capture_gelsight.py` containing the raw capture data |
| `--output_dir` | *(same as `--data_dir`)* | Directory to write re-rendered videos; created if it does not exist |
| `--touch_idx` | *(all)* | Single touch index to re-render; omit to process all touches in `--data_dir` |
| `--render_mask_thres` | `0.0` | Height threshold in metres for contact detection. Negative = tighter (fewer pixels in contact), positive = looser (more pixels) |
| `--peak_ratio` | *(original)* | Re-trim the contact window using this peak_ratio on the saved diff curve before rendering. Omit to keep the original trim boundaries |
| `--num_frames` | *(original)* | Number of output frames; defaults to the frame count from the original capture |
| `--dry_run` | off | Print what would be processed without writing any files |

**How the threshold works:**

The contact mask per frame is:

```
contact = height_map_0 < pressing_depth_i + render_mask_thres
```

where `pressing_depth_i = dot(sensor_z_0, tvec_i - tvec_0)` tracks the sensor's advance depth over time. Setting `render_mask_thres = 0` (default) activates contact exactly when the object surface reaches the gel plane. Increasing it makes the mask appear earlier/larger; decreasing it makes it tighter.

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
