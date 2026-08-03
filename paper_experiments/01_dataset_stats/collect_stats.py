"""Collect statistics over the three tactile-analogies benchmark datasets.

Scans the rendered Taxim outputs and writes a JSON summary that the report
builder turns into HTML + Markdown.

Datasets
  ref   : Taxim/results/gen_contact_full_tactile_normal_pseudo_mini
          (reference touches of the ground-truth-retrieval benchmark)
  query : Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini
          (query touches of the ground-truth-retrieval benchmark)
  raw   : Taxim/results/gen_contact_raw_eval_tactile_normal_pseudo_mini
          (full-pipeline benchmark, all touches on one object pooled together)
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
RES = f"{ROOT}/Taxim/results"

DATASETS = {
    "ref": "gen_contact_full_tactile_normal_pseudo_mini",
    "query": "gen_contact_full_query_tactile_normal_pseudo_mini",
    "raw_eval": "gen_contact_raw_eval_tactile_normal_pseudo_mini",
}

# obj_scale_factor -> how much bigger the rendered patch is than the sensor
# footprint. 100 = exactly the sensor (1x), 50 = 2x, 25 = 4x.
SCALE_TO_X = {"100": "1x", "50": "2x", "25": "4x"}

VID_RE = re.compile(r"^(\d+)_([a-z_]+)\.mp4$")
IMG_RE = re.compile(r"^(\d+)_scale(\d+)_([a-z]+)\.(jpg|npz)$")


def dir_bytes(path):
    total = 0
    for dirpath, _, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, n))
            except OSError:
                pass
    return total


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"frames": n, "height": h, "width": w, "fps": fps}


def scan(name, rel, video_probe_objects=40):
    base = os.path.join(RES, rel)
    obj_ids = sorted((int(d) for d in os.listdir(base)
                      if d.isdigit() and os.path.isdir(os.path.join(base, d))))
    per_obj_touches = []
    modality_scale = Counter()          # (modality, scale) -> count of touches
    video_kinds = Counter()             # e.g. tactile_normal, mask, render_mask
    npz_shapes = Counter()
    frame_counts, heights, widths, fpss = [], [], [], []
    render_res = Counter()              # rendered map resolution per (scale, modality)

    for i, oid in enumerate(obj_ids):
        d = os.path.join(base, str(oid))
        names = os.listdir(d)
        touches = set()
        for n in names:
            m = VID_RE.match(n)
            if m:
                touches.add(int(m.group(1)))
                video_kinds[m.group(2)] += 1
                continue
            m = IMG_RE.match(n)
            if m:
                touches.add(int(m.group(1)))
                if m.group(4) == "jpg":
                    modality_scale[(m.group(3), m.group(2))] += 1
        per_obj_touches.append(len(touches))

        # Probe a subset for video geometry / array shapes (metadata reads only).
        if i < video_probe_objects:
            for t in sorted(touches):
                vp = os.path.join(d, f"{t}_tactile_normal.mp4")
                if os.path.exists(vp):
                    info = video_info(vp)
                    if info:
                        frame_counts.append(info["frames"])
                        heights.append(info["height"])
                        widths.append(info["width"])
                        fpss.append(info["fps"])
            for sc in ("100", "50", "25"):
                for mod in ("normal", "height"):
                    p = os.path.join(d, f"{sorted(touches)[0]}_scale{sc}_{mod}.npz")
                    if os.path.exists(p):
                        try:
                            z = np.load(p)
                            k = list(z.keys())[0]
                            npz_shapes[(mod, sc, str(z[k].shape))] += 1
                            render_res[(sc, mod, str(z[k].shape[:2]))] += 1
                        except Exception:
                            pass

    return {
        "name": name,
        "rel_path": rel,
        "n_objects": len(obj_ids),
        "obj_id_min": min(obj_ids) if obj_ids else None,
        "obj_id_max": max(obj_ids) if obj_ids else None,
        "touches_per_object": {
            "min": int(np.min(per_obj_touches)),
            "max": int(np.max(per_obj_touches)),
            "mean": float(np.mean(per_obj_touches)),
            "hist": dict(Counter(per_obj_touches)),
        },
        "total_touches": int(np.sum(per_obj_touches)),
        "video_kinds": {k: v for k, v in sorted(video_kinds.items())},
        "modality_scale_counts": {f"{m}@scale{s}": c
                                  for (m, s), c in sorted(modality_scale.items())},
        "tactile_video": {
            "probed_objects": min(video_probe_objects, len(obj_ids)),
            "n_probed": len(frame_counts),
            "frames_min": int(np.min(frame_counts)) if frame_counts else None,
            "frames_max": int(np.max(frame_counts)) if frame_counts else None,
            "frames_mean": float(np.mean(frame_counts)) if frame_counts else None,
            "height": int(np.median(heights)) if heights else None,
            "width": int(np.median(widths)) if widths else None,
            "fps": float(np.median(fpss)) if fpss else None,
        },
        "npz_shapes": {f"{m}@scale{s}": sh for (m, s, sh) in npz_shapes},
        "disk_bytes": dir_bytes(base),
    }


def main():
    out = {"scale_to_x": SCALE_TO_X, "datasets": {}}
    for name, rel in DATASETS.items():
        print(f"scanning {name} ...", flush=True)
        out["datasets"][name] = scan(name, rel)
        print(f"  -> {out['datasets'][name]['n_objects']} objects, "
              f"{out['datasets'][name]['total_touches']} touches", flush=True)

    # ObjectFolder source meshes actually used
    objdir = f"{ROOT}/Taxim/data/ObjectFolder"
    if os.path.isdir(objdir):
        n_meshes = sum(1 for d in os.listdir(objdir)
                       if os.path.isfile(os.path.join(objdir, d, "model.obj")))
        out["objectfolder_meshes"] = n_meshes

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote", dst)


if __name__ == "__main__":
    sys.exit(main())
