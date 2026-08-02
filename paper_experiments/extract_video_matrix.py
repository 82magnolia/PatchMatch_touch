"""Build the video-frame comparison matrices for Jobs 2 and 3.

Both figures are the same shape: one row per method, one column per video frame,
sampled evenly across the touch. Every panel is also written out as its own PNG so
the figure can be re-laid-out later without re-running any experiment.

  --figure job2   reference video, then each method's prediction, then ground truth.
                  Methods whose prediction is a single tiled image (the quilting
                  baseline) get the image in the first column and "N/A (image only)"
                  in the rest, as the outline asks.

  --figure job3   the refinement ablations against the same coarse input:
                  w/o network refinement, w/o temporal FiLM, w/o normal
                  concatenation, the full model, and ground truth.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
PANEL_W, PANEL_H = 320, 240
LABEL_W = 190


def read_frames(path):
    if not path or not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames or None


def sample(frames, n):
    if not frames:
        return [None] * n
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def placeholder(text):
    img = np.full((PANEL_H, PANEL_W, 3), 245, np.uint8)
    for i, line in enumerate(text.split("\n")):
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(img, line, ((PANEL_W - tw) // 2, PANEL_H // 2 - 8 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 110), 1, cv2.LINE_AA)
    return img


def row_label(text):
    img = np.full((PANEL_H, LABEL_W, 3), 255, np.uint8)
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if len(t) > 22:
            lines.append(cur)
            cur = w
        else:
            cur = t
    lines.append(cur)
    y = PANEL_H // 2 - (len(lines) - 1) * 11
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (20, 20, 20), 1, cv2.LINE_AA)
        y += 22
    return img


def col_header(n_cols, width):
    img = np.full((26, width, 3), 255, np.uint8)
    for c in range(n_cols):
        x = LABEL_W + c * PANEL_W + 8
        cv2.putText(img, f"frame {c + 1}", (x, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def build(rows, n_cols, out_dir, tag):
    """rows: list of (label, frames_or_None, image_only_bool)."""
    os.makedirs(out_dir, exist_ok=True)
    built = []
    for label, frames, image_only in rows:
        picks = sample(frames, n_cols)
        cells = [row_label(label)]
        for c, img in enumerate(picks):
            if img is None:
                cell = placeholder("missing")
            elif image_only and c > 0:
                cell = placeholder("N/A\n(image only)")
            else:
                cell = cv2.resize(img, (PANEL_W, PANEL_H))
                slug = label.lower().replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
                cv2.imwrite(os.path.join(out_dir, f"{tag}_{slug}_f{c}.png"),
                            cv2.cvtColor(cell, cv2.COLOR_RGB2BGR))
            cells.append(cell)
        built.append(np.hstack(cells))
    sheet = np.vstack(built)
    sheet = np.vstack([col_header(n_cols, sheet.shape[1]), sheet])
    path = os.path.join(out_dir, f"{tag}_matrix.png")
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return path


def pick_object(metrics_json, fallback, require=()):
    """Pick a representative object/touch that has videos saved for every row.

    The eval runs cap video writing with --max_videos, so most objects have
    metrics but no frames on disk. `require` lists video-path templates (with
    {obj}/{touch}) that must all exist for a candidate to be usable. Among the
    usable ones we take an upper-quartile touch rather than the single best, so
    the figure is representative instead of cherry-picked.
    """
    p = os.path.join(ROOT, metrics_json)
    if not os.path.exists(p):
        return fallback
    with open(p) as f:
        d = json.load(f)
    per_touch = d.get("per_touch", {})
    if not per_touch:
        return fallback

    ranked = sorted(per_touch.items(), key=lambda kv: -kv[1]["PSNR"])
    usable = []
    for key, _ in ranked:
        obj, touch = key.split("_")
        if all(os.path.exists(os.path.join(ROOT, t.format(obj=obj, touch=touch)))
               for t in require):
            usable.append((int(obj), int(touch)))
    if not usable:
        return fallback
    return usable[len(usable) // 4]


def job2(args):
    obj, touch = pick_object(
        "log/paper_job2_refine_ours_normal/metrics.json", (1, 0),
        require=["log/paper_job2_refine_ours_normal/videos/{obj}_{touch}_enhanced.mp4",
                 "log/paper_job2_baselines/quilting/{obj}/transfer/{touch}_transferred.mp4",
                 "log/paper_job2_baselines/inr/{obj}/transfer/{touch}_transferred.mp4",
                 args.tarf_variant + "/{obj}/transfer/{touch}_transferred.mp4"])
    if args.object is not None:
        obj, touch = args.object, (args.touch if args.touch is not None else touch)
    print(f"job2 figure: object {obj}, touch {touch}")

    tdir = os.path.join(ROOT, "log/paper_job2_pipeline_normal", str(obj), "transfer")
    rows = [
        ("Reference tactile normal", read_frames(os.path.join(tdir, f"{touch}_ref_tactile_normal.mp4")), False),
        ("Tactile Normal Quilting", read_frames(os.path.join(
            ROOT, "log/paper_job2_baselines/quilting", str(obj), "transfer", f"{touch}_transferred.mp4")), True),
        ("ObjectFolder INR", read_frames(os.path.join(
            ROOT, "log/paper_job2_baselines/inr", str(obj), "transfer", f"{touch}_transferred.mp4")), False),
        # TaRF predicts a single still image per query, so it is an image-only row.
        ("TaRF", read_frames(os.path.join(
            ROOT, args.tarf_variant, str(obj), "transfer", f"{touch}_transferred.mp4")), True),
        ("Ours: coarse transfer", read_frames(os.path.join(tdir, f"{touch}_transferred.mp4")), False),
        ("Ours: refined", read_frames(os.path.join(
            ROOT, "log/paper_job2_refine_ours_normal/videos", f"{obj}_{touch}_enhanced.mp4")), False),
        ("Ground truth", read_frames(os.path.join(tdir, f"{touch}_query_tactile_normal.mp4")), False),
    ]
    out = os.path.join(ROOT, "log/paper_job2_figure_assets")
    path = build(rows, args.cols, out, f"obj{obj}_touch{touch}")
    print(f"-> {path}")


def job3(args):
    obj, touch = pick_object(
        "log/paper_job3_refine_ours_mod_normal/metrics.json", (1, 0),
        require=[f"log/paper_job3_refine_{a}_mod_normal/videos/{{obj}}_{{touch}}_enhanced.mp4"
                 for a in ("ours", "wo_film", "wo_cat")])
    if args.object is not None:
        obj, touch = args.object, (args.touch if args.touch is not None else touch)
    print(f"job3 figure: object {obj}, touch {touch}")

    tdir = os.path.join(ROOT, "log/paper_job3_ablation/mod_normal", str(obj), "transfer")
    vid = lambda arm: os.path.join(ROOT, f"log/paper_job3_refine_{arm}_mod_normal/videos",
                                   f"{obj}_{touch}_enhanced.mp4")
    rows = [
        ("w/o network refinement", read_frames(os.path.join(tdir, f"{touch}_transferred.mp4")), False),
        ("w/o temporal FiLM", read_frames(vid("wo_film")), False),
        ("w/o normal concatenation", read_frames(vid("wo_cat")), False),
        ("Ours: full model", read_frames(vid("ours")), False),
        ("Ground truth", read_frames(os.path.join(tdir, f"{touch}_query_tactile_normal.mp4")), False),
    ]
    out = os.path.join(ROOT, "log/paper_job3_figure_assets")
    path = build(rows, args.cols, out, f"obj{obj}_touch{touch}")
    print(f"-> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", required=True, choices=["job2", "job3"])
    p.add_argument("--cols", type=int, default=5, help="Number of frames shown")
    p.add_argument("--object", type=int, default=None)
    p.add_argument("--touch", type=int, default=None)
    p.add_argument("--tarf_variant", default="log/paper_job2_baselines/tarf_v3",
                   help="Which trained TaRF checkpoint's predictions to show "
                        "(tarf, tarf_v2 or tarf_v3); job2 figure only")
    args = p.parse_args()
    (job2 if args.figure == "job2" else job3)(args)


if __name__ == "__main__":
    sys.exit(main())
