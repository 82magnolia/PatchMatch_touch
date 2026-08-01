"""Visualise what the TaRF baseline actually predicts.

Two panels, both written to log/paper_tarf_diagnostic/:

  tarf_vs_gt.png      several query touches, one per row:
                      query normal render | TaRF prediction | our refined
                      prediction | ground-truth query touch.
                      Shows whether TaRF is in the right output domain and
                      whether it recovers any surface structure.

  tarf_candidates.png the 8 DDIM candidates TaRF sampled for one query, with the
                      one its reranker selected outlined, next to the ground
                      truth. If every candidate is structureless the problem is
                      the diffusion model, not the reranking step.

Reads only what the baseline already wrote (generation/{q}/candidate_*.png and
selected.png), so it re-runs in seconds.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
TARF = os.path.join(ROOT, "log/paper_job1_baselines/tarf")
OURS_TRANSFER = os.path.join(ROOT, "log/paper_job1_transfer_normal")
OURS_REFINE = os.path.join(ROOT, "log/paper_job1_refine_ours_normal/videos")
QUERY_BASE = os.path.join(ROOT, "Taxim/results/gen_contact_full_query_tactile_normal_pseudo_mini")
OUT = os.path.join(ROOT, "log/paper_tarf_diagnostic")
W, H = 320, 240


def mid_frame(path):
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def read_img(path):
    if not os.path.exists(path):
        return None
    im = cv2.imread(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def cell(img, caption, border=None):
    if img is None:
        img = np.full((H, W, 3), 245, np.uint8)
        cv2.putText(img, "missing", (110, H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (120, 120, 120), 1, cv2.LINE_AA)
    else:
        img = cv2.resize(img, (W, H))
    if border is not None:
        img = cv2.copyMakeBorder(img[3:-3, 3:-3], 3, 3, 3, 3,
                                 cv2.BORDER_CONSTANT, value=border)
    strip = np.full((24, W, 3), 255, np.uint8)
    cv2.putText(strip, caption[:40], (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def panel_vs_gt(pairs):
    rows = []
    for obj, q in pairs:
        tarf = mid_frame(os.path.join(TARF, str(obj), "transfer", f"{q}_transferred.mp4"))
        gt = mid_frame(os.path.join(OURS_TRANSFER, str(obj), f"{q}_query_tactile_normal.mp4"))
        ours = mid_frame(os.path.join(OURS_REFINE, f"{obj}_{q}_enhanced.mp4"))
        qn = read_img(os.path.join(QUERY_BASE, str(obj), f"{q}_scale100_normal.jpg"))
        rows.append(np.hstack([
            cell(qn, f"obj {obj} touch {q}: query normal"),
            cell(tarf, "TaRF prediction"),
            cell(ours, "Ours (refined)"),
            cell(gt, "Ground truth"),
        ]))
    return np.vstack(rows)


def panel_candidates(obj, q):
    gdir = os.path.join(TARF, str(obj), "generation", str(q))
    meta_path = os.path.join(TARF, str(obj), "generation", "metadata.json")
    selected, scores = None, None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        entry = meta.get(str(q)) or meta.get(q) or {}
        if isinstance(entry, dict):
            selected = entry.get("selected")
            scores = entry.get("scores")

    cands = sorted(p for p in os.listdir(gdir) if p.startswith("candidate_"))
    cells = []
    for i, name in enumerate(cands):
        img = read_img(os.path.join(gdir, name))
        chosen = (selected is not None and i == selected)
        cap = f"candidate {i}" + (" [chosen]" if chosen else "")
        if scores and i < len(scores):
            cap += f"  score {scores[i]:.3f}"
        cells.append(cell(img, cap, border=(30, 120, 30) if chosen else None))
    gt = mid_frame(os.path.join(OURS_TRANSFER, str(obj), f"{q}_query_tactile_normal.mp4"))
    cells.append(cell(gt, "GROUND TRUTH", border=(20, 20, 200)))

    per_row = 3
    rows = [np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)]
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)])
            if r.shape[1] < width else r for r in rows]
    return np.vstack(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    objs = sorted((int(d) for d in os.listdir(TARF) if d.isdigit()))[:4]
    pairs = [(o, 0) for o in objs]

    a = panel_vs_gt(pairs)
    pa = os.path.join(OUT, "tarf_vs_gt.png")
    cv2.imwrite(pa, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    print(f"wrote {pa}")

    obj, q = pairs[0]
    b = panel_candidates(obj, q)
    pb = os.path.join(OUT, "tarf_candidates.png")
    cv2.imwrite(pb, cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
    print(f"wrote {pb}  (object {obj}, touch {q})")


if __name__ == "__main__":
    sys.exit(main())
