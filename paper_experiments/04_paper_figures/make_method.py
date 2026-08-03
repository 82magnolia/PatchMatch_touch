"""Method overview figure: retrieval, coarse alignment, network refinement.

Three versions of the same three-step story:

  v1  one column, each step on a shaded panel, drawn in the most detail
  v2  one column, flat look (no panel shading), reference database shown as a
      vertical list, network drawn as a plain chain of blocks
  v3  two-column width, the three steps running left to right

Every image and every number is real (see prep_assets.py): the similarity
values are DINOv3 cosine similarities between normal renders, the lines are
SuperPoint + SuperGlue inliers, and the tactile frames are benchmark videos
and refinement-network output. All three steps show the same touch: the
retrieval step's top-1 pick is the reference that steps 2 and 3 then use.
"""
import argparse
import os
import pickle

import numpy as np

from figlib import ASSETS, C_QRY, C_REF, Page, load, load_normal, sensor_box

OUT = "/home/junhokim/Projects/PatchMatch_gpu/log/paper_job04_paper_figures"
C_MATCH = "#f2c200"
C_NET = "#5b4b8a"
# Fill of the box behind each step. The faint grey separates the steps from the
# page, but a white fill keeps the whole figure on one background, which reads
# better once the normal renders themselves are on white too (see load_normal).
PANEL_FACE = "#f4f5f7"
PANEL_EDGE = "#d5d8dd"
C_FILM_E = "#c9a227"
C_FILM_F = "#fdf3d6"
C_FILM_T = "#6b5300"


# --------------------------------------------------------------------- pieces
def seg_hits_rect(p0, p1, rect):
    """Does the segment p0 -> p1 touch the axis-aligned rect (x0, y0, x1, y1)?

    Liang-Barsky clipping: walk the four edges, narrowing the slice of the
    segment that could still be inside. An empty slice means it misses.
    """
    x0, y0, x1, y1 = rect
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for num, den in ((-dx, p0[0] - x0), (dx, x1 - p0[0]),
                     (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if num == 0:                    # parallel to this edge
            if den < 0:
                return False
            continue
        t = den / num
        if num < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def draw_matches(p, xl, yl, xr, yr, w, h, md, n_lines=14, lw=0.45, box_scale=25,
                 marker="D", ms=1.3, flip=False):
    """Draw real inlier correspondences between two already-placed renders.

    Correspondences whose line would cross either sensor-footprint box are left
    out, so the red boxes stay readable. Each drawn line gets a small diamond at
    both ends, marking the two points that were actually matched. `flip` says the
    query render is the one on the left, so its coordinates go with (xl, yl).
    """
    im_h, im_w = load(f"{ASSETS}/{md['tag']}_match_left.png").shape[:2]
    near, far = ("xy_r", "xy_l") if flip else ("xy_l", "xy_r")

    def pts(i):
        ax, ay = md[near][i]
        bx, by = md[far][i]
        return ((xl + ax / im_w * w, yl + ay / im_h * h),
                (xr + bx / im_w * w, yr + by / im_h * h))

    # the drawn footprint of each red box, in page inches
    half = 100.0 / float(box_scale) * 2      # box side is 1/(100/scale) of the image
    boxes = [(x + w / 2 - w / half, y + h / 2 - h / half,
              x + w / 2 + w / half, y + h / 2 + h / half)
             for x, y in ((xl, yl), (xr, yr))]

    keep = [i for i in np.where(md["inlier"])[0]
            if not any(seg_hits_rect(*pts(i), b) for b in boxes)]
    if len(keep) > n_lines:
        keep = [keep[j] for j in np.linspace(0, len(keep) - 1, n_lines).round().astype(int)]
    for i in keep:
        p.line(list(pts(i)), color=C_MATCH, lw=lw, z=6, alpha=0.95,
               marker=marker, ms=ms)
    return len(keep)


def sim_bar(p, x, y, w, h, sim, color, lo=0.4, hi=1.0):
    """Similarity shown against a fixed [lo, hi] scale, not rescaled per figure."""
    frac = float(np.clip((sim - lo) / (hi - lo), 0.02, 1.0))
    p.box(x, y, w, h, facecolor="#e6e8ec", edgecolor="none", lw=0, radius=0.004, z=3)
    p.box(x, y, w * frac, h, facecolor=color, edgecolor="none", lw=0, radius=0.004, z=4)


def net_block(p, x, y, w, h, label, sub=None, size=4.6):
    p.box(x, y, w, h, facecolor="#efedf6", edgecolor=C_NET, lw=0.7, radius=0.014, z=3)
    p.text(x + w / 2, y + h / 2 - (0.045 if sub else 0), label, size=size, ha="center",
           color="#2c2440", z=6)
    if sub:
        p.text(x + w / 2, y + h / 2 + 0.05, sub, size=size - 0.6, ha="center",
               color="0.45", z=6)


def film_group(p, x, y, w, h, n_arrows, arrow_top, note_x=None, size=4.3,
               note="scales and shifts each stage (FiLM)"):
    p.box(x, y, w, h, facecolor=C_FILM_F, edgecolor=C_FILM_E, lw=0.7)
    p.text(x + w / 2, y + h / 2, "frame number t", size=size, ha="center", color=C_FILM_T)
    for k in range(n_arrows):
        xx = x + 0.07 + k * ((w - 0.16) / max(n_arrows - 1, 1))
        p.arrow(xx, y - 0.005, xx, arrow_top, color=C_FILM_E, lw=0.6, mut=3)
    if note_x is None:      # no room beside the box: put the note underneath
        p.text(x + w / 2, y + h + 0.09, note.replace("each stage ", "each stage\n"),
               size=size - 0.1, ha="center", color=C_FILM_T)
    else:
        p.text(note_x, y + h / 2, note, size=size - 0.1, color=C_FILM_T)


# ------------------------------------------------------------------ version 1
def v1(d, out):
    ret, md, tag, mid = d["ret"], d["md"], d["tag"], d["mid"]
    p = Page(3.35, 4.64)
    L, R = 0.09, 3.26

    # ------------------------------------------------------------ 1 retrieval
    p.panel(0.04, 0.06, 3.27, 1.44, facecolor=PANEL_FACE, edgecolor=PANEL_EDGE)
    p.text(L, 0.19, "1", size=7.5, weight="bold", color=C_NET)
    p.text(L + 0.09, 0.19, "Retrieve the most similar reference touch", size=5.6,
           color="0.15")
    p.text(R, 0.19, f"{len(ret['show'])} of {ret['n_db']} references shown", size=4.3,
           ha="right", color="0.45")

    qw, qh = 0.60, 0.45
    qx, qy = L, 0.46
    p.img(qx, qy, qw, qh, load_normal(ret["qimg"]), edge=C_QRY, lw=0.9)
    p.text(qx, qy - 0.06, "query location", size=4.8, color=C_QRY)
    p.text(qx, qy + qh + 0.09, "geometry only,\nnever touched", size=4.3, color="0.45")

    fx, fw, fh = 0.79, 0.30, 0.17
    fy = qy + qh / 2 - fh / 2
    net_block(p, fx, fy, fw, fh, "DINOv3", size=4.8)
    p.arrow(qx + qw + 0.015, fy + fh / 2, fx - 0.015, fy + fh / 2, lw=0.8, mut=4)
    p.text(fx + fw / 2, fy + fh + 0.14, "cosine similarity\nof DINOv3 features",
           size=4.2, ha="center", color="0.45")

    ex0, ew, gap = 1.29, 0.40, 0.065
    p.arrow(fx + fw + 0.015, fy + fh / 2, ex0 - 0.02, fy + fh / 2, lw=0.8, mut=4)
    for k, (idx, sim) in enumerate(ret["show"]):
        x = ex0 + k * (ew + gap)
        top1 = idx == ret["top1"]
        col = C_REF if top1 else "0.72"
        p.img(x, 0.46, ew, 0.30, load_normal(f"{ASSETS}/{ret['tag']}_db{idx}_normal.png"),
              edge=col, lw=0.9)
        p.img(x, 0.79, ew, 0.30, load(f"{ASSETS}/{ret['tag']}_db{idx}_touch.png"),
              edge=col, lw=0.9)
        sim_bar(p, x, 1.13, ew, 0.05, sim, col)
        p.text(x + ew / 2, 1.24, f"{sim:.2f}", size=4.4, ha="center",
               color=C_REF if top1 else "0.5", weight="bold" if top1 else "normal")
        if top1:
            p.text(x + ew / 2, 0.41, "best match", size=4.4, ha="center", color=C_REF)
    p.text(ex0, 1.38, "top: geometry at 1x the sensor     bottom: the touch measured there",
           size=4.2,
           color="0.45")

    # ------------------------------------------------------ 2 coarse alignment
    p.panel(0.04, 1.58, 3.27, 1.42, facecolor=PANEL_FACE, edgecolor=PANEL_EDGE)
    p.text(L, 1.71, "2", size=7.5, weight="bold", color=C_NET)
    p.text(L + 0.09, 1.71, "Line the two geometry renders up", size=5.6, color="0.15")

    mw, mh, my = 1.05, 0.79, 1.98
    mxl, mxr = L, L + mw + 0.28
    # query first, then the reference it was matched against. Both renders cover
    # 4x the sensor, so the red box marks the quarter-size centre patch the
    # sensor actually sits on.
    p.img(mxl, my, mw, mh, sensor_box(load_normal(f"{ASSETS}/{tag}_match_right.png")),
          edge="black", lw=0.9)
    p.img(mxr, my, mw, mh, sensor_box(load_normal(f"{ASSETS}/{tag}_match_left.png")),
          edge=C_REF, lw=0.9)
    shown = draw_matches(p, mxl, my, mxr, my, mw, mh, md, flip=True)
    p.text(mxl, my - 0.06, "query", size=4.6, color=C_QRY)
    p.text(mxr, my - 0.06, "best match", size=4.6, color=C_REF)
    p.text(mxl, my + mh + 0.12,
           "geometry at 4x the sensor; the red box marks the sensor itself\n"
           f"{shown} of the {int(md['inlier'].sum())} agreeing SuperPoint + SuperGlue "
           f"correspondences drawn", size=4.2, color="0.45", linespacing=1.4)

    cx, cw, ch = 2.60, 0.55, 0.41
    p.text(cx, my + 0.06, "fit one warp\n(homography),\nwarp the video", size=4.4, color="0.25")
    p.img(cx, my + 0.34, cw, ch, load(f"{ASSETS}/{tag}_coarse_{mid:03d}.png"),
          edge="0.55", lw=0.9)
    p.text(cx, my + 0.34 + ch + 0.07, "coarse transfer", size=4.4, color="0.35")
    p.arrow(mxr + mw + 0.02, my + mh / 2, cx - 0.02, my + mh / 2, lw=0.8, mut=4)

    # ---------------------------------------------------------- 3 refinement
    p.panel(0.04, 3.08, 3.27, 1.52, facecolor=PANEL_FACE, edgecolor=PANEL_EDGE)
    p.text(L, 3.20, "3", size=7.5, weight="bold", color=C_NET)
    p.text(L + 0.09, 3.20, "Refine with a small network", size=5.6, color="0.15")

    tw, th = 0.42, 0.315
    p.text(L, 3.37, "the coarse frames and the query normal map, stacked as channels",
           size=4.2, color="0.45")
    ys = [3.46, 3.815, 4.17]
    # the two stills are spaced by d["gap"] purely so they read as two different
    # pictures; the network itself is fed consecutive frames
    p.img(L, ys[0], tw, th, load(f"{ASSETS}/{tag}_coarse_{mid - d['gap']:03d}.png"),
          edge="0.55")
    p.img(L, ys[1], tw, th, load(f"{ASSETS}/{tag}_coarse_{mid:03d}.png"), edge="0.55")
    p.img(L, ys[2], tw, th, load_normal(f"{ASSETS}/{tag}_querynorm_scale100.png"), edge=C_QRY)

    bx, ey, eh = 0.73, 3.66, 0.62
    ymid = ey + eh / 2
    net_block(p, bx, ey, 0.48, eh, "encoder", "4 stages")
    net_block(p, bx + 0.54, ey + 0.12, 0.38, eh - 0.24, "bottleneck")
    net_block(p, bx + 0.98, ey, 0.48, eh, "decoder", "skip links")
    p.arrow(L + tw + 0.02, ymid, bx - 0.015, ymid, lw=0.8, mut=4)
    p.arrow(bx + 0.48, ymid, bx + 0.535, ymid, lw=0.7, mut=3.5)
    p.arrow(bx + 0.92, ymid, bx + 0.975, ymid, lw=0.7, mut=3.5)

    ox = bx + 1.59
    p.img(ox, ymid - th / 2, tw, th, load(f"{ASSETS}/{tag}_pred_{mid:03d}.png"),
          edge=C_QRY, lw=0.9)
    p.arrow(bx + 1.46, ymid, ox - 0.015, ymid, lw=0.8, mut=4)
    p.text(ox, ymid + th / 2 + 0.08, "refined frame", size=4.4, color=C_QRY)

    film_group(p, bx, 4.36, 0.70, 0.15, 4, ey + eh + 0.02, note_x=bx + 0.76)
    p.save(out)


# ------------------------------------------------------------------ version 2
def v2(d, out):
    ret, md, tag, mid = d["ret"], d["md"], d["tag"], d["mid"]
    p = Page(3.35, 4.36)
    L = 0.09

    def head(y, n, title, right=None):
        p.text(L, y, n, size=7.0, weight="bold", color=C_NET)
        p.text(L + 0.09, y, title, size=5.6, color="0.15")
        if right:
            p.text(3.26, y, right, size=4.3, ha="right", color="0.45")
        p.line([(L, y + 0.09), (3.26, y + 0.09)], color="0.85", lw=0.6)

    # ------------------------------------------------------------ 1 retrieval
    head(0.14, "1", "Retrieve the most similar reference touch",
         f"{len(ret['show'])} of {ret['n_db']} references")
    qw, qh = 0.72, 0.54
    qx, qy = L, 0.36
    p.img(qx, qy, qw, qh, sensor_box(load_normal(ret["qimg"])), edge=C_QRY, lw=0.9)
    p.text(qx, qy + qh + 0.09, "query location:\ngeometry only", size=4.4, color=C_QRY)
    p.text(qx, qy + qh + 0.34, "compared using DINOv3\nfeatures (cosine similarity)",
           size=4.2, color="0.45")

    ex, ew, eh = 1.00, 0.40, 0.30
    p.text(ex, 0.30, "geometry (1x)", size=4.2, color="0.45")
    p.text(ex + ew + 0.04, 0.30, "the touch measured there", size=4.2, color="0.45")
    for k, (idx, sim) in enumerate(ret["show"]):
        y = 0.34 + k * 0.345
        top1 = idx == ret["top1"]
        col = C_REF if top1 else "0.72"
        p.img(ex, y, ew, eh, load_normal(f"{ASSETS}/{ret['tag']}_db{idx}_normal.png"), edge=col)
        p.img(ex + ew + 0.04, y, ew, eh, load(f"{ASSETS}/{ret['tag']}_db{idx}_touch.png"),
              edge=col)
        bx = ex + 2 * ew + 0.14
        sim_bar(p, bx, y + eh / 2 - 0.035, 0.82, 0.07, sim, col)
        p.text(bx + 0.86, y + eh / 2, f"{sim:.2f}", size=4.4, va="center",
               color=C_REF if top1 else "0.5", weight="bold" if top1 else "normal")
        if top1:
            p.text(bx, y + eh / 2 - 0.10, "best match", size=4.2, color=C_REF)

    # ------------------------------------------------------ 2 coarse alignment
    head(1.80, "2", "Line the two geometry renders up")
    mw, mh, my = 1.05, 0.79, 2.06
    mxl, mxr = L, L + mw + 0.28
    p.img(mxl, my, mw, mh, load_normal(f"{ASSETS}/{tag}_match_left.png"), edge=C_REF, lw=0.9)
    p.img(mxr, my, mw, mh, load_normal(f"{ASSETS}/{tag}_match_right.png"), edge=C_QRY, lw=0.9)
    shown = draw_matches(p, mxl, my, mxr, my, mw, mh, md)
    p.text(mxl, my - 0.06, "best match", size=4.6, color=C_REF)
    p.text(mxr, my - 0.06, "query", size=4.6, color=C_QRY)
    cx = 2.60
    p.img(cx, my + 0.20, 0.55, 0.41, load(f"{ASSETS}/{tag}_coarse_{mid:03d}.png"),
          edge="0.55", lw=0.9)
    p.arrow(mxr + mw + 0.02, my + mh / 2, cx - 0.02, my + mh / 2, lw=0.8, mut=4)
    p.text(cx, my + 0.14, "coarse transfer", size=4.4, color="0.3")
    p.text(mxl, my + mh + 0.09,
           f"geometry at 4x the sensor; {shown} of the {int(md['inlier'].sum())} agreeing "
           f"correspondences drawn, fixing one warp for the reference video", size=4.2,
           color="0.45")

    # ---------------------------------------------------------- 3 refinement
    head(3.10, "3", "Refine with a small network")
    tw, th = 0.44, 0.33
    ix, iy = L, 3.38
    p.img(ix, iy, tw, th, load(f"{ASSETS}/{tag}_coarse_{mid:03d}.png"), edge="0.55")
    p.img(ix, iy + th + 0.05, tw, th, load_normal(f"{ASSETS}/{tag}_querynorm_scale100.png"),
          edge=C_QRY)
    p.text(ix, iy + 2 * th + 0.17, "coarse frames +\nquery normal map", size=4.2,
           color="0.45")
    ymid = iy + th + 0.025
    x = ix + tw + 0.16
    for name, w in (("encoder", 0.46), ("bottleneck", 0.52), ("decoder", 0.46)):
        net_block(p, x, ymid - 0.15, w, 0.30, name)
        p.arrow(x + w + 0.005, ymid, x + w + 0.075, ymid, lw=0.7, mut=3.5)
        x += w + 0.08
    p.arrow(ix + tw + 0.02, ymid, ix + tw + 0.145, ymid, lw=0.7, mut=3.5)
    p.img(x, ymid - th / 2, tw, th, load(f"{ASSETS}/{tag}_pred_{mid:03d}.png"),
          edge=C_QRY, lw=0.9)
    p.text(x, ymid + th / 2 + 0.09, "refined frame", size=4.4, color=C_QRY)
    film_group(p, ix + tw + 0.16, ymid + 0.30, 0.62, 0.15, 3, ymid + 0.155,
               note_x=ix + tw + 0.84)
    p.save(out)


# ------------------------------------------------------------------ version 3
def v3(d, out):
    """Two-column-wide variant: the three steps run left to right."""
    ret, md, tag, mid = d["ret"], d["md"], d["tag"], d["mid"]
    p = Page(7.0, 2.34)
    cols, widths = [0.06, 2.42, 4.72], [2.24, 2.18, 2.22]
    titles = ["1   Retrieve the most similar reference touch",
              "2   Line the two geometry renders up",
              "3   Refine with a small network"]
    for x, w, t in zip(cols, widths, titles):
        p.panel(x, 0.06, w, 2.22, facecolor=PANEL_FACE, edgecolor=PANEL_EDGE)
        p.text(x + 0.08, 0.19, t, size=6.0, color="0.15")

    # --- 1 retrieval
    x0 = cols[0] + 0.08
    p.img(x0, 0.42, 0.62, 0.465, sensor_box(load_normal(ret["qimg"])), edge=C_QRY, lw=0.9)
    p.text(x0, 0.36, "query", size=4.8, color=C_QRY)
    p.text(x0, 0.96, "geometry only,\nnever touched", size=4.3, color="0.45")
    net_block(p, x0 + 0.76, 0.57, 0.34, 0.17, "DINOv3", size=4.8)
    p.arrow(x0 + 0.635, 0.655, x0 + 0.745, 0.655, lw=0.8, mut=4)
    p.text(x0 + 0.76, 0.86, "cosine similarity", size=4.2, color="0.45")
    ew = 0.42
    for k, (idx, sim) in enumerate(ret["show"]):
        x = x0 + k * (ew + 0.075)
        top1 = idx == ret["top1"]
        col = C_REF if top1 else "0.72"
        p.img(x, 1.24, ew, ew * 0.75, load_normal(f"{ASSETS}/{ret['tag']}_db{idx}_normal.png"),
              edge=col)
        p.img(x, 1.24 + ew * 0.75 + 0.03, ew, ew * 0.75,
              load(f"{ASSETS}/{ret['tag']}_db{idx}_touch.png"), edge=col)
        sim_bar(p, x, 1.24 + 2 * ew * 0.75 + 0.07, ew, 0.05, sim, col)
        p.text(x + ew / 2, 1.24 + 2 * ew * 0.75 + 0.18, f"{sim:.2f}", size=4.3,
               ha="center", color=C_REF if top1 else "0.5",
               weight="bold" if top1 else "normal")
        if top1:
            p.text(x + ew / 2, 1.19, "best match", size=4.3, ha="center", color=C_REF)
    p.text(x0, 1.13, f"reference database ({ret['n_db']} touches)", size=4.3,
           color="0.35")
    p.arrow(x0 + 1.10, 0.78, x0 + 1.30, 1.16, lw=0.8, mut=4,
            connection="arc3,rad=0.25")

    # --- 2 coarse alignment
    x0 = cols[1] + 0.08
    mw, mh, my = 0.95, 0.71, 0.46
    p.img(x0, my, mw, mh, load_normal(f"{ASSETS}/{tag}_match_left.png"), edge=C_REF, lw=0.9)
    p.img(x0 + mw + 0.12, my, mw, mh, load_normal(f"{ASSETS}/{tag}_match_right.png"),
          edge=C_QRY, lw=0.9)
    shown = draw_matches(p, x0, my, x0 + mw + 0.12, my, mw, mh, md, n_lines=12)
    p.text(x0, my - 0.06, "best match", size=4.6, color=C_REF)
    p.text(x0 + mw + 0.12, my - 0.06, "query", size=4.6, color=C_QRY)
    p.text(x0, my + mh + 0.10,
           f"{shown} of the {int(md['inlier'].sum())} agreeing SuperPoint + SuperGlue\n"
           f"correspondences drawn", size=4.2, color="0.45")
    p.img(x0 + 1.14, 1.44, 0.55, 0.41, load(f"{ASSETS}/{tag}_coarse_{mid:03d}.png"),
          edge="0.55", lw=0.9)
    p.text(x0 + 0.10, 1.54, "one fitted warp\nis applied to the\nreference video", size=4.4,
           color="0.3")
    p.text(x0 + 1.14, 1.39, "coarse transfer", size=4.4, color="0.3")

    # --- 3 refinement
    x0 = cols[2] + 0.08
    tw, th = 0.36, 0.27
    p.img(x0, 0.62, tw, th, load(f"{ASSETS}/{tag}_coarse_{mid:03d}.png"), edge="0.55")
    p.img(x0, 0.62 + th + 0.04, tw, th, load_normal(f"{ASSETS}/{tag}_querynorm_scale100.png"),
          edge=C_QRY)
    p.text(x0, 0.62 + 2 * th + 0.15, "coarse frames +\nquery normal map", size=4.2,
           color="0.45")
    ey, eh = 0.62, 0.58
    ymid = ey + eh / 2
    bx = x0 + tw + 0.13
    net_block(p, bx, ey, 0.34, eh, "encoder", "4 stages", size=4.3)
    net_block(p, bx + 0.39, ey + 0.12, 0.30, eh - 0.24, "bottleneck", size=3.9)
    net_block(p, bx + 0.74, ey, 0.34, eh, "decoder", "skip links", size=4.3)
    p.arrow(x0 + tw + 0.02, ymid, bx - 0.015, ymid, lw=0.7, mut=3.5)
    p.arrow(bx + 0.34, ymid, bx + 0.385, ymid, lw=0.7, mut=3.5)
    p.arrow(bx + 0.69, ymid, bx + 0.735, ymid, lw=0.7, mut=3.5)
    ox = bx + 1.16
    p.img(ox, ymid - th / 2, tw, th, load(f"{ASSETS}/{tag}_pred_{mid:03d}.png"),
          edge=C_QRY, lw=0.9)
    p.arrow(bx + 1.08, ymid, ox - 0.015, ymid, lw=0.7, mut=3.5)
    p.text(ox, ymid + th / 2 + 0.09, "refined frame", size=4.4, color=C_QRY)
    film_group(p, bx, ey + eh + 0.20, 0.60, 0.15, 4, ey + eh + 0.02)
    p.save(out)


def figure_inputs(tag, ret_tag, n_db, mid, gap=1):
    """Everything a version function draws, resolved from the cached assets.

    Kept as one function so that anything else which needs to know exactly what
    a figure shows -- export_method_assets.py, for one -- asks the same question
    and cannot drift from what is actually drawn.
    """
    r = pickle.load(open(f"{ASSETS}/{ret_tag}_retrieval.pkl", "rb"))
    scores = [(i, float(s)) for i, s in zip(r["idxs"], r["loo_scores"]) if np.isfinite(s)]
    scores.sort(key=lambda t: -t[1])
    # the best match plus a spread of weaker ones, so the bars differ visibly
    picks = [scores[0]] + [scores[int(round(q * (len(scores) - 1)))]
                           for q in np.linspace(0.35, 1.0, n_db - 1)]
    qimg = f"{ASSETS}/{ret_tag}_query_normal.png"
    if not os.path.exists(qimg):        # leave-one-out variant keeps the query in the set
        qimg = f"{ASSETS}/{ret_tag}_db{r['query_idx']}_normal.png"
    ret = dict(tag=ret_tag, qimg=qimg, top1=r["order"][0], show=picks,
               n_db=len(scores))

    md = pickle.load(open(f"{ASSETS}/{tag}_matches.pkl", "rb"))
    md["tag"] = tag
    return dict(ret=ret, md=md, tag=tag, mid=mid, gap=gap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1000_7")
    ap.add_argument("--ret_tag", default="bench1000_7")
    ap.add_argument("--n_db", type=int, default=4, help="reference touches shown")
    ap.add_argument("--mid", type=int, default=25, help="frame index used for stills")
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--name", default="method",
                    help="output stem; files are <name>_<version>.pdf/.png")
    ap.add_argument("--panel", default="shaded", choices=["shaded", "white"],
                    help="fill of the box behind each step")
    ap.add_argument("--frame_gap", type=int, default=1,
                    help="how many frames apart the two coarse stills in step 3 are. "
                         "The network is fed consecutive frames (dataset.py builds "
                         "[previous, current]), so 1 is what it actually sees; a larger "
                         "gap only makes the two thumbnails easier to tell apart")
    args = ap.parse_args()

    if args.panel == "white":
        global PANEL_FACE, PANEL_EDGE
        PANEL_FACE, PANEL_EDGE = "white", "#c8ccd2"

    d = figure_inputs(args.tag, args.ret_tag, args.n_db, args.mid, args.frame_gap)
    os.makedirs(OUT, exist_ok=True)
    for v in args.versions:
        {"v1": v1, "v2": v2, "v3": v3}[v](d, f"{OUT}/{args.name}_{v}")


if __name__ == "__main__":
    main()
