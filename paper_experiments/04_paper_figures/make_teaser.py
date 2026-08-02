"""Teaser figure: the tactile analogy, in the spirit of the Image Analogies teaser.

The story the figure has to tell in one column:

    geometry here : the touch we measured here
      ::
    geometry over there : the touch our method predicts over there

Three layouts of that same story are produced, so the paper can pick one:

  v1  analogy    two rows, "A : A'  ::  B : B'", nothing else
  v2  verified   the same two rows plus the held-out ground-truth video
  v3  columns    portrait variant: reference / predicted / ground truth as three
                 columns, with video time running down the page

All pixels come from log/paper_job04_paper_figures/assets (see prep_assets.py):
real benchmark videos and real refinement-network output.
"""
import argparse
import os
import pickle

from figlib import ASSETS, C_GT, C_QRY, C_REF, Page, load, load_normal, sensor_box

OUT = "/home/junhokim/Projects/PatchMatch_gpu/log/paper_job04_paper_figures"

HDR_GEOM = "geometry at the sensor"
HDR_VID = "tactile normal video — press in, then withdraw"


def frames(tag, kind, idxs):
    return [load(f"{ASSETS}/{tag}_{kind}_{i:03d}.png") for i in idxs]


def norm_render(tag, kind):
    """The 4x-field-of-view geometry render, background whitened, sensor footprint boxed."""
    who = "refnorm" if kind == "ref" else "querynorm"
    return sensor_box(load_normal(f"{ASSETS}/{tag}_{who}_scale25.png"))


# ------------------------------------------------------------------ version 1
def v1(tag, idxs, meta, out):
    """Two-column analogy: geometry : video, given over predicted."""
    n = len(idxs)
    p = Page(7.0, 2.30)
    iw, ih, gap = 0.90, 0.675, 0.03
    x_norm = 0.09
    x_colon = x_norm + iw + 0.06
    x_vid = x_colon + 0.18
    vid_w = n * iw + (n - 1) * gap
    rows = [(0.44, "ref", C_REF, "Given —", "a touch we already made here"),
            (1.36, "pred", C_QRY, "Predicted —", "what a touch would feel like here")]

    p.text(x_norm, 0.17, HDR_GEOM, size=5.6, color="0.35")
    p.text(x_vid, 0.17, HDR_VID, size=5.6, color="0.35")
    p.text(x_vid + vid_w - 0.62, 0.17, "time", size=5.2, ha="right", color="0.5",
           style="italic")
    p.arrow(x_vid + vid_w - 0.55, 0.17, x_vid + vid_w, 0.17, color="0.6", lw=0.7, mut=4)

    for y, kind, col, lab, sub in rows:
        p.text(x_norm, y - 0.10, f"{lab} {sub}", size=6.0, color=col)
        p.img(x_norm, y, iw, ih, norm_render(tag, kind), edge=col, lw=0.9)
        for k, im in enumerate(frames(tag, kind, idxs)):
            p.img(x_vid + k * (iw + gap), y, iw, ih, im, edge=col, lw=0.9)
        p.text(x_colon + 0.09, y + ih / 2, ":", size=12, ha="center", color="0.35")

    p.text(x_colon + 0.09, rows[0][0] + ih + 0.045, "::", size=9.5, ha="center",
           color="0.35")
    p.text(x_norm, 2.10,
           "Red box: the patch the sensor covers; the wider view around it is what our method "
           "compares.", size=4.8, color="0.45")
    if meta.get("footnote"):
        p.text(x_norm, 2.19, meta["footnote"], size=4.8, color="0.45")
    p.save(out)


# ------------------------------------------------------------------ version 2
def v2(tag, idxs, meta, out):
    p = Page(3.35, 2.66)
    iw, ih, gap = 0.72, 0.54, 0.03
    x_norm = 0.08
    x_colon = x_norm + iw + 0.055
    x_vid = x_colon + 0.16
    vid_w = 3 * iw + 2 * gap
    rows = [(0.42, "ref", C_REF, "Given —", "a touch we already made here"),
            (1.18, "pred", C_QRY, "Predicted —", "what a touch would feel like here"),
            (1.94, "gt", C_GT, "Truth —", "the real touch, held out for checking")]

    p.text(x_norm, 0.15, HDR_GEOM, size=4.8, color="0.35")
    p.text(x_vid, 0.15, HDR_VID, size=4.8, color="0.35")

    for y, kind, col, lab, sub in rows:
        p.text(x_norm, y - 0.07, f"{lab} {sub}", size=5.2, color=col)
        if kind == "gt":
            # a third geometry cell would just repeat the query render, so say that
            # instead of implying the method gets another input
            p.box(x_norm, y, iw, ih, facecolor="none", edgecolor="0.78", lw=0.6,
                  dashed=True)
            p.text(x_norm + iw / 2, y + ih / 2, "same location\nas the row above",
                   size=4.4, ha="center", color="0.55", style="italic")
        else:
            p.img(x_norm, y, iw, ih, norm_render(tag, kind), edge=col, lw=0.9)
        for k, im in enumerate(frames(tag, kind, idxs)):
            p.img(x_vid + k * (iw + gap), y, iw, ih, im, edge=col, lw=0.9)
        p.text(x_colon + 0.08, y + ih / 2, ":", size=10, ha="center", color="0.35")

    p.text(x_colon + 0.08, rows[0][0] + ih + 0.055, "::", size=8.0, ha="center",
           color="0.35")
    p.text(x_norm, 2.58,
           f"Benchmark object {meta['obj']}, touch {meta['pair']}; this prediction scores "
           f"{meta.get('psnr_refined', float('nan')):.1f} dB against the truth.",
           size=4.2, color="0.45")
    p.save(out)


# ------------------------------------------------------------------ version 3
def v3(tag, idxs, meta, out):
    p = Page(3.35, 3.70)
    iw, ih = 0.93, 0.70
    gapx, gapy = 0.10, 0.045
    x0 = 0.30
    y_norm = 0.40
    y_vid = 1.28
    cols = [("ref", C_REF, "reference touch", "given"),
            ("pred", C_QRY, "our prediction", "never touched"),
            ("gt", C_GT, "ground truth", "held out")]

    for c, (kind, col, title, sub) in enumerate(cols):
        x = x0 + c * (iw + gapx)
        p.text(x + iw / 2, 0.14, title, size=5.4, ha="center", weight="bold", color=col)
        p.text(x + iw / 2, 0.24, sub, size=4.6, ha="center", color="0.45")
        if kind == "gt":
            p.box(x, y_norm, iw, ih, facecolor="none", edgecolor="0.78", lw=0.6,
                  dashed=True)
            p.text(x + iw / 2, y_norm + ih / 2, "same location as\nthe middle column",
                   size=4.4, ha="center", color="0.55", style="italic")
        else:
            p.img(x, y_norm, iw, ih, norm_render(tag, kind), edge=col, lw=0.9)
        for k, im in enumerate(frames(tag, kind, idxs)):
            p.img(x, y_vid + k * (ih + gapy), iw, ih, im, edge=col, lw=0.9)

    p.text(0.22, y_norm + ih / 2, "geometry", size=4.8, ha="center", va="center",
           color="0.3", rotation=90)
    p.text(0.22, y_vid + (3 * ih + 2 * gapy) / 2, "touch over time", size=4.8,
           ha="center", va="center", color="0.3", rotation=90)
    p.arrow(0.115, y_vid + 0.08, 0.115, y_vid + 3 * ih + 2 * gapy - 0.02,
            color="0.55", lw=0.8, mut=4)

    xa = x0 + iw + 0.012
    p.arrow(xa, y_norm + ih / 2, xa + gapx - 0.024, y_norm + ih / 2,
            color="0.4", lw=1.0, mut=5)
    p.text(x0 + iw + gapx / 2, y_norm - 0.05, "by analogy", size=4.4, ha="center",
           color="0.4", style="italic")
    p.text(x0, 3.62,
           f"Benchmark object {meta['obj']}, touch {meta['pair']}; red box = the patch "
           f"the sensor covers.", size=4.2, color="0.45")
    p.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="951_7", help="asset prefix written by prep_assets.py")
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[4, 12, 21, 29, 38, 46],
                    help="frame indices to show; v1 shows all of them, the "
                         "one-column versions show three")
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    args = ap.parse_args()

    meta = pickle.load(open(f"{ASSETS}/{args.tag}_meta.pkl", "rb"))
    if "psnr_refined" not in meta:      # ground-truth-retrieval touches score elsewhere
        recs = pickle.load(open("/home/junhokim/Projects/PatchMatch_gpu/log/"
                                "paper_job02_gt_retrieval_figure_normalmatch/"
                                "per_touch_metrics.pkl", "rb"))
        for r in recs:
            if r["obj"] == meta["obj"] and r["pair"] == meta["pair"]:
                meta["psnr_refined"] = r["refined"]["PSNR"]
                meta["psnr_coarse"] = r["coarse"]["PSNR"]
    if meta.get("pinned_ref") is not None:
        moved = (f" The query has been moved {meta['moved_mm']:.1f} mm across the surface "
                 f"and re-simulated." if meta.get("moved_mm") else "")
        meta["footnote"] = (f"Object {meta['obj']}: the reference is touch "
                            f"{meta['pinned_ref']}, held fixed rather than retrieved."
                            + moved)
    elif meta.get("ref_idx") is not None:
        meta["footnote"] = (f"Object {meta['obj']}: touch {meta['ref_idx']} was retrieved "
                            f"by the method itself as the reference for query touch "
                            f"{meta['pair']}.")

    os.makedirs(OUT, exist_ok=True)
    fns = {"v1": v1, "v2": v2, "v3": v3}
    few = args.frames if len(args.frames) <= 3 else [args.frames[0],
                                                    args.frames[len(args.frames) // 2],
                                                    args.frames[-1]]
    for v in args.versions:
        fns[v](args.tag, args.frames if v == "v1" else few, meta,
               f"{OUT}/teaser_{v}_{args.tag}")


if __name__ == "__main__":
    main()
