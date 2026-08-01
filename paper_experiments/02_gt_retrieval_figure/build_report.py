"""HTML + Markdown report for the ground-truth-retrieval qualitative figure job."""
import base64
import os
import pickle

import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OUT = f"{ROOT}/log/paper_job02_gt_retrieval_figure"
HERE = os.path.dirname(os.path.abspath(__file__))

recs = pickle.load(open(f"{OUT}/per_touch_metrics.pkl", "rb"))
summ = pickle.load(open(f"{OUT}/summary.pkl", "rb"))
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode("ascii")


def img_tag(p, w="100%"):
    if not os.path.exists(p):
        return "<i>missing</i>"
    return f'<img style="width:{w};display:block" src="data:image/png;base64,{b64(p)}">'


def stat(key, m):
    return np.array([r[key][m] for r in recs])


rows = ""
for label, key in (("Coarse transfer (feature matching only)", "coarse"),
                   ("Refined transfer (ours, refinement network)", "refined")):
    cells = "".join(f"<td>{stat(key, m).mean():.4f}</td>" if m in ("SSIM", "LPIPS", "MSE")
                    else f"<td>{stat(key, m).mean():.2f}</td>" for m in METRICS)
    rows += f"<tr><th>{label}</th>{cells}</tr>"
gain = stat("refined", "PSNR").mean() - stat("coarse", "PSNR").mean()

# per-object breakdown
objs = sorted({r["obj"] for r in recs})
obj_rows = ""
for o in objs:
    rs = [r for r in recs if r["obj"] == o]
    obj_rows += (f"<tr><td>{o}</td><td>{len(rs)}</td>"
                 f"<td>{np.mean([r['coarse']['PSNR'] for r in rs]):.2f}</td>"
                 f"<td>{np.mean([r['refined']['PSNR'] for r in rs]):.2f}</td>"
                 f"<td>{np.mean([r['refined']['PSNR'] - r['coarse']['PSNR'] for r in rs]):+.2f}</td></tr>")

fig_path = f"{OUT}/figure_gt_retrieval_3x6.png"

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Job 2 &mdash; Ground-truth retrieval qualitative figure</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      max-width:1150px;margin:2rem auto;padding:0 1.2rem;line-height:1.6;color:#1b1b1f}}
 h1{{border-bottom:2px solid #333;padding-bottom:.4rem}}
 h2{{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.93rem}}
 th,td{{border:1px solid #ddd;padding:.42rem .7rem;text-align:left}}
 td{{text-align:right}} th:first-child{{text-align:left}}
 thead th{{background:#f0f2f5}}
 code{{background:#f4f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 .note{{background:#eef6ff;border-left:4px solid #3d7fd1;padding:.7rem 1rem;margin:1rem 0}}
 figure{{margin:1.2rem 0}} figcaption{{color:#555;font-size:.9rem;margin-top:.4rem}}
 .scroll{{overflow-x:auto}}
</style></head><body>

<h1>Job 2 &mdash; Ground-truth retrieval: qualitative figure</h1>
<p>This job produces the figure that shows what our method predicts, touch by touch, on
the ground-truth-retrieval benchmark. "Ground-truth retrieval" means the correct reference
touch is handed to the method, so the figure isolates the two stages we care about here:
the coarse alignment by feature matching, and the refinement network that cleans it up.</p>

<h2>What was run</h2>
<table>
<tr><th>Coarse alignment</th><td style="text-align:left">SuperPoint keypoints + SuperGlue matching on the
    curvature renders at 4x the sensor footprint, followed by a median translation offset
    (<code>main_retrieval_transfer_feat_match.py</code>)</td></tr>
<tr><th>Refinement network</th><td style="text-align:left">ReBotNet-S with the query normal map concatenated to the
    input and sinusoidal temporal FiLM &mdash; <code>log/rebot_checkpoints_S_geomcat_film/best.pth</code>,
    epoch {summ['epoch']}</td></tr>
<tr><th>Evaluated on</th><td style="text-align:left">{summ['n_objects']} held-out objects (951&ndash;1000),
    {summ['n_touches']} touch locations, 50 frames each</td></tr>
<tr><th>Hardware</th><td style="text-align:left">one RTX 2080 Ti</td></tr>
</table>

<h2>The figure</h2>
<figure>{img_tag(fig_path)}
<figcaption>Three touch locations (rows) &times; six views (columns). Column 1 is the middle
frame of the reference touch we copy from; columns 2 and 3 are the rendered surface normals
at the reference and query sensor poses, shown at 4x the sensor footprint &mdash; that is the
field of view the feature matching actually runs at, and it makes the object recognisable
(a lamp, a stool, a step-ladder in these three rows). The <b style="color:#c00">red box</b>
marks the 1x sensor footprint, i.e. the patch of surface the tactile columns actually see.
Column 4 is the coarse transfer produced
by feature matching alone; column 5 is our refined prediction; column 6 is the true tactile
reading at the query pose. All tactile frames are the middle frame of the press, which is the
deepest point of contact. The three rows are drawn from the 95th, 50th and 20th percentile of
the refinement gain, so the figure is not only showing the cases that work best.</figcaption></figure>

<div class="note">
<b>How to read the tactile images.</b> They are surface-normal maps: the colour of a pixel
encodes which way the gel surface is tilted at that point. Flat, untouched gel is the uniform
light blue-violet; a ridge or edge pressing into the gel shows up as a sharp colour change.
</div>

<h2>How much the refinement network helps</h2>
<p>Averaged over all {summ['n_touches']} evaluation touches and all 50 frames of each touch.
PSNR and SSIM are higher-is-better; LPIPS and MSE are lower-is-better.</p>
<table><thead><tr><th>Stage</th><th>PSNR (dB)</th><th>SSIM</th><th>LPIPS</th><th>MSE</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>The refinement network adds <b>{gain:+.2f} dB</b> of PSNR on top of the coarse alignment.</p>

<div class="note">These are our method's numbers only. The comparison against the baselines
(Tactile Normal Quilting, implicit neural representations, TaRF) is a separate job that runs
on the other machine; this table is the "ours" row of that comparison, computed here because
the coarse transfer had to be run locally anyway.</div>

<h2>Per-object breakdown</h2>
<div class="scroll">
<table><thead><tr><th>Object</th><th>Touches</th><th>Coarse PSNR</th><th>Refined PSNR</th><th>Gain</th></tr></thead>
<tbody>{obj_rows}</tbody></table></div>

<h2>Where the assets are</h2>
<ul>
<li><code>log/paper_job02_gt_retrieval_figure/figure_gt_retrieval_3x6.png</code> &mdash; the stitched figure</li>
<li><code>log/paper_job02_gt_retrieval_figure/assets/&lt;object&gt;_&lt;touch&gt;_0N_*.png</code> &mdash; every
    individual cell, at full resolution, for all {summ['n_touches']} touches (not just the three shown).
    For the touches used in the figure, the normal renders are additionally saved at all three
    fields of view (<code>..._scale100/50/25.png</code> = 1x / 2x / 4x), each with a
    <code>_box</code> variant carrying the red 1x-footprint outline</li>
<li><code>log/paper_job02_gt_retrieval_figure/per_touch_metrics.pkl</code> &mdash; per-touch metrics</li>
<li><code>paper_experiments/02_gt_retrieval_figure/</code> &mdash; the scripts and the Markdown fact sheet</li>
</ul>
</body></html>
"""

MD = f"""# Job 2 — Ground-truth retrieval qualitative figure (and our-method metrics)

## Setup

- Coarse alignment: SuperPoint + SuperGlue on curvature renders at 4x the sensor footprint,
  median translation offset (`main_retrieval_transfer_feat_match.py`).
- Refinement: ReBotNet-S, query normal map concatenated + sinusoidal temporal FiLM,
  `log/rebot_checkpoints_S_geomcat_film/best.pth` (epoch {summ['epoch']}).
- Evaluation set: objects 951–1000, {summ['n_touches']} touch locations, 50 frames each.
- Hardware: one RTX 2080 Ti.

## Table — ours, ground-truth retrieval benchmark

Averaged over all {summ['n_touches']} touches and all frames.

| Method | PSNR (dB) | SSIM | LPIPS | MSE |
|---|---|---|---|---|
| Ours (coarse transfer only) | {stat('coarse','PSNR').mean():.2f} | {stat('coarse','SSIM').mean():.4f} | {stat('coarse','LPIPS').mean():.4f} | {stat('coarse','MSE').mean():.4f} |
| Ours (refined) | {stat('refined','PSNR').mean():.2f} | {stat('refined','SSIM').mean():.4f} | {stat('refined','LPIPS').mean():.4f} | {stat('refined','MSE').mean():.4f} |

Refinement gain: **{gain:+.2f} dB** PSNR.

Standard deviation across touches (for error bars if wanted):
PSNR coarse {stat('coarse','PSNR').std():.2f}, refined {stat('refined','PSNR').std():.2f}.

## Figure

`log/paper_job02_gt_retrieval_figure/figure_gt_retrieval_3x6.png` — 3 rows (touch locations)
x 6 columns:

1. reference touch, middle frame
2. reference surface-normal render (4x field of view, 1x sensor footprint boxed in red)
3. query surface-normal render (4x field of view, 1x sensor footprint boxed in red)
4. coarse transfer, middle frame
5. refined transfer (ours), middle frame
6. ground-truth query touch, middle frame

Per-cell PNGs for **all** {summ['n_touches']} evaluation touches are in
`log/paper_job02_gt_retrieval_figure/assets/`, named
`<object>_<touch>_01_ref_touch.png` … `_06_gt_query.png`, so a different row selection can be
made without re-running anything.

## Caveats

- The middle frame of the 50-frame press is used as the representative frame; because the
  press schedule is press-in-then-withdraw, this is also the deepest-contact frame.
- These are our method's numbers only. Baseline comparisons run on the other machine.
"""

open(os.path.join(HERE, "report.html"), "w").write(HTML)
open(os.path.join(HERE, "results.md"), "w").write(MD)
open(f"{OUT}/report.html", "w").write(HTML)
print("wrote report.html + results.md")
print(f"coarse PSNR {stat('coarse','PSNR').mean():.2f} -> refined {stat('refined','PSNR').mean():.2f}")
