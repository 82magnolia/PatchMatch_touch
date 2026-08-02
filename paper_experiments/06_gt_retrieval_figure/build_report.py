"""HTML report for the ground-truth-retrieval qualitative figure (fig_gt_retrieval)."""
import base64
import os
import pickle

import cv2

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
JOB = f"{ROOT}/log/paper_job06_gt_retrieval_figure"
METRICS = f"{ROOT}/log/paper_job02_gt_retrieval_figure_normalmatch/per_touch_metrics.pkl"
TOUCHES = [("974_5", "Candidate 1, row 3"), ("978_4", "Candidate 2, row 1"),
           ("969_3", "Candidate 2, row 2"), ("970_7", "Candidate 3, row 1"),
           ("995_5", "Candidate 3, row 3")]


def b64(p, max_width=1500, quality=85):
    im = cv2.imread(p)
    if im.shape[1] > max_width:
        h = int(round(im.shape[0] * max_width / im.shape[1]))
        im = cv2.resize(im, (max_width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def fig(path, width="100%"):
    if not os.path.exists(path):
        return f"<p><i>{os.path.basename(path)} missing</i></p>"
    return (f'<img style="width:{width};display:block;border:1px solid #ddd" '
            f'src="data:image/jpeg;base64,{b64(path)}">')


scores = {f"{r['obj']}_{r['pair']}": r for r in pickle.load(open(METRICS, "rb"))}
rows = "".join(
    f"<tr><td>{i}</td><td>object {t.split('_')[0]}, touch {t.split('_')[1]}</td>"
    f"<td>{src}</td><td>{scores[t]['coarse']['PSNR']:.1f}</td>"
    f"<td>{scores[t]['refined']['PSNR']:.1f}</td></tr>"
    for i, (t, src) in enumerate(TOUCHES, start=1))
mean_c = sum(scores[t]["coarse"]["PSNR"] for t, _ in TOUCHES) / len(TOUCHES)
mean_r = sum(scores[t]["refined"]["PSNR"] for t, _ in TOUCHES) / len(TOUCHES)

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Job 6 &mdash; ground-truth retrieval figure</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      max-width:1150px;margin:2rem auto;padding:0 1.2rem;line-height:1.6;color:#1b1b1f}}
 h1{{border-bottom:2px solid #333;padding-bottom:.4rem}}
 h2{{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 table{{border-collapse:collapse;margin:.8rem 0;font-size:.93rem}}
 th,td{{border:1px solid #ddd;padding:.35rem .7rem;text-align:left}}
 th{{background:#f7f7f9}}
 code{{background:#f4f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 figure{{margin:1.4rem 0}} figcaption{{color:#555;font-size:.9rem;margin-top:.4rem}}
 .note{{background:#eef6ff;border-left:4px solid #3d7fd1;padding:.7rem 1rem;margin:1rem 0}}
 pre{{background:#f6f6f8;padding:.7rem 1rem;overflow-x:auto;font-size:.86rem}}
</style></head><body>

<h1>Job 6 &mdash; the ground-truth-retrieval qualitative figure</h1>

<p>The figure for <code>paper_source/figures/fig_gt_retrieval.tex</code>: five touch locations
down the page, six streams across &mdash; the reference touch, the two surface normal renders, the
coarse transfer, our refined prediction, and the ground-truth touch. The five rows are the ones
picked out of <code>log/paper_figure_candidates.html</code>.</p>

<figure>{fig(f"{JOB}/gt_retrieval.png")}
<figcaption>White page, black text, a white gap between every frame, and rows separated by twice
the gap between columns.</figcaption></figure>

<h2>The rows</h2>
<table><tr><th>row</th><th>touch</th><th>picked from</th><th>coarse (dB)</th>
<th>refined (dB)</th></tr>{rows}</table>
<p>Mean over the five rows: {mean_c:.1f} dB coarse, {mean_r:.1f} dB refined. For context, the
whole 400-touch benchmark averages 22.9 dB coarse and 31.3 dB refined, so these rows are better
than typical &mdash; they were chosen to be legible, not representative, and the table in the
paper is what carries the average case.</p>

<h2>Drawing choices</h2>
<ul>
 <li><b>Outlines on the geometry columns only.</b> The tactile frames fill their cell edge to
     edge, so a box around them would just add clutter. The normal renders are mostly pale
     background, and without a boundary they bleed into the page.</li>
 <li><b>White background inside the normal renders.</b> Taxim writes empty space as exactly black;
     repainting it white matches the page and keeps the figure light. Only what the reader sees is
     repainted &mdash; retrieval and matching still run on the untouched renders.</li>
 <li><b>Four times the sensor footprint</b> in those two columns, with the sensor's own footprint
     boxed in red, since that wider view is what the correspondences are computed on.
     <code>--normal_scale 100</code> would show the 1x view instead.</li>
</ul>

<h2>Files</h2>
<pre>log/paper_job06_gt_retrieval_figure/
  gt_retrieval.pdf, gt_retrieval.png     the figure
  assets_pdf/gt_retrieval/               one PNG per cell, named
                                         rowN_&lt;object&gt;_&lt;touch&gt;_colM_&lt;stream&gt;.png</pre>

<h2>Regenerating</h2>
<pre><code>conda activate pm_touch
python paper_experiments/06_gt_retrieval_figure/make_gt_figure.py \\
    --touches 974_5 978_4 969_3 970_7 995_5
python paper_experiments/06_gt_retrieval_figure/build_report.py
</code></pre>
<p>Any of the 400 benchmark touches can be swapped in by name; the per-cell PNGs for all of them
are in <code>log/paper_job02_gt_retrieval_figure_normalmatch/assets/</code>.</p>

<footer style="margin-top:2.5rem;color:#777;font-size:.85rem">
Assets: <code>log/paper_job06_gt_retrieval_figure/</code>
</footer>
</body></html>
"""
open(f"{JOB}/report.html", "w").write(HTML)
print("wrote", f"{JOB}/report.html")
