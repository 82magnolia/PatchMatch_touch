"""HTML report for the 3D-reconstruction figure (fig_recon)."""
import base64
import os
import pickle

import cv2

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
JOB = f"{ROOT}/log/paper_job05_recon_figure"
CHOSEN, ALT = 11, 13
TAG = f"993_2_ref_cand{CHOSEN}"
PDFS = [("993_2", 993, 2, "a monitor arm"), ("994_3", 994, 3, "a metal stool")]


def b64(p, max_width=1500, quality=85):
    im = cv2.imread(p)
    if im.shape[1] > max_width:
        h = int(round(im.shape[0] * max_width / im.shape[1]))
        im = cv2.resize(im, (max_width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def fig(path, width="100%"):
    if not os.path.exists(path):
        return f"<p><i>{os.path.basename(path)} not generated</i></p>"
    return (f'<img style="width:{width};display:block;border:1px solid #ddd" '
            f'src="data:image/jpeg;base64,{b64(path)}">')


recs = {r["cand"]: r for r in pickle.load(open(f"{JOB}/refsweep/candidates.pkl", "rb"))}
c, a = recs[CHOSEN], recs[ALT]
ranked = sorted(recs.values(), key=lambda r: -r["psnr_refined"])
rows = "".join(
    f"<tr><td>{r['cand']}</td><td>{r['moved_mm']:.1f} mm</td>"
    f"<td>{r['direction_deg']}&deg;</td><td>{r['psnr_coarse']:.1f}</td>"
    f"<td>{r['psnr_refined']:.1f}</td></tr>" for r in ranked)

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Job 5 &mdash; 3D reconstruction figure</title>
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

<h1>Job 5 &mdash; the 3D reconstruction figure</h1>

<p>The figure for <code>paper_source/figures/fig_recon.tex</code>, on two touches. Columns are
frames of one press; the three rows are our prediction at the query location, the 3D relief
obtained by integrating those predicted normals into a heightmap, and the colour image a
camera-based tactile sensor would have measured, simulated from that same heightmap. Each row uses
only the row above it &mdash; no ground truth anywhere in the chain.</p>

<div class="note">
Object 993 is a monitor arm; its touch 2 was chosen because the prediction keeps the run of fine
slats on the arm, and that detail survives the transfer, the Poisson integration and the optical
simulation. Object 994 is a metal stool, whose touch 3 is a clean curved edge and the
higher-scoring of the two. Both use the benchmark's own reference pairing, so their numbers are
the ones in <code>log/paper_figure_candidates.html</code>: 24.0 &rarr; 29.9 dB for 993, and
25.2 &rarr; 37.7 dB for 994.
</div>

<h2>The figures</h2>

<p>Two of them, both as PDF with selectable text so they paste straight into a document:
<code>recon_993_2.pdf</code> and <code>recon_994_3.pdf</code> in
<code>log/paper_job05_recon_figure/</code>. Three rows &mdash; the prediction, the 3D relief
integrated from it, and the simulated colour image &mdash; on white, with black labels and white
gaps between frames. There is no reference row: where the prediction was transferred from belongs
in the caption.</p>

<p>The height axis of the relief is drawn at 0.16 of the surface's width, against 0.42 in the
earlier version, and the shading is recomputed at the same exaggeration so a flatter surface also
casts flatter shadows. <code>--z_scale</code> changes it.</p>

<p>The relief's shading was washed out after the surface was flattened: squashing the height axis
also squashes the range of surface angles, so everything landed in a narrow band of greys. Two
changes fix it. The shading strength is back at the value the un-flattened figure used
(<code>--relief_light</code>, default 9.0) rather than being scaled down with the height, and the
shading is stretched about its mid-tone by <code>--relief_contrast</code> (default 1.35) with the
darkest facets allowed down to <code>--relief_ambient</code> 0.10. Pushing the stretch much past
1.5 starts to blow out the lit faces, so the gain does most of the work and the stretch only
finishes it.</p>

<p>The relief keeps its black backdrop &mdash; the grey surface reads far better against black than
against the white of the page. Only that row is dark; the page, the labels and the gaps are white
and black as before. <code>--relief_bg white</code> switches it if that is ever wanted.</p>

<figure>{fig(f"{JOB}/recon_993_2.png")}<figcaption>Object 993 (a monitor arm), touch 2.</figcaption></figure><figure>{fig(f"{JOB}/recon_994_3.png")}<figcaption>Object 994 (a metal stool), touch 3.</figcaption></figure>

<h2>An earlier four-row version, with a reference row</h2>
<figure>{fig(f"{JOB}/figure_{TAG}.png")}
<figcaption>Object 993, query touch 2, transferred from a reference touch {c['moved_mm']:.1f} mm
away. Coarse transfer {c['psnr_coarse']:.1f} dB, refined {c['psnr_refined']:.1f} dB.</figcaption>
</figure>

<h2>Choosing where the reference touch was taken</h2>

<p>The query stays exactly as the benchmark has it. What moved is the <b>reference</b>: the touch
the prediction is transferred from was re-simulated at sixteen nearby spots &mdash; four distances
(0.9, 1.8, 3.3, 5.3 mm across the surface) in four directions &mdash; and each was run through the
full coarse-alignment and refinement chain against the fixed query. That way the figure is not
quietly transferring from a touch taken at the very same place.</p>

<p>Sixteen candidates, best first:</p>
<table><tr><th>candidate</th><th>moved</th><th>direction</th><th>coarse (dB)</th>
<th>refined (dB)</th></tr>{rows}</table>

<p><b>Candidate {CHOSEN}</b> was chosen: the highest score of the sweep, and its reference touch
shows the bracket clearly offset from where the query sits, so rows 1 and 2 are visibly different
places rather than near-copies. Candidate {ALT} ({a['moved_mm']:.1f} mm,
{a['psnr_refined']:.1f} dB) is kept as an alternative with a larger shift.</p>

<figure>{fig(f"{JOB}/refsweep/candidates.png", "62%")}
<figcaption>Every candidate. Left to right: the shifted reference's geometry render, its tactile
frame at the deepest press, the coarse transfer, and the fixed ground truth.</figcaption></figure>

<figure>{fig(f"{JOB}/figure_993_2_ref_cand{ALT}.png")}
<figcaption>The alternative, at a {a['moved_mm']:.1f} mm shift.</figcaption></figure>

<h2>The object</h2>
<figure>{fig(f"{JOB}/object_renders/993_sheet.png")}
<figcaption>Six views of object 993 with its own texture. It is a black monitor arm, so it reads
mostly as a silhouette at this size; the close-up below carries the detail.</figcaption></figure>

<figure>{fig(f"{JOB}/object_renders/994_sheet.png")}
<figcaption>Six views of object 994, a metal stool with a wooden seat.</figcaption></figure>

<figure>{fig(f"{JOB}/object_renders/closeup/994_sheet.png")}
<figcaption>Close-up of object 994 at the touch location (orange).</figcaption></figure>

<figure>{fig(f"{JOB}/object_renders/closeup/993_sheet.png")}
<figcaption>Close-ups of the arm where the touches sit: blue is the shifted reference touch, orange
is the fixed query. They are about a millimetre apart.</figcaption></figure>

<h2>The folder</h2>
<pre>log/paper_job05_recon_figure/figure_assets/recon_pdf/&lt;object&gt;_&lt;touch&gt;/
  figure/            recon_&lt;object&gt;_&lt;touch&gt;.pdf and .png
  cells/             one PNG per cell: prediction/ relief_3d/ simulated_rgb/
  object_renders/    views/ closeup/ (and marked/ for 993)
  sources/           the videos behind it, the query renders, the checkpoint
  MANIFEST.txt       what each file is, and the numbers for this touch

and the earlier four-row version:
log/paper_job05_recon_figure/figure_assets/{TAG}/
  figure/            the four rows already stitched
  cells/             one PNG per cell, grouped by row:
                     reference_touch/ our_prediction/ heightmap_3d/ simulated_rgb/
  reference_shift/   the chosen reference's video and renders, the contact points,
                     the sweep sheet and scores, the alignment fit, the checkpoint
  object_renders/    views/ marked/ closeup/
  alternates/        the larger-shift version of the figure
  MANIFEST.txt       what each file is, and the numbers for this touch</pre>
<p>Everything in there is a symlink, so nothing is duplicated on disk.</p>

<h2>Regenerating</h2>
<pre><code>conda activate pm_touch
cd paper_experiments

# re-simulate reference touches around the original spot
python 04_paper_figures/shift_query_touch.py --obj 993 --which ref --touch 2 \\
    --dists 0.006 0.012 0.022 0.035 --n_dirs 4 --tag 993_ref2_shift

# try each of them against the fixed query
python 05_recon_figure/sweep_reference_shift.py --obj 993 --query 2 \\
    --shift_dir .../shifted/993_ref2_shift --sheet

# build the figure from the chosen one
python 03_recon_visuotactile/make_recon_figure.py --obj 993 --pair 2 --layout nested \\
    --transfer_dir .../paper_job05_recon_figure/refsweep/cand11 \\
    --out_dir .../paper_job05_recon_figure --tag {TAG} --n_cols 6

# the object, and the folder
python 04_paper_figures/render_object.py --obj 993 --views 6 --ambient 0.85 --light 6 \\
    --out .../paper_job05_recon_figure/object_renders
python 05_recon_figure/make_recon_pdf.py --obj 993 --pair 2
python 05_recon_figure/make_recon_pdf.py --obj 994 --pair 3
python 04_paper_figures/render_object.py --obj 994 --views 6 --ambient 0.6 --light 4 \
    --out .../paper_job05_recon_figure/object_renders
python 05_recon_figure/collect_pdf_assets.py
python 05_recon_figure/collect_recon_assets.py --cand 11
python 05_recon_figure/build_report.py
</code></pre>

<footer style="margin-top:2.5rem;color:#777;font-size:.85rem">
Assets: <code>log/paper_job05_recon_figure/</code> &middot; refinement network:
<code>log/rebot_checkpoints_S_geomcat_film/best.pth</code>
</footer>
</body></html>
"""

open(f"{JOB}/report.html", "w").write(HTML)
print("wrote", f"{JOB}/report.html")
