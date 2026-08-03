"""Summary of the candidate examples picked for fig_gt_retrieval and fig_recon."""
import base64
import json
import os

import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OUTDIR = f"{ROOT}/log/paper_fig_candidates"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = f"{ROOT}/log/paper_figure_candidates.html"

man = json.load(open(f"{OUTDIR}/manifest.json"))
allc = json.load(open(os.path.join(HERE, "candidates.json")))

# Hand-written notes, after looking at every rendered candidate.
GT_NOTES = {
    1: ("recommended",
        "Best of the five. Row 2 (object 964) is the strongest single panel we have: the coarse "
        "transfer invents a smooth diagonal ridge, and the refinement recovers the corner plus the "
        "row of bumps almost exactly as in the ground truth &mdash; the reader can see the network "
        "doing real work, not just sharpening. The three rows are also genuinely different "
        "geometry (a lamp rim, a chair joint with bumps, a curved handle)."),
    2: ("strong alternative",
        "Row 2 (object 969) is a clean save on a box corner: the coarse transfer smears the corner "
        "away and the refinement puts it back. All three rows are visually distinct. Slightly less "
        "dramatic than candidate 1 but arguably tidier and easier to read at column width."),
    3: ("usable",
        "Row 2 (object 951) is a broad smooth bulge with little internal structure, and rows 1 and 3 "
        "are both slab edges, so the figure repeats itself. Fine if you want a calmer figure."),
    4: ("strong alternative, most detail",
        "Row 2 (object 993) keeps a run of fine slats through the prediction, which is the best "
        "evidence in the set that the method preserves high-frequency detail. Row 1 (object 981) is "
        "a nice failure-mode-to-fix: the coarse transfer produces a hard corner where the ground "
        "truth is a smooth curve. Row 3 is a rounded stool corner."),
    5: ("weakest",
        "Rows 1 and 3 are near-duplicates &mdash; both flat slab edges seen at a similar angle &mdash; "
        "so the figure wastes two of its three rows. Row 2 (object 952) is good. Use only if you "
        "need these specific objects."),
}

RECON_NOTES = {
    "994_3": ("recommended",
              "Cleanest all-rounder. The contact clearly grows and shrinks across the six columns, "
              "the 3D relief is a well-defined dome with a visible crease, and the simulated RGB "
              "reads unambiguously. Nothing in it needs explaining away."),
    "993_2": ("best detail",
              "The richest geometry of the five: a slatted structure whose individual ridges survive "
              "into the heightmap and the RGB. Most impressive 3D row, but the contact changes less "
              "across the press, so the columns look more alike &mdash; slightly undercuts the "
              "\"it is a video\" point the figure is also making."),
    "978_4": ("strong alternative",
              "A long ridge with a strong, clean relief. Also the clearest reference-versus-prediction "
              "contrast in the set: the reference contact is horizontal and the predicted one runs "
              "diagonally, so the figure doubles as evidence that the method follows query geometry "
              "rather than copying the reference."),
    "975_2": ("usable, with a caveat",
              "Good dome relief, but there is a thin vertical streak artifact in the predicted normals "
              "in the middle four frames, and it propagates into the simulated RGB. Visible on close "
              "inspection at full size."),
    "981_5": ("weakest",
              "The contact is a broad, shallow rim, so after height normalisation the 3D relief comes "
              "out washed out and nearly flat. The 3D row is the whole point of this figure, so this "
              "one undersells it."),
}

FIELDS = [("refined_psnr", "Refined PSNR", "{:.1f}"), ("gain", "Gain over coarse", "{:+.1f}"),
          ("contact", "Contact strength", "{:.2f}"), ("structure", "Structure", "{:.3f}"),
          ("pose_diff", "Pose difference", "{:.3f}"), ("temporal", "Temporal change", "{:.2f}")]


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode("ascii")


def img(p):
    if not os.path.exists(p):
        return "<p><i>missing</i></p>"
    return f'<img src="data:image/png;base64,{b64(p)}">'


def pct(key, val):
    """Where this value sits in the distribution over all 400 touches."""
    v = np.array([c[key] for c in allc])
    return float((v < val).mean() * 100)


def metric_table(recs, keys):
    head = "".join(f"<th>{lab}</th>" for _, lab, _ in FIELDS if _ in keys or True)
    head = "".join(f"<th>{lab}</th>" for k, lab, _ in FIELDS if k in keys)
    body = ""
    for r in recs:
        cells = "".join(f"<td>{fmt.format(r[k])}<span class='pc'>p{pct(k, r[k]):.0f}</span></td>"
                        for k, lab, fmt in FIELDS if k in keys)
        body += f"<tr><th>object {r['obj']}, touch {r['pair']}</th>{cells}</tr>"
    return f"<table class='m'><thead><tr><th>Touch</th>{head}</tr></thead><tbody>{body}</tbody></table>"


GT_KEYS = ["refined_psnr", "gain", "contact", "structure", "pose_diff"]
RC_KEYS = ["refined_psnr", "contact", "structure", "temporal"]

gt_secs = ""
for entry in man["gt_retrieval"]:
    i = entry["candidate"]
    tag, note = GT_NOTES.get(i, ("", ""))
    cls = "rec" if tag.startswith("recommended") else ""
    gt_secs += f"""
<section class="cand {cls}">
  <h3>Candidate {i} <span class="tag">{tag}</span></h3>
  <p class="note">{note}</p>
  {img(entry['out'])}
  {metric_table(entry['touches'], GT_KEYS)}
</section>"""

rc_secs = ""
for entry in man["recon"]:
    key = f"{entry['obj']}_{entry['pair']}"
    tag, note = RECON_NOTES.get(key, ("", ""))
    cls = "rec" if tag.startswith("recommended") else ""
    rc_secs += f"""
<section class="cand {cls}">
  <h3>Candidate {entry['candidate']} &mdash; object {entry['obj']}, touch {entry['pair']}
      <span class="tag">{tag}</span></h3>
  <p class="note">{note}</p>
  {img(entry['out'])}
  {metric_table([entry['rec']], RC_KEYS)}
</section>"""

n_gt_pass = sum(1 for c in allc if c["refined_psnr"] >= 26 and c["contact"] >= 0.45
                and c["gain"] >= 3 and 0.08 <= c["cover_ref"] <= 0.85
                and 0.08 <= c["cover_query"] <= 0.85 and c["pose_diff"] >= 0.04)

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Figure candidates &mdash; qualitative examples</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1180px;
      margin:2rem auto;padding:0 1.2rem;line-height:1.65;color:#1b1b1f}}
 h1{{border-bottom:2px solid #333;padding-bottom:.4rem;margin-bottom:.2rem}}
 h2{{margin-top:2.8rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 h3{{margin:0 0 .3rem}}
 .lead{{color:#555;margin-top:0}}
 img{{width:100%;display:block;border:1px solid #ddd;border-radius:4px;margin:.6rem 0}}
 .cand{{margin:2rem 0;padding:1rem 1.2rem;border:1px solid #e3e3e8;border-radius:10px;background:#fcfcfd}}
 .cand.rec{{border-color:#1f7a3d;background:#f4fbf6}}
 .tag{{font-size:.75rem;background:#555;color:#fff;border-radius:10px;padding:.1rem .6rem;
       margin-left:.5rem;vertical-align:middle;font-weight:400}}
 .cand.rec .tag{{background:#1f7a3d}}
 .note{{color:#444;font-size:.93rem;margin:.2rem 0 .6rem}}
 table.m{{border-collapse:collapse;width:100%;font-size:.85rem;margin-top:.5rem}}
 table.m th,table.m td{{border:1px solid #e0e0e4;padding:.3rem .55rem;text-align:right}}
 table.m thead th{{background:#f0f2f5;text-align:center}}
 table.m tbody th{{text-align:left;font-weight:600;background:#fafafb}}
 .pc{{color:#999;font-size:.75em;margin-left:.35rem}}
 code{{background:#f4f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 .note-box{{background:#eef6ff;border-left:4px solid #3d7fd1;padding:.75rem 1rem;margin:1.1rem 0}}
 .warn{{background:#fffbe6;border-left:4px solid #e0b400;padding:.75rem 1rem;margin:1.1rem 0}}
 ul li{{margin:.25rem 0}}
</style></head><body>

<h1>Figure candidates &mdash; qualitative examples</h1>
<p class="lead">Candidate examples for the two figures produced by the Local jobs:
<code>fig_gt_retrieval</code> and <code>fig_recon</code>. Five candidates each, rendered at full
resolution in <code>log/paper_fig_candidates/</code>.</p>

<div class="warn"><b>Scope.</b> Only the two figures whose experiments run on this machine are
covered. <code>fig_full_pipeline</code> and <code>fig_ablation</code> come from Dirac jobs (the
baseline comparison and the ablation study), and <code>fig_teaser</code> and
<code>fig_method</code> are illustrations rather than experiment outputs. TaRF is not visualised
anywhere here.</p></div>

<h2>How the candidates were chosen</h2>
<p>Picking by PSNR alone gives bad figures: the highest-PSNR touches are the ones where almost
nothing happens, a faint contact on a flat patch, which look empty on the page. So each of the 400
evaluation touches was scored on the properties that actually decide whether a panel reads well:</p>
<ul>
<li><b>Contact strength</b> &mdash; how far the true touch departs from flat gel. A contact you can see.</li>
<li><b>Structure</b> &mdash; edge density inside the contact. Distinguishes a shaped contact (an edge,
    a rim, a row of bumps) from one smooth featureless bulge.</li>
<li><b>Gain</b> &mdash; refined PSNR minus coarse PSNR. If this is near zero, columns 4 and 5 of the
    figure look identical and the figure argues that the refinement network is pointless.</li>
<li><b>Pose difference</b> &mdash; how much the query touch differs from the reference touch. If the two
    are nearly the same, the analogy looks trivial.</li>
<li><b>Render coverage</b> &mdash; how much of the 4x normal render is actual surface rather than empty
    background, so columns 2 and 3 show a recognisable object and not a black frame.</li>
<li><b>Temporal change</b> &mdash; how much the contact grows and shrinks over the press. Only matters for
    <code>fig_recon</code>, whose columns are frames: without this every column looks the same.</li>
</ul>
<p>Touches first have to clear hard cut-offs on all of these, then the survivors are ranked by a
weighted score and spread over distinct objects. {n_gt_pass} of the 400 touches clear the
<code>fig_gt_retrieval</code> cut-offs and 94 clear the <code>fig_recon</code> ones. The numbers
under each candidate below show the raw value and, in grey, its percentile among all 400 touches.</p>

<div class="note-box"><b>Ordering.</b> Candidates are listed best-first by the automatic score, but the
written verdict on each is my own after looking at every rendered candidate at full size &mdash; and it
does not always agree with the score. Where it disagrees I say so.</div>

<h2>fig_gt_retrieval &mdash; 3 &times; 6 qualitative results</h2>
<p>Columns: reference touch &middot; reference normal render (4x, 1x sensor footprint in red) &middot;
query normal render (4x, boxed) &middot; coarse transfer &middot; refined transfer (ours) &middot;
ground-truth query touch. Each candidate is a complete figure; you can also mix rows across
candidates, since every row is an independent touch.</p>
{gt_secs}

<h2>fig_recon &mdash; 3D reconstruction and visuo-tactile simulation</h2>
<p>Rows: reference tactile normal video &middot; our predicted tactile normal video &middot; 3D relief
integrated from the prediction &middot; simulated RGB visuo-tactile frames. Columns are frames sampled
from the in-contact part of the press. The 3D row uses the shading style from
<code>train_refine_scripts/time_cond_sweep/height3d_geomcat_film.py</code> &mdash; matte Lambertian
shading computed from the surface's own slopes, light at (-0.5, -0.6, 0.7), view elevation 55 /
azimuth -62.</p>
{rc_secs}

<h2>Files</h2>
<ul>
<li><code>log/paper_fig_candidates/gt_retrieval/candidate_&lt;1..5&gt;.png</code></li>
<li><code>log/paper_fig_candidates/recon/figure_&lt;object&gt;_&lt;touch&gt;.png</code>, with every
    individual cell under <code>recon/assets/&lt;object&gt;_&lt;touch&gt;/</code></li>
<li><code>log/paper_fig_candidates/manifest.json</code> &mdash; which touches are in which candidate</li>
<li><code>paper_experiments/04_figure_candidates/</code> &mdash; the scoring and build scripts,
    <code>candidates.json</code> (all 400 scored touches), and the Markdown fact sheet</li>
</ul>
<p>Per-cell PNGs for the <code>fig_gt_retrieval</code> rows are already cached for all 400 touches in
<code>log/paper_job02_gt_retrieval_figure_normalmatch/assets/</code>, so any row can be swapped without
re-running the network. To render a different combination:</p>
<pre><code>python paper_experiments/02_gt_retrieval_figure/make_figure.py \\
    --base log/paper_job02_gt_retrieval_figure_normalmatch \\
    --touches 964_3 969_3 993_2 --out log/paper_fig_candidates/gt_retrieval/mine.png

python paper_experiments/03_recon_visuotactile/make_recon_figure.py \\
    --obj 994 --pair 3 --n_cols 6 --out_dir log/paper_fig_candidates/recon</code></pre>
</body></html>
"""

open(OUT_HTML, "w").write(HTML)

MD_ROWS = "\n".join(
    f"| {e['candidate']} | " + ", ".join(f"{t['obj']}_{t['pair']}" for t in e["touches"]) +
    f" | {GT_NOTES.get(e['candidate'], ('',''))[0]} |"
    for e in man["gt_retrieval"])
MD_RC = "\n".join(
    "| {c} | {k} | {v} |".format(
        c=e["candidate"], k=f"{e['obj']}_{e['pair']}",
        v=RECON_NOTES.get(f"{e['obj']}_{e['pair']}", ("", ""))[0])
    for e in man["recon"])

MD = f"""# Figure candidates (Local jobs only)

Full report with images: `log/paper_figure_candidates.html`
Assets: `log/paper_fig_candidates/`

Scope: `fig_gt_retrieval` and `fig_recon` only. `fig_full_pipeline` and `fig_ablation` are Dirac
jobs; `fig_teaser` and `fig_method` are illustrations, not experiment outputs. No TaRF.

## fig_gt_retrieval candidates (each is a complete 3 x 6 figure)

| Candidate | Touches (object_touch) | Verdict |
|---|---|---|
{MD_ROWS}

Recommended: **candidate 1** (980_6, 964_3, 974_5). Object 964 is the strongest single panel —
the coarse transfer invents a smooth diagonal ridge and the refinement recovers the corner plus
the row of bumps, so the reader sees the network doing real work.

## fig_recon candidates

| Candidate | Touch | Verdict |
|---|---|---|
{MD_RC}

Recommended: **994_3**. If the paper needs to emphasise fine-detail preservation instead, use
**993_2**; if it needs to emphasise that the prediction follows query geometry rather than copying
the reference, use **978_4**.

## Selection method

Touches must clear hard cut-offs on contact strength, structure (edge density in the contact),
refinement gain, pose difference, and normal-render coverage; `fig_recon` additionally requires
temporal change across the press. Survivors are ranked by a weighted score and spread over distinct
objects. Cut-offs and weights are in `paper_experiments/04_figure_candidates/build_candidates.py`
(`GATES`, `WEIGHTS`); per-touch scores for all 400 touches are in `candidates.json`.

Why not rank by PSNR: the highest-PSNR touches are the ones where nothing visible happens, so they
make empty-looking panels.
"""

open(os.path.join(HERE, "results.md"), "w").write(MD)
open(f"{OUTDIR}/report.html", "w").write(HTML)
print("wrote", OUT_HTML)
