"""HTML report for the teaser and method-overview figure drafts."""
import base64
import json
import os
import pickle

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OUT = f"{ROOT}/log/paper_job04_paper_figures"
TEASER_TAG = "951_7"
METHOD_TAG = "1000_7"


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode("ascii")


def fig(name, width="100%"):
    p = f"{OUT}/{name}.png"
    if not os.path.exists(p):
        return f"<p><i>{name}.png not generated</i></p>"
    return (f'<img style="width:{width};display:block;border:1px solid #ddd" '
            f'src="data:image/png;base64,{b64(p)}">')


rc = json.load(open(f"{OUT}/retrieval_check.json"))
md = pickle.load(open(f"{OUT}/assets/{METHOD_TAG}_matches.pkl", "rb"))
ret = pickle.load(open(f"{OUT}/assets/bench{METHOD_TAG}_retrieval.pkl", "rb"))
recs = pickle.load(open(f"{ROOT}/log/paper_job02_gt_retrieval_figure_normalmatch/"
                        "per_touch_metrics.pkl", "rb"))
score = {(r["obj"], r["pair"]): r for r in recs}
t_obj, t_pair = (int(x) for x in TEASER_TAG.split("_"))
m_obj, m_pair = (int(x) for x in METHOD_TAG.split("_"))
t_rec, m_rec = score[(t_obj, t_pair)], score[(m_obj, m_pair)]
top1_sim = float(max(ret["loo_scores"]))

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Job 4 &mdash; teaser and method figure drafts</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      max-width:1150px;margin:2rem auto;padding:0 1.2rem;line-height:1.6;color:#1b1b1f}}
 h1{{border-bottom:2px solid #333;padding-bottom:.4rem}}
 h2{{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 h3{{margin-top:1.6rem}}
 table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.93rem}}
 th,td{{border:1px solid #ddd;padding:.42rem .7rem;text-align:left;vertical-align:top}}
 th{{background:#f7f7f9}}
 code{{background:#f4f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 figure{{margin:1.4rem 0}} figcaption{{color:#555;font-size:.9rem;margin-top:.4rem}}
 .note{{background:#eef6ff;border-left:4px solid #3d7fd1;padding:.7rem 1rem;margin:1rem 0}}
 .warn{{background:#fff6e6;border-left:4px solid #d9a13b;padding:.7rem 1rem;margin:1rem 0}}
 ul li{{margin:.25rem 0}}
</style></head><body>

<h1>Job 4 &mdash; drafts of the teaser figure and the method overview figure</h1>

<p>This job produces the two figures asked for in <code>paper_assets/figures_plan.md</code>:
a teaser in the style of the Image Analogies teaser, and a step-by-step method overview.
Three versions of each were drawn so a layout can be picked before anything is polished.
Every version exists as a <b>PNG</b> (for looking at) and a <b>PDF</b> (for dropping into
LaTeX; the text stays real text, not pixels).</p>

<div class="note">
<b>All the pictures and numbers inside the figures are real.</b> The tactile videos come
from the benchmark, the predicted frames come from actually running the trained refinement
network, the yellow lines are actual SuperPoint + SuperGlue correspondences, and the
similarity bars are actual DINOv3 cosine similarities. Nothing is a hand-drawn stand-in.
</div>

<h2>Where the files are</h2>
<table>
<tr><th>Figures</th><td><code>log/paper_job04_paper_figures/teaser_v1_{TEASER_TAG}.png</code>,
 <code>teaser_v2_…</code>, <code>teaser_v3_…</code>, <code>method_v1.png</code>,
 <code>method_v2.png</code>, <code>method_v3.png</code> (each also as <code>.pdf</code>)</td></tr>
<tr><th>Cached image pieces</th><td><code>log/paper_job04_paper_figures/assets/</code> &mdash;
 every video frame, geometry render, correspondence set and similarity score used above</td></tr>
<tr><th>Scripts</th><td><code>paper_experiments/04_paper_figures/</code>:
 <code>prep_assets.py</code> (makes the pieces), <code>make_teaser.py</code>,
 <code>make_method.py</code>, <code>figlib.py</code> (shared layout helpers),
 <code>build_report.py</code> (this page)</td></tr>
</table>

<h2>Teaser figure</h2>

<p>The idea being sold in one column: <i>this is the geometry where we touched, and this is
what that touch felt like; here is the geometry somewhere else, so here is what a touch there
would feel like.</i> That is the same four-part &ldquo;A is to A&prime; as B is to B&prime;&rdquo;
arrangement the Image Analogies teaser uses, which is why the colon and double-colon marks are
kept.</p>

<p>The example is object {t_obj}, touch {t_pair} of the ground-truth-retrieval benchmark. It was
chosen because the touch has clearly visible dynamics &mdash; the gel is flat at the start,
deeply pressed in the middle, and flat again at the end &mdash; so three frames really do look
like a video rather than three copies of one picture. Coarse transfer alone scores
{t_rec['coarse']['PSNR']:.1f} dB on this touch and the refined prediction scores
{t_rec['refined']['PSNR']:.1f} dB.</p>

<h3>Version 1 &mdash; the plain analogy (2 rows)</h3>
<figure>{fig(f"teaser_v1_{TEASER_TAG}")}
<figcaption>Two rows only: what we were given, and what we predict. Nothing about accuracy is
claimed inside the figure, which keeps it uncluttered; a reader checks the numbers later in the
paper. 3.35 &times; 1.98 inches (one column).</figcaption></figure>

<h3>Version 2 &mdash; the analogy plus the held-out truth (3 rows)</h3>
<figure>{fig(f"teaser_v2_{TEASER_TAG}")}
<figcaption>Adds the real touch that was hidden from the method, so the reader can compare the
prediction against it immediately. The geometry cell of the third row is deliberately left empty
(dashed) so it does not look as though the truth row had an extra input.
3.35 &times; 2.66 inches.</figcaption></figure>

<h3>Version 3 &mdash; three columns, time running down the page</h3>
<figure>{fig(f"teaser_v3_{TEASER_TAG}", "62%")}
<figcaption>A tall, narrow variant: reference, prediction and truth side by side, with the three
video frames stacked downwards. It reads more like a comparison table and less like an analogy,
but it fits a narrow column well and leaves room for a long caption beside it.
3.35 &times; 3.70 inches.</figcaption></figure>

<h2>Method overview figure</h2>

<p>Three steps, in the order the method runs them:</p>
<ol>
 <li><b>Retrieval.</b> The query location is only known through its geometry. Both the query and
     every reference touch are passed through DINOv3 (a general-purpose image feature extractor)
     and compared by cosine similarity &mdash; a number between &minus;1 and 1 saying how alike two
     feature vectors are. The most similar reference wins.</li>
 <li><b>Coarse alignment.</b> SuperPoint (a keypoint detector) and SuperGlue (a matcher) find
     matching points between the two geometry renders. The matches that agree on one common warp
     fix that warp, and it is applied to the reference video.</li>
 <li><b>Refinement.</b> The warped frames, with the query normal map stacked alongside them as
     extra channels, go through a small encoder&ndash;bottleneck&ndash;decoder network. The frame
     number enters separately and rescales the features at each encoder stage (FiLM).</li>
</ol>

<p>All three steps in the figure show the <b>same</b> touch: object {m_obj}, touch {m_pair}. That
matters, because step 1 picks a reference and steps 2&ndash;3 have to be showing that same
reference for the figure to be honest. On this touch DINOv3 picks reference {ret['order'][0]} at
similarity {top1_sim:.2f}, which is the reference the benchmark pairs with this query, so the rest
of the figure follows on from it. SuperPoint + SuperGlue then produce {len(md['xy_l'])} matches, of
which {int(md['inlier'].sum())} agree on a single warp. Coarse transfer scores
{m_rec['coarse']['PSNR']:.1f} dB here, refinement {m_rec['refined']['PSNR']:.1f} dB.</p>

<h3>Version 1 &mdash; one column, three shaded panels</h3>
<figure>{fig("method_v1", "70%")}
<figcaption>The most detailed version. The reference database is a filmstrip with a similarity bar
under each entry, drawn on a fixed 0.4&ndash;1.0 scale so the bars are comparable rather than
stretched to fit. 3.35 &times; 4.64 inches.</figcaption></figure>

<h3>Version 2 &mdash; one column, flat, database as a vertical list</h3>
<figure>{fig("method_v2", "70%")}
<figcaption>No panel shading, just a rule under each step title. The reference entries are stacked
vertically with long similarity bars, which makes the ranking easier to read at small size. The
network is a plain chain of blocks. 3.35 &times; 4.36 inches.</figcaption></figure>

<h3>Version 3 &mdash; two columns wide, steps left to right</h3>
<figure>{fig("method_v3")}
<figcaption>The same content spread across a full page width, for the case where the method figure
gets a two-column slot. It is much shorter vertically, at the price of smaller pictures inside each
step. 7.0 &times; 2.34 inches.</figcaption></figure>

<h2>Choices worth knowing about</h2>

<table>
<tr><th>Which matching modality</th>
 <td>Surface normals at four times the sensor footprint, matching the paper default recorded in
 <code>paper_experiments/02_gt_retrieval_figure/results.md</code>. The coarse-transfer videos and
 refined frames are therefore taken from the <code>…_normalmatch</code> run, not the curvature
 one.</td></tr>
<tr><th>&ldquo;Warp&rdquo; rather than &ldquo;affine map&rdquo;</th>
 <td>The plan text says an affine map is fitted. The transfer runs that produced these results
 actually fitted a homography (<code>--transform_type homography</code>, which is the default in
 <code>main_retrieval_transfer_feat_match.py</code>). The figures say &ldquo;warp&rdquo; and
 &ldquo;homography&rdquo; so they match what was run; if the paper is going to say affine, the
 transfers need re-running with that setting.</td></tr>
<tr><th>How much of the network is drawn</th>
 <td>Encoder, bottleneck, decoder, the concatenated normal map and the FiLM path from the frame
 number. The real network also has a second parallel branch that pools large patches of both input
 frames through a small transformer and adds the result at the bottleneck. It is left out on
 purpose &mdash; the plan asks for the rough architecture only &mdash; so the figure is a
 simplification, not a full description.</td></tr>
<tr><th>Red boxes on the geometry renders</th>
 <td>The wide render is four times the area the sensor covers. The red box marks the sensor's own
 footprint inside it, so a reader can see that the method compares much more context than the
 patch being predicted.</td></tr>
<tr><th>Which touches are shown</th>
 <td>The teaser uses object {t_obj} touch {t_pair} (strong press dynamics); the method figure uses
 object {m_obj} touch {m_pair} (a lattice-like surface, so the correspondence lines land on visible
 structure, and retrieval agrees with the benchmark pairing). Neither is the best-scoring touch in
 the benchmark.</td></tr>
</table>

<div class="warn">
<b>One number that is easy to over-claim.</b> While picking a touch for the method figure, DINOv3
retrieval was run over all {rc['queries']} queries of the {rc['objects']}-object
ground-truth-retrieval benchmark. Its top pick matches the reference the benchmark pairs the query
with in {rc['top1_equals_gt_pair']} of {rc['queries']} cases ({100 * rc['rate']:.0f}%). That is a
side observation from choosing an example, not a benchmark result: on that benchmark the pairing is
given, so retrieval is not what is being measured. It is written to
<code>log/paper_job04_paper_figures/retrieval_check.json</code> in case it is useful later.
</div>

<h2>Regenerating</h2>
<pre><code>conda activate pm_touch
cd paper_experiments/04_paper_figures

# cached pieces (only needed once per touch)
python prep_assets.py --do touch match --obj 951 --pair 7
python prep_assets.py --do touch match benchret --obj 1000 --pair 7

# figures
python make_teaser.py --tag 951_7 --frames 4 25 45
python make_method.py --tag 1000_7 --ret_tag bench1000_7
python build_report.py
</code></pre>
<p>Both figure scripts take <code>--versions</code> to redraw just one version, and
<code>make_teaser.py --tag 1000_7</code> would draw the teaser on the method figure's touch
instead, if a single example across both figures is preferred.</p>

<footer style="margin-top:2.5rem;color:#777;font-size:.85rem">
Figures and assets: <code>log/paper_job04_paper_figures/</code> &middot; refinement network:
<code>log/rebot_checkpoints_S_geomcat_film/best.pth</code>
</footer>
</body></html>
"""

os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/report.html", "w").write(HTML)
print("wrote", f"{OUT}/report.html")
