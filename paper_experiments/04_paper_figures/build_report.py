"""HTML report for the teaser and method-overview figure drafts."""
import base64
import json
import os
import pickle

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
OUT = f"{ROOT}/log/paper_job04_paper_figures"
TEASER_PICKS = ["fp27_11", "fp25_13", "fp29_3", "fp26_12", "fp23_14"]
TEASER_ALTS = ["fp8_15", "fp18_2", "fp10_5", "fp28_0"]
TEASER_TAG = TEASER_PICKS[0]
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
m_obj, m_pair = (int(x) for x in METHOD_TAG.split("_"))
m_rec = score[(m_obj, m_pair)]

fp = pickle.load(open(f"{ROOT}/log/paper_job04_paper_figures/fullpipe/candidates.pkl", "rb"))
n_scored, n_objs = len(fp), len({r["obj"] for r in fp})
meta = {t: pickle.load(open(f"{OUT}/assets/{t}_meta.pkl", "rb"))
        for t in TEASER_PICKS + TEASER_ALTS}


def teaser_block(tags, kind):
    out = []
    for t in tags:
        m = meta[t]
        out.append(
            f"<figure>{fig(f'teaser_v1_{t}')}<figcaption><b>{t}</b> &mdash; object "
            f"{m['obj']}, query touch {m['pair']}. The method retrieved touch "
            f"{m['ref_idx']} on its own as the reference. Coarse transfer "
            f"{m['psnr_coarse']:.1f} dB, refined {m['psnr_refined']:.1f} dB "
            f"({kind}).</figcaption></figure>")
    return "".join(out)
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
<tr><th>Teaser figures</th><td><code>log/paper_job04_paper_figures/teaser_v1_fp&lt;object&gt;_&lt;touch&gt;.png</code>
 &mdash; nine of them, the five picks and four runners-up (each also as <code>.pdf</code>)</td></tr>
<tr><th>Method figures</th><td><code>method_v1.png</code>, <code>method_v2.png</code>,
 <code>method_v3.png</code> in the same folder (unchanged from the first round)</td></tr>
<tr><th>Full-pipeline runs</th><td><code>log/paper_job04_paper_figures/fullpipe/</code> &mdash;
 retrieval results, coarse transfers and per-touch scores for the objects swept;
 <code>candidates.pkl</code> holds one record per query touch</td></tr>
<tr><th>Cached image pieces</th><td><code>log/paper_job04_paper_figures/assets/</code> &mdash;
 every video frame, geometry render, correspondence set and similarity score used above</td></tr>
<tr><th>LaTeX</th><td><code>paper_source/figures/fig_teaser.tex</code> &mdash; switched to
 <code>figure*</code> (two columns) with an updated caption; the placeholder rule is still in
 place, no graphic dropped in yet</td></tr>
<tr><th>Scripts</th><td><code>paper_experiments/04_paper_figures/</code>:
 <code>run_full_pipeline_local.py</code> (runs the pipeline here),
 <code>sweep_teaser.py</code> (ranks and exports examples),
 <code>prep_assets.py</code> (pieces for the method figure), <code>make_teaser.py</code>,
 <code>make_method.py</code>, <code>figlib.py</code> (shared layout helpers),
 <code>build_report.py</code> (this page)</td></tr>
</table>

<h2>Teaser figure</h2>

<p>The idea being sold across the page: <i>this is the geometry where we touched, and this is
what that touch felt like; here is the geometry somewhere else, so here is what a touch there
would feel like.</i> That is the same four-part &ldquo;A is to A&prime; as B is to B&prime;&rdquo;
arrangement the Image Analogies teaser uses, which is why the colon and double-colon marks are
kept.</p>

<p>Three things changed from the first drafts:</p>
<ul>
 <li><b>Two columns.</b> The figure is now 7.0 &times; 2.22 inches, so it spans the full page
     width. <code>paper_source/figures/fig_teaser.tex</code> was switched from
     <code>figure</code> to <code>figure*</code> to match. The graphic itself was not put into
     the LaTeX yet &mdash; the placeholder rule is still there.</li>
 <li><b>Six frames per video instead of three.</b> The extra width pays for a press that
     actually reads as a press: contact starting, deepening, and releasing.</li>
 <li><b>White background on the geometry renders.</b> Empty space around the object used to be
     black, which dominated the figure. Taxim writes empty space as exactly black while surface
     pixels never come out dark, so a single brightness threshold separates them cleanly &mdash;
     no hand-drawn masks involved.</li>
</ul>

<h3>Examples now come from the full-pipeline benchmark</h3>

<p>The earlier drafts used the ground-truth-retrieval benchmark, where the matching reference is
handed to the method. A teaser should show the whole system, so the examples were re-drawn from
the <b>full-pipeline benchmark</b>, where the method has to find its own reference among the
object's other touches. Those runs live on the other machine, so they were reproduced here:
<code>run_full_pipeline_local.py</code> rebuilds each object's reference / query split from the
very same manifest the paper reports
(<code>paper_experiments/job2_full_pipeline/splits.json</code>, seed 0), runs DINOv3 retrieval and
SuperPoint&nbsp;+&nbsp;SuperGlue coarse alignment with the same flags as
<code>job2_full_pipeline/run_transfer.sh</code>, then refines with the paper's network.</p>

<p><b>{n_objs} objects, {n_scored} held-out query touches</b> were run and scored. Candidates were
then ranked by four things measured off the images themselves &mdash; how strong the contact is,
how much the press changes over time, how different the reference and query geometry are, and the
prediction's accuracy &mdash; and the top two dozen were laid out as a contact sheet
(<code>log/paper_job04_paper_figures/teaser_candidates.png</code>) to choose from by eye.</p>

<h3>The five picks</h3>
{teaser_block(TEASER_PICKS, "full pipeline, retrieval included")}

<h3>Also rendered, not picked</h3>
<p>Kept because they may suit a different caption: <code>fp8_15</code> and <code>fp18_2</code>
are clean but the two rows look alike, so the analogy is less striking; <code>fp10_5</code> has
visible banding in the prediction; <code>fp28_0</code>'s reference touch barely makes contact, so
the top row looks empty.</p>
{teaser_block(TEASER_ALTS, "full pipeline, runner-up")}

<h3>The two one-column layouts, for comparison</h3>
<p>These were the other two drafts. They are still one column and still show three frames; they
are kept only so the two-column choice can be compared against them. Both are drawn on the same
full-pipeline example as the first pick.</p>
<figure>{fig("teaser_v2_" + TEASER_TAG, "48%")}
<figcaption>Adds the held-out ground truth as a third row so the prediction can be checked against
it on the spot.</figcaption></figure>
<figure>{fig("teaser_v3_" + TEASER_TAG, "42%")}
<figcaption>Three columns &mdash; reference, prediction, truth &mdash; with video time running down
the page.</figcaption></figure>

<div class="warn">
The files named <code>teaser_v1_951_7</code>, <code>teaser_v2_951_7</code> and
<code>teaser_v3_951_7</code> are the earlier drafts on the ground-truth-retrieval benchmark. They
are superseded by the full-pipeline versions above and are kept only for comparison.
</div>

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
 <td>The teaser examples come from the full-pipeline benchmark and are the five listed above; the
 method figure still uses object {m_obj} touch {m_pair} of the ground-truth-retrieval benchmark (a
 lattice-like surface, so the correspondence lines land on visible structure). None of them is the
 best-scoring touch in its benchmark: the five teaser picks score
 {", ".join(f"{meta[t]['psnr_refined']:.1f}" for t in TEASER_PICKS)} dB against a benchmark
 average of 31.6 dB.</td></tr>
<tr><th>White backgrounds</th>
 <td>Applied only to the geometry renders, and only to pixels whose brightest channel is below 96.
 Taxim leaves empty space at exactly zero and surface normals never encode to a dark colour &mdash;
 across these renders no pixel at all falls between 64 and 120 &mdash; so nothing on the object can
 be repainted by accident. Where a render shows white <i>inside</i> the object's outline, that is a
 real gap in the geometry, not a whitened surface.</td></tr>
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

# teaser: run the full pipeline here, rank candidates, export the chosen ones
python run_full_pipeline_local.py --n_objects 30
python sweep_teaser.py --sheet                       # writes the contact sheet
python sweep_teaser.py --tags 27_11 25_13 29_3 26_12 23_14
for t in fp27_11 fp25_13 fp29_3 fp26_12 fp23_14; do
    python make_teaser.py --tag $t --versions v1
done

# method figure (unchanged)
python prep_assets.py --do touch match benchret --obj 1000 --pair 7
python make_method.py --tag 1000_7 --ret_tag bench1000_7

python build_report.py
</code></pre>
<p><code>make_teaser.py</code> takes <code>--frames</code> to choose which frames of the press are
shown (six by default) and <code>--versions</code> to redraw a single layout.
<code>run_full_pipeline_local.py --n_objects</code> controls how much of the benchmark is swept;
objects already transferred are skipped, so it can be extended without redoing work.</p>

<footer style="margin-top:2.5rem;color:#777;font-size:.85rem">
Figures and assets: <code>log/paper_job04_paper_figures/</code> &middot; refinement network:
<code>log/rebot_checkpoints_S_geomcat_film/best.pth</code>
</footer>
</body></html>
"""

os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/report.html", "w").write(HTML)
print("wrote", f"{OUT}/report.html")
