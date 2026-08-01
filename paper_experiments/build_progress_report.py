"""Overall progress report across all local paper-experiment jobs."""
import base64
import json
import os
import pickle

import numpy as np

ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
LOG = f"{ROOT}/log"
OUT = f"{LOG}/paper_experiments_progress.html"

stats = json.load(open(f"{ROOT}/paper_experiments/01_dataset_stats/stats.json"))
D = stats["datasets"]
recs = pickle.load(open(f"{LOG}/paper_job02_gt_retrieval_figure/per_touch_metrics.pkl", "rb"))
summ = pickle.load(open(f"{LOG}/paper_job02_gt_retrieval_figure/summary.pkl", "rb"))


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode("ascii")


def img(p):
    if not os.path.exists(p):
        return "<p><i>missing</i></p>"
    return f'<img style="width:100%;display:block;border:1px solid #ddd" src="data:image/png;base64,{b64(p)}">'


def st(k, m):
    return np.mean([r[k][m] for r in recs])


tot_touch = sum(D[k]["total_touches"] for k in D)
tot_frames = sum(D[k]["total_touches"] * 50 for k in D)

JOBS = [
    ("Summarize statistics for dataset generation", "done",
     "Counted every object, touch and frame across the three benchmark datasets, plus the "
     "simulation settings used to produce them.",
     "paper_experiments/01_dataset_stats/ &middot; log/paper_job01_dataset_stats/report.html"),
    ("Ground-truth retrieval two-column figure (with all image assets saved)", "done",
     "Ran the coarse transfer over all 50 held-out objects, ran the pretrained refinement "
     "network over all 400 touches, cached every figure cell as a PNG, and stitched the 3 &times; 6 figure.",
     "paper_experiments/02_gt_retrieval_figure/ &middot; log/paper_job02_gt_retrieval_figure/"),
    ("3D surface reconstruction and visuo-tactile sensor simulation figure", "done",
     "Built the four-row figure (reference video, predicted video, 3D point clouds, simulated "
     "RGB) for six candidate touches, plus a full-length video of the main example.",
     "paper_experiments/03_recon_visuotactile/ &middot; log/paper_job03_recon_visuotactile/"),
]

rows = "".join(
    f'<tr><td style="text-align:left"><b>{t}</b><div class="sub">{d}</div></td>'
    f'<td><span class="badge">{s}</span></td>'
    f'<td style="text-align:left"><code>{loc}</code></td></tr>'
    for t, s, d, loc in JOBS)

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Tactile Analogies &mdash; local experiment progress</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      max-width:1150px;margin:2rem auto;padding:0 1.2rem;line-height:1.65;color:#1b1b1f}}
 h1{{border-bottom:2px solid #333;padding-bottom:.4rem;margin-bottom:.2rem}}
 h2{{margin-top:2.6rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 h3{{margin-top:1.8rem}}
 .lead{{color:#555;margin-top:0}}
 table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.93rem}}
 th,td{{border:1px solid #ddd;padding:.45rem .7rem;text-align:right;vertical-align:top}}
 thead th{{background:#f0f2f5;text-align:left}}
 .sub{{font-weight:400;color:#666;font-size:.85rem;margin-top:.2rem}}
 .badge{{background:#1f7a3d;color:#fff;border-radius:10px;padding:.1rem .6rem;font-size:.8rem}}
 code{{background:#f4f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 .big{{display:flex;gap:1rem;flex-wrap:wrap;margin:1.2rem 0}}
 .big div{{flex:1 1 160px;background:#f7f7f9;border:1px solid #e3e3e8;border-radius:8px;
           padding:.8rem 1rem;text-align:center}}
 .big b{{display:block;font-size:1.5rem;color:#1a4f8a}}
 .note{{background:#eef6ff;border-left:4px solid #3d7fd1;padding:.75rem 1rem;margin:1.1rem 0}}
 .warn{{background:#fffbe6;border-left:4px solid #e0b400;padding:.75rem 1rem;margin:1.1rem 0}}
 figure{{margin:1.3rem 0}} figcaption{{color:#555;font-size:.9rem;margin-top:.4rem}}
 ul li{{margin:.25rem 0}}
</style></head><body>

<h1>Tactile Analogies &mdash; local experiment progress</h1>
<p class="lead">All three jobs listed under "Local Jobs" in
<code>paper_experiments/experiment_plan.md</code> are complete. Nothing was written into
<code>paper_source/</code>.</p>

<div class="big">
 <div><b>3 / 3</b>local jobs finished</div>
 <div><b>400</b>touches transferred and refined</div>
 <div><b>+{st('refined','PSNR') - st('coarse','PSNR'):.1f} dB</b>gain from refinement</div>
 <div><b>{2418 + 288:,}</b>figure assets saved</div>
</div>

<h2>Job status</h2>
<table><thead><tr><th>Job</th><th style="text-align:left">Status</th><th style="text-align:left">Where</th></tr></thead>
<tbody>{rows}</tbody></table>

<div class="note">Each job has its own detailed report and a Markdown fact sheet with
paper-ready numbers:
<ul>
<li><code>paper_experiments/01_dataset_stats/report.html</code> &middot; <code>results.md</code></li>
<li><code>paper_experiments/02_gt_retrieval_figure/report.html</code> &middot; <code>results.md</code></li>
<li><code>paper_experiments/03_recon_visuotactile/report.html</code> &middot; <code>results.md</code></li>
</ul></div>

<h2>1. Dataset generation statistics</h2>
<p>The benchmark, as it exists on disk right now:</p>
<table><thead><tr><th style="text-align:left">Dataset</th><th>Objects</th><th>Touches / object</th>
<th>Total touches</th><th>Video frames</th></tr></thead><tbody>
<tr><td style="text-align:left">Reference touches (ground-truth-retrieval)</td><td>{D['ref']['n_objects']}</td>
    <td>8</td><td>{D['ref']['total_touches']:,}</td><td>{D['ref']['total_touches']*50:,}</td></tr>
<tr><td style="text-align:left">Query touches (ground-truth-retrieval)</td><td>{D['query']['n_objects']}</td>
    <td>8</td><td>{D['query']['total_touches']:,}</td><td>{D['query']['total_touches']*50:,}</td></tr>
<tr><td style="text-align:left">Full-pipeline benchmark</td><td>{D['raw_eval']['n_objects']}</td>
    <td>{D['raw_eval']['touches_per_object']['mean']:.1f} (6&ndash;31)</td>
    <td>{D['raw_eval']['total_touches']:,}</td><td>{D['raw_eval']['total_touches']*50:,}</td></tr>
<tr><td style="text-align:left"><b>Total</b></td><td><b>&mdash;</b></td><td><b>&mdash;</b></td>
    <td><b>{tot_touch:,}</b></td><td><b>{tot_frames:,}</b></td></tr>
</tbody></table>
<p>Every touch is a 50-frame, 240 &times; 320 tactile-normal video at 5 fps, simulated by Taxim
with a scaled GelSight Mini calibration, pressing from 0 to 10 depth units and withdrawing.
Each touch also carries RGB, normal, height, curvature and shape-index renderings of the
surface at 1x, 2x and 4x the sensor footprint.</p>

<h2>2. Ground-truth retrieval figure</h2>
<figure>{img(f"{LOG}/paper_job02_gt_retrieval_figure/figure_gt_retrieval_3x6.png")}
<figcaption>Columns: reference touch &middot; reference surface normal (4x field of view) &middot;
query surface normal (4x) &middot; coarse transfer &middot; refined transfer (ours) &middot;
ground-truth query touch. Rows are drawn from the 95th, 50th and 20th percentile of the
refinement gain, so this is not a best-case-only selection.</figcaption></figure>

<h3>Our-method numbers on the 50 held-out objects</h3>
<p>A by-product of having to run the coarse transfer locally: metrics over all
{summ['n_touches']} evaluation touches, every frame counted.</p>
<table><thead><tr><th style="text-align:left">Stage</th><th>PSNR (dB)</th><th>SSIM</th>
<th>LPIPS</th><th>MSE</th></tr></thead><tbody>
<tr><td style="text-align:left">Coarse transfer (feature matching only)</td>
    <td>{st('coarse','PSNR'):.2f}</td><td>{st('coarse','SSIM'):.4f}</td>
    <td>{st('coarse','LPIPS'):.4f}</td><td>{st('coarse','MSE'):.4f}</td></tr>
<tr><td style="text-align:left">Refined transfer (ours)</td>
    <td><b>{st('refined','PSNR'):.2f}</b></td><td><b>{st('refined','SSIM'):.4f}</b></td>
    <td><b>{st('refined','LPIPS'):.4f}</b></td><td><b>{st('refined','MSE'):.4f}</b></td></tr>
</tbody></table>
<div class="warn">These are <b>our method only</b>. The baseline comparison
(Tactile Normal Quilting, implicit neural representations, TaRF) is a Dirac job and was not
run here. Treat this as the "ours" row of that table, already computed.</div>

<h2>3. 3D reconstruction and visuo-tactile simulation</h2>
<figure>{img(f"{LOG}/paper_job03_recon_visuotactile/figure_951_5.png")}
<figcaption>Object 951, touch 5. Columns are frames of the press. Row 1: reference tactile
normal video. Row 2: our predicted tactile normal video. Row 3: 3D point cloud from
integrating row 2 into a heightmap. Row 4: RGB visuo-tactile frames from running row 3
through Taxim's optical model. The reference contact is horizontal while the query contact
is tilted, and the prediction follows the query geometry.</figcaption></figure>
<p>Five further candidate examples (objects 977, 981, 969, 967, 965) are in
<code>log/paper_job03_recon_visuotactile/</code>, along with a full-length 50-frame video of
the main example.</p>

<h2>Decisions made along the way</h2>
<ul>
<li><b>The coarse transfer had to be recomputed locally.</b>
    <code>log/transfer_feat_match_pseudo_mini_tactile_normal_superpoint_superglue</code> did not
    exist on this machine, and both figure jobs depend on it. It was regenerated for the 50
    held-out objects (951&ndash;1000) with exactly the settings the plan specifies: SuperPoint +
    SuperGlue on curvature renders at 4x the sensor footprint, median translation offset.
    That took about 11 minutes.</li>
<li><b>A separate inference script was written instead of using <code>rebot_net/eval.py</code>.</b>
    That script hard-codes the test split as <code>all_ids[950:]</code>, which selects nothing
    when the transfer directory holds only the 50 eval objects. The new script
    (<code>paper_experiments/02_gt_retrieval_figure/run_refine_eval.py</code>) takes an explicit
    object range and otherwise reproduces the same data loading, model construction and metrics.
    No repository code was modified.</li>
<li><b>Checkpoint used:</b> <code>log/rebot_checkpoints_S_geomcat_film/best.pth</code>
    (epoch {summ['epoch']}, validation PSNR 32.12) &mdash; the plan's "RebotNet + temporal FiLM +
    normal map concatenation" model.</li>
<li><b>PSNR on exactly-matching frames.</b> The no-contact frames at the start and end of a press
    can be reproduced exactly by the coarse transfer, giving MSE 0 and infinite PSNR. Those
    frames are capped at 100 dB, which is the convention <code>rebot_net/eval.py</code> already
    uses.</li>
<li><b>Normal-render columns are shown at 4x, not 1x.</b> At 1x the render is mostly empty
    background and tells the reader nothing; at 4x the object is recognisable and it is also the
    field of view the matching actually runs at. All three scales are saved as assets.</li>
</ul>

<h2>Not run (Dirac jobs)</h2>
<p>Per the plan these belong on the other machine and were left alone: the baseline comparison
on the ground-truth-retrieval benchmark, the full-pipeline benchmark comparison, the ablation
study, and the runtime analysis. The runtime analysis in particular is specified for an
RTX 3090; this machine has an RTX 2080 Ti and an RTX 2080, so measuring it here would not
produce the number the paper wants.</p>

<h2>Reproducing any of this</h2>
<pre><code>conda activate pm_touch
# job 1
python paper_experiments/01_dataset_stats/collect_stats.py
python paper_experiments/01_dataset_stats/build_report.py
# job 2 (coarse transfer first, then refinement, then figure)
bash   paper_experiments/02_gt_retrieval_figure/run_transfer_eval50.sh
python paper_experiments/02_gt_retrieval_figure/run_refine_eval.py
python paper_experiments/02_gt_retrieval_figure/make_figure.py
python paper_experiments/02_gt_retrieval_figure/build_report.py
# job 3
python paper_experiments/03_recon_visuotactile/make_recon_figure.py --obj 951 --pair 5 --video
python paper_experiments/03_recon_visuotactile/build_report.py
</code></pre>
</body></html>
"""

open(OUT, "w").write(HTML)
print("wrote", OUT)
