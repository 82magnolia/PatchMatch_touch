"""Emits log/render_mask_study.html from the benchmark JSON + filmstrip assets."""

import json
import os
import sys
from os import path as osp

import numpy as np

S = os.environ.get("SCRATCH", "/tmp")
res = json.load(open(osp.join(S, "final.json")))
rows_path = osp.join("log", "render_mask_assets", "rows.npy")
strips = np.load(rows_path, allow_pickle=True) if osp.exists(rows_path) else []

MAX = 0.46


def bar(v, kind):
    pct = 100 * v / MAX
    cls = "ref" if kind == "ref" else "method"
    return (f'<div class="barwrap"><div class="bar {cls}" style="width:{pct:.1f}%"></div>'
            f'<span class="barval">{v:.3f}</span></div>')


body = []
body.append("""
<h1>Render-mask generation: what actually limits it</h1>
<p class="sub">Evaluation of the <code>"hard"</code> render-mask path in
<code>real_data_transfer/</code> over <b>1002 touches</b> from
<code>log/real_data_real_retrieval</code>, scored against an image-derived
pseudo-ground-truth contact mask
(<code>compute_contact_mask</code> on each touch's shadow video).
All method numbers below are on a held-out half of the touches.</p>

<div class="callout warn">
<b>Headline:</b> the current production render mask is <i>not better than
predicting nothing</i> — IoU 0.277 versus 0.286 for an all-empty mask. And the
best method found here wins only by degenerating into a temporal on/off gate:
a trivial baseline that floods the <i>entire</i> footprint whenever the tactile
diff exceeds a threshold matches it exactly. The height map contributes almost
no spatial information, so tuning the pressing-depth estimate — the thing this
investigation was aimed at — is largely a dead end.
</div>
""")

# ── chart ────────────────────────────────────────────────────────────────────
body.append('<h2>Held-out accuracy</h2>')
body.append('<p class="sub">IoU over all frames, empty-vs-empty counted as a '
            'match. Higher is better. Grey = degenerate reference that uses no '
            'geometry; blue = a real method.</p>')
body.append('<div class="legend"><span><i class="sw method"></i>method</span>'
            '<span><i class="sw ref"></i>degenerate reference</span></div>')
body.append('<div class="chart">')
for r in res["rows"]:
    body.append(f'<div class="row"><div class="lab">{r["name"]}</div>'
                f'{bar(r["iou_all"], r["kind"])}</div>')
body.append(f'<div class="row"><div class="lab em">ORACLE per-frame '
            f'(ceiling)</div>{bar(res["oracle_per_frame"], "ref")}</div>')
body.append(f'<div class="row"><div class="lab em">ORACLE per-touch</div>'
            f'{bar(res["oracle_per_touch"], "ref")}</div>')
body.append('</div>')

# ── table ────────────────────────────────────────────────────────────────────
body.append("""
<h2>Full metrics</h2>
<div class="scroll"><table>
<thead><tr><th>configuration</th><th>IoU<sub>all</sub></th>
<th>IoU<sub>contact</sub></th><th>Dice</th><th>false-positive area</th></tr></thead><tbody>
""")
for r in res["rows"]:
    em = ' class="refrow"' if r["kind"] == "ref" else ""
    body.append(f'<tr{em}><td>{r["name"]}</td><td>{r["iou_all"]:.4f}</td>'
                f'<td>{r["iou_contact"]:.4f}</td><td>{r["dice"]:.4f}</td>'
                f'<td>{r["fp_area"]:.4f}</td></tr>')
body.append(f'<tr class="refrow"><td>ORACLE per-frame</td>'
            f'<td>{res["oracle_per_frame"]:.4f}</td><td>—</td><td>—</td><td>—</td></tr>')
body.append(f'<tr class="refrow"><td>ORACLE per-touch</td>'
            f'<td>{res["oracle_per_touch"]:.4f}</td><td>—</td><td>—</td><td>—</td></tr>')
body.append('</tbody></table></div>')
body.append('<p class="note">IoU<sub>contact</sub> is over frames that actually '
            'contain contact; false-positive area is the mean predicted area on '
            'the 28% of frames with no contact. Both are needed — each metric '
            'alone has a degenerate optimum.</p>')

# ── findings ─────────────────────────────────────────────────────────────────
body.append("""
<h2>Findings</h2>

<h3>1. The production mask is no better than a blank mask</h3>
<p>At the shipped <code>--render_mask_thres -0.005</code> the ARuCO-driven mask
scores IoU 0.277 against 0.286 for always-empty. On frames that do contain
contact it reaches 0.112, while simply marking the entire valid footprint
scores 0.162 — i.e. thresholding the height map is <i>worse than not
thresholding it at all</i>.</p>

<h3>2. The marker carries no per-touch level information</h3>
<p>Regressing the oracle's ideal threshold against the available signals
(<code>test_scripts/analyze_oracle_thr.py</code>):</p>
<div class="scroll"><table><thead><tr><th>signal</th>
<th>within-touch r (temporal)</th><th>between-touch r (level)</th></tr></thead><tbody>
<tr><td>ARuCO pressing depth</td><td>+0.51</td><td class="bad">−0.08</td></tr>
<tr><td>tactile diff-from-blank</td><td class="good">+0.64</td><td class="good">+0.47</td></tr>
</tbody></table></div>
<p>The per-touch level is <b>52%</b> of the variance the threshold has to
explain, and the marker explains none of it. That is the mechanism behind
finding 1, and it is why every diff-driven variant beats the baseline.</p>

<h3>3. …but the diff-driven win is just temporal gating</h3>
<p>The best-scoring schedule (<code>diff_affine</code>, threshold affine in the
diff) reaches 0.391. So does a reference that marks the <i>whole</i> valid
region whenever the diff clears a fixed threshold and nothing otherwise
(0.391). Its median mask covers <b>99% of the valid region</b> during contact.
It has learned <i>when</i>, not <i>where</i>.</p>

<h3>4. The spatial ceiling itself is low</h3>
<p>An oracle allowed to pick the ideal threshold per frame reaches only 0.439,
against 0.402 for the oracle-gated flood-fill. <b>The entire spatial value of
the height map, extracted perfectly, is worth ~0.037 IoU.</b></p>
<p>Per-touch, the mask is also misplaced: allowing a per-touch 2D shift lifts
the oracle by <b>+28%</b>, and 96% of touches want a shift over 8&nbsp;px. But a
single <i>global</i> shift — i.e. a correction to
<code>ARUCO_TO_CONTACT_X_M/_Y_M</code> — recovers only <b>+1.5%</b> held out.
The error is per-touch pose noise, not a calibration constant.</p>

<h3>5. A real, separate bug: the two videos are on different clocks</h3>
<p><code>{i}_render_mask.mp4</code> spans the whole detected segment uniformly,
but the <code>{i}_shadow.mp4</code> it is written beside — and hstacked with in
<code>{i}_shadow_render_mask.mp4</code> — is <code>trim_and_resample</code>'d to
the contact window. That window is on average <b>64%</b> of the segment (median
51%), so frame <i>t</i> of the two videos refers to a different instant for
<b>71% of touches</b>. This is independent of everything above.</p>
""")

# ── filmstrips ───────────────────────────────────────────────────────────────
if len(strips):
    body.append('<h2>What the masks look like</h2>'
                '<p class="sub"><span class="key ok"></span>correct&nbsp;'
                '<span class="key fp"></span>false positive&nbsp;'
                '<span class="key fn"></span>missed. Columns are early / peak / '
                'late within one touch.</p>')
    for e in strips[:3]:
        body.append(f'<h4>{e["name"]}</h4><div class="scroll"><table class="strip"><tbody>')
        for meth in ["baseline", "diff_scaled", "diff_affine"]:
            body.append(f'<tr><th>{meth}</th>')
            for col in e["cols"]:
                body.append('<td><img alt="mask overlay" src="data:image/png;base64,'
                            + col["imgs"][meth] + '"></td>')
            body.append('</tr>')
        body.append('</tbody></table></div>')

# ── recommendations ──────────────────────────────────────────────────────────
body.append("""
<h2>Recommendations</h2>
<ol>
<li><b>Fix the timeline bug.</b> Independent of any algorithm change and a
strict improvement in what the paired video means. Now the default: the render
mask is sampled over the same contact window the shadow video is trimmed to
(see <code>render_mask_eval_positions</code>). Re-run
<code>process_single_shot.py</code> to regenerate aligned outputs.</li>

<li><b>Do not ship <code>diff_affine</code>.</b> It scores best but is a
temporal gate that throws the geometry away — almost certainly not what a
render mask is for. Implemented and documented as
<code>--render_mask_depth_mode diff_affine</code> so the result is reproducible,
with the caveat in the code.</li>

<li><b>Consider <code>diff_scaled</code> as the conservative default.</b> Keeps
the geometric threshold and only replaces frame-to-frame marker jitter:
0.277 → 0.336 IoU without saturating (mask areas stay in the 0.3–0.7 range on a
held-out session). It is a real but modest gain.</li>

<li><b>Redirect effort to pose/geometry fidelity.</b> Findings 3–4 say the
depth schedule is close to exhausted. The open questions are why the projected
height map lands in the wrong place per touch — ARuCO pose jitter, the
marker→gel offset, or ZED depth quality on these objects — since that is what
caps the ceiling at 0.44.</li>

<li><b>Re-examine whether the pseudo-GT is the right target.</b> Every number
here is against a mask derived from the tactile image. If the render mask is
meant as tactile-<i>independent</i> supervision, then diff-driven modes are
circular by construction and even this benchmark is measuring the wrong
thing.</li>
</ol>

<h2>Caveats</h2>
<ul>
<li>Pseudo-GT is <code>compute_contact_mask</code> at threshold 0.05. The
ranking baseline &lt; diff_scaled &lt; diff_affine holds at 0.03 and 0.08 too,
with morphological opening applied end-to-end.</li>
<li>The diff-driven modes use the tactile image, which the pseudo-GT is also
derived from. Their advantage is therefore partly definitional — a further
reason not to read finding 3 as a win.</li>
<li>Absolute diff levels shift between sessions (blank frame, lighting, gel).
<code>diff_affine</code> is anchored to each segment's own no-contact minimum
for that reason, but still saturated on a held-out session.</li>
<li>Threshold-response tables omit morphological opening for speed; the final
comparison was re-verified with it on.</li>
</ul>

<h2>Reproducing</h2>
<pre><code>conda activate pm_real

# cache pseudo-GT + geometry per touch, then score (morph applied)
python test_scripts/eval_render_mask.py \\
    --variants baseline,diff_max_raw,diff_affine

# sweeps + oracles via threshold-response tables
python test_scripts/sweep_render_mask.py --sweep main
python test_scripts/sweep_render_mask.py --sweep affine

# the diagnostics behind findings 2 and 4
python test_scripts/analyze_oracle_thr.py
python test_scripts/analyze_spatial_align.py
python test_scripts/analyze_global_shift.py</code></pre>
""")

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Render-mask study</title><style>
:root{color-scheme:light dark;
 --surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#77756f;
 --line:#e3e2dd;--method:#2a78d6;--ref:#9a9892;--good:#008300;--bad:#e34948;
 --warnbg:#fff8e6;--warnln:#eda100}
@media (prefers-color-scheme:dark){:root{
 --surface:#1a1a19;--card:#232322;--ink:#fff;--ink2:#c3c2b7;--ink3:#96958c;
 --line:#3a3a38;--method:#3987e5;--ref:#6f6e69;--good:#3fa93f;--bad:#e66767;
 --warnbg:#2e2718;--warnln:#c98500}}
*{box-sizing:border-box}
body{margin:0;padding:2.2rem 1.2rem 5rem;background:var(--surface);color:var(--ink);
 font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:56rem;margin:0 auto}
h1{font-size:1.75rem;line-height:1.25;margin:0 0 .5rem}
h2{font-size:1.2rem;margin:2.6rem 0 .7rem;padding-top:1.1rem;border-top:1px solid var(--line)}
h3{font-size:1rem;margin:1.6rem 0 .4rem}
h4{font-size:.85rem;margin:1.4rem 0 .4rem;color:var(--ink2);font-family:ui-monospace,monospace}
p,li{color:var(--ink2)} li{margin:.35rem 0}
.sub{color:var(--ink3);font-size:.92rem}
.note{font-size:.85rem;color:var(--ink3)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;
 background:var(--card);border:1px solid var(--line);border-radius:4px;padding:.05em .35em}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:.9rem 1rem;overflow-x:auto}
pre code{background:none;border:0;padding:0;font-size:.82rem;line-height:1.6}
.callout{border-radius:8px;padding:.9rem 1.1rem;margin:1.4rem 0;font-size:.94rem;
 background:var(--card);border:1px solid var(--line)}
.callout.warn{background:var(--warnbg);border-left:3px solid var(--warnln)}
.chart{margin:1rem 0 .4rem}
.row{display:flex;align-items:center;gap:.7rem;margin:.28rem 0}
.lab{flex:0 0 15rem;font-size:.83rem;color:var(--ink2);text-align:right;
 overflow-wrap:anywhere}
.lab.em{font-style:italic;color:var(--ink3)}
.barwrap{flex:1;display:flex;align-items:center;gap:.55rem;min-width:0}
.bar{height:15px;border-radius:0 4px 4px 0;min-width:2px}
.bar.method{background:var(--method)} .bar.ref{background:var(--ref)}
.barval{font-size:.78rem;color:var(--ink2);font-variant-numeric:tabular-nums}
.legend{display:flex;gap:1.1rem;font-size:.82rem;color:var(--ink2);margin:.5rem 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:.35rem;
 vertical-align:-1px}
.sw.method{background:var(--method)} .sw.ref{background:var(--ref)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.6rem 0}
th,td{text-align:left;padding:.42rem .7rem;border-bottom:1px solid var(--line);
 white-space:nowrap}
th{color:var(--ink3);font-weight:600}
td{color:var(--ink2);font-variant-numeric:tabular-nums}
tr.refrow td{color:var(--ink3);font-style:italic}
td.good,.good{color:var(--good)} td.bad,.bad{color:var(--bad)}
table.strip td{padding:.2rem}
table.strip th{font-family:ui-monospace,monospace;font-size:.75rem;
 color:var(--ink2);font-style:normal}
table.strip img{width:150px;height:auto;display:block;border-radius:4px;
 border:1px solid var(--line);max-width:100%}
.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 .25rem 0 .6rem;
 vertical-align:-1px}
.key.ok{background:#6ec86e}.key.fp{background:#eb5a5a}.key.fn{background:#5aa0eb}
</style></head><body><main>
""" + "\n".join(body) + "</main></body></html>"

os.makedirs("log", exist_ok=True)
out = osp.join("log", "render_mask_study.html")
open(out, "w").write(HTML)
print(f"wrote {out} ({len(HTML)/1024:.0f} KB)")
