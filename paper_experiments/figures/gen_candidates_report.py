"""Build the HTML summary of figure candidates.

Reads log/paper_figure_candidates/candidates.json and embeds each candidate's
preview, together with the numbers that got it selected, so the choice can be
checked rather than taken on trust.

The previews shown here are the spaced ones written by space_and_export.py (white
space between the frames), and each candidate links to its raw panels and its PDF
under log/paper_figure_assets/. Run space_and_export.py before this script.
"""
import base64
import datetime
import glob
import json
import os

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
CAND = os.path.join(ROOT, "log/paper_figure_candidates/candidates.json")
OUT = os.path.join(ROOT, "log/paper_figure_candidates_report.html")
ASSETS = "log/paper_figure_assets"

FIGURES = [
    ("teaser", "fig_teaser.tex", "Teaser (one column)",
     "A : A' :: B : B'. Top row is the reference geometry and the touch recorded there; "
     "bottom row is the query geometry and our prediction for it.",
     "Ranked by visible structure, by how much the reference and query poses actually "
     "differ (a near-identical pair would make the analogy trivial), and by prediction quality.",
     ["structure", "ref_query_dist", "psnr_ours"]),
    ("gt_retrieval", "fig_gt_retrieval.tex", "Ground-truth retrieval, 3 x 6 (two columns)",
     "Columns: reference touch, reference normal, query normal, coarse transfer, refined "
     "prediction, ground truth. Pick any three rows.",
     "Ranked by structure and by gain (refined minus coarse PSNR), so the refinement column "
     "visibly differs from the coarse column.",
     ["structure", "gain", "psnr_ours"]),
    ("full_pipeline", "fig_full_pipeline.tex", "Full-pipeline comparison (two columns)",
     "Rows: reference video, quilting, ObjectFolder INR, ours coarse, ours refined, ground "
     "truth. Columns are frames across the press.",
     "Ranked by structure and by our margin over the best baseline on that touch, so the "
     "figure agrees with the quantitative table. Touches whose ObjectFolder INR row comes "
     "out as a single flat colour are passed over, since a blank baseline row makes the "
     "comparison pointless; inr_content below is how much that row actually draws (0 is "
     "flat, the best touches here reach about 0.024). Columns are sampled from the part of "
     "the press that is in contact, so no row opens or closes on an empty panel.",
     ["structure", "psnr_ours", "best_baseline", "margin", "inr_content"]),
    ("ablation", "fig_ablation.tex", "Ablation of the refinement stage (one column)",
     "Panels: w/o refinement, w/o temporal FiLM, w/o normal concatenation, ours, ground truth.",
     "Ranked by ablation gap (ours minus the better ablation) so removing a component is "
     "visibly damaging, not just numerically.",
     ["abl_gap", "structure", "psnr_ours", "psnr_wo_film", "psnr_wo_cat"]),
    ("recon", "fig_recon.tex", "3D reconstruction and visuo-tactile simulation (two columns)",
     "Rows: reference video, predicted video, 3D relief integrated from the prediction, "
     "simulated visuo-tactile RGB. Columns are in-contact frames.",
     "Ranked by relief depth and contact area, so there is real 3D shape to integrate. "
     "Rows 3 and 4 come from height3d_geomcat_film.py and predicted_normal_to_rgb.py.",
     ["relief", "contact_frac", "psnr_ours"]),
    ("method", "fig_method.tex", "Method overview, panel (b) (one column)",
     "The coarse-alignment illustration: SuperPoint/SuperGlue correspondences between the "
     "reference and query normal maps, and the homography fitted to them.",
     "Ranked by inlier count and ratio, in-bounds warp fraction, and whether the offset stage "
     "succeeded, so the panel shows clean matches rather than a degenerate fit.",
     ["matches", "inliers", "valid_fraction", "psnr_coarse"]),
]

CSS = """
:root{--bg:#fff;--fg:#16181d;--mut:#5f6672;--line:#e3e6ea;--panel:#f6f8fa;--acc:#1f5fa8}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e9ecef;--mut:#9aa2ad;
 --line:#2a2f36;--panel:#1b1f25;--acc:#7db3f0}}
:root[data-theme="dark"]{--bg:#14161a;--fg:#e9ecef;--mut:#9aa2ad;--line:#2a2f36;--panel:#1b1f25;--acc:#7db3f0}
:root[data-theme="light"]{--bg:#fff;--fg:#16181d;--mut:#5f6672;--line:#e3e6ea;--panel:#f6f8fa;--acc:#1f5fa8}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.1rem 5rem;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .3rem;letter-spacing:-.01em}
h2{font-size:1.2rem;margin:2.6rem 0 .3rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 1.4rem}
.spec{color:var(--mut);font-size:14px;margin:.2rem 0 .5rem}
.how{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);
 border-radius:6px;padding:.6rem .85rem;font-size:13.5px;color:var(--mut);margin:.5rem 0 1.1rem}
.cand{border:1px solid var(--line);border-radius:9px;padding:.85rem;margin:1rem 0;background:var(--panel)}
.cand.top{border-color:var(--acc);border-width:2px}
.hdr{display:flex;flex-wrap:wrap;gap:.5rem 1rem;align-items:baseline;margin-bottom:.55rem}
.id{font-weight:650;font-size:15px}
.badge{font-size:11.5px;font-weight:650;padding:.1rem .5rem;border-radius:999px;
 background:color-mix(in srgb,var(--acc) 17%,transparent);color:var(--acc)}
.stats{color:var(--mut);font-size:12.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.files{margin-top:.5rem;font-size:12.5px;color:var(--mut)}
img{max-width:100%;height:auto;display:block;border-radius:5px;border:1px solid var(--line);background:#000}
.scroll{overflow-x:auto}
code{background:var(--panel);padding:.08rem .32rem;border-radius:4px;font-size:12.5px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--mut);font-size:13px}
"""


def b64(p):
    p = p if p.startswith("/") else os.path.join(ROOT, p)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def main():
    with open(CAND) as f:
        cands = json.load(f)

    secs = []
    for key, tex, title, spec, how, stats in FIGURES:
        items = cands.get(key, [])
        blocks = []
        for i, c in enumerate(items):
            adir = f"{ASSETS}/{key}/{c['object']}_{c['touch']}"
            if key == "method":
                src = b64(f"{adir}/figure.png")
                if src is None:
                    figs = sorted(glob.glob(os.path.join(ROOT, c["dir"], "*_matches.png")))
                    pick = [p for p in figs if os.path.basename(p).startswith(f"{c['touch']}_")]
                    src = b64(pick[0] if pick else (figs[0] if figs else ""))
            else:
                # spaced version first, tight original only if it has not been built yet
                src = b64(c["preview"].replace("preview.png", "preview_spaced.png")) \
                    or b64(c["preview"])
            line = "  ".join(f"{s}={fmt(c[s])}" for s in stats if s in c and c[s] is not None)
            has_assets = os.path.isdir(os.path.join(ROOT, adir))
            files = f"""
  <div class="files">raw frames <code>{adir}/panels/</code>
    &middot; PDF <code>{adir}/figure.pdf</code></div>""" if has_assets else ""
            blocks.append(f"""
<div class="cand{' top' if i == 0 else ''}">
  <div class="hdr">
    <span class="id">object {c['object']}, touch {c['touch']}</span>
    {'<span class="badge">top pick</span>' if i == 0 else ''}
    <span class="stats">{esc(line)}</span>
  </div>
  <div class="scroll">{f'<img src="{src}" alt="candidate preview">' if src else
                       '<p class="stats">preview missing</p>'}</div>{files}
</div>""")
        allpdf = f"{ASSETS}/{key}/{key}_all_candidates.pdf"
        pdfline = (f'<p class="spec">All {len(items)} candidates as one PDF, a page each: '
                   f'<code>{allpdf}</code></p>'
                   if os.path.exists(os.path.join(ROOT, allpdf)) else "")
        secs.append(f"""
<h2>{esc(title)}</h2>
<p class="spec"><code>paper_source/figures/{tex}</code> &mdash; {esc(spec)}</p>
<div class="how"><strong>How these were chosen.</strong> {esc(how)}</div>
{pdfline}
{''.join(blocks) if blocks else '<p class="stats">no candidates generated</p>'}""")

    html = f"""<title>Figure candidates &mdash; Tactile Analogies</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Qualitative figure candidates</h1>
<p class="sub">Five options for each figure in <code>paper_source/figures/</code>, with the
numbers behind each pick</p>
<div class="how">
<p style="margin:.2rem 0"><strong>Why not just pick by PSNR.</strong> The highest-PSNR touches in
this benchmark are the flattest, most featureless contacts &mdash; easy to predict and
uninformative to look at. Each figure below therefore ranks on its own objective, combining how
much there is to see with how well the method did. All candidates are on distinct objects.</p>
<p style="margin:.5rem 0 .2rem"><strong>Where the assets are.</strong> Everything needed to
lay a chosen candidate out in the paper, without re-running anything, is under
<code>log/paper_figure_assets/&lt;figure&gt;/&lt;object&gt;_&lt;touch&gt;/</code>:
<code>panels/</code> holds the individual frames at 320&times;240 with the caption text
removed (these are the ones to put in LaTeX, so labels can be set in the document),
<code>figure.png</code> is the spaced layout shown below, and <code>figure.pdf</code> is
the same layout as a PDF page. Each figure also has a single
<code>&lt;figure&gt;_all_candidates.pdf</code> with one candidate per page.
<code>log/paper_figure_assets/manifest.json</code> indexes all of it. Scores stay in
<code>log/paper_figure_candidates/candidates.json</code>, per-touch descriptors in
<code>descriptors.json</code>.</p>
<p style="margin:.5rem 0 .2rem"><strong>About the spacing.</strong> The previews below put
white space between neighbouring frames, so it is clear where one frame ends and the next
begins. The gap is only in the preview; the frames in <code>panels/</code> are untouched.</p>
</div>
{''.join(secs)}
<footer>Generated {datetime.datetime.now():%Y-%m-%d %H:%M} &middot;
 selection: <code>paper_experiments/figures/</code> &middot;
 3D from <code>height3d_geomcat_film.py</code>, RGB from <code>predicted_normal_to_rgb.py</code>
</footer></div>"""
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    for key, *_ in FIGURES:
        print(f"  {key:16s} {len(cands.get(key, []))} candidates")


if __name__ == "__main__":
    main()
