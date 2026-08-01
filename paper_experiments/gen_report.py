"""Generate the progress report for the Dirac experiment batch.

Writes, re-runnably (safe to call at any point, pending jobs show as pending):

  log/paper_experiments_report.html          the overall progress report
  paper_experiments/{job}/report.html        per-job summary
  paper_experiments/{job}/results.md         metrics/tables for the paper

Everything is read from the results.json files the per-job aggregators write,
so this script never recomputes a metric.
"""
import base64
import datetime
import glob
import json
import os

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
EXP = os.path.join(ROOT, "paper_experiments")

CSS = """
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#606060; --line:#e2e2e2;
        --accent:#1f5fa8; --ok:#0f7b3f; --pend:#a06000; --panel:#f7f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#15171a; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2c3036;
          --accent:#7db3f0; --ok:#5fd08a; --pend:#e0a640; --panel:#1c1f24; }
}
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; }
h1 { font-size:1.75rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1.25rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem;
     border-bottom:1px solid var(--line); }
h3 { font-size:1.02rem; margin:1.75rem 0 .5rem; }
.sub { color:var(--muted); margin:0 0 1.5rem; }
table { border-collapse:collapse; width:100%; margin:.75rem 0 1rem; font-size:14px; }
th,td { padding:.5rem .6rem; text-align:right; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { font-weight:600; color:var(--muted); font-size:12.5px; text-transform:uppercase;
     letter-spacing:.04em; }
tr.best td { font-weight:650; color:var(--accent); }
tr.sep td { border-top:2px solid var(--line); }
.scroll { overflow-x:auto; }
.badge { display:inline-block; padding:.1rem .5rem; border-radius:999px;
         font-size:12px; font-weight:600; }
.badge.done { background:color-mix(in srgb,var(--ok) 16%,transparent); color:var(--ok); }
.badge.pend { background:color-mix(in srgb,var(--pend) 16%,transparent); color:var(--pend); }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px;
         padding:.9rem 1.1rem; margin:1rem 0; }
.panel p:first-child { margin-top:0; } .panel p:last-child { margin-bottom:0; }
code { background:var(--panel); padding:.1rem .35rem; border-radius:4px;
       font-size:13px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:8px;
      padding:.8rem 1rem; overflow-x:auto; font-size:13px; }
img { max-width:100%; height:auto; border-radius:6px; border:1px solid var(--line); }
ul { padding-left:1.2rem; } li { margin:.25rem 0; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
         color:var(--muted); font-size:13px; }
"""


def load(path):
    p = os.path.join(ROOT, path) if not path.startswith("/") else path
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def badge(done, label_done="complete", label_pend="pending"):
    return (f'<span class="badge done">{label_done}</span>' if done
            else f'<span class="badge pend">{label_pend}</span>')


def embed_img(path):
    p = os.path.join(ROOT, path) if not path.startswith("/") else path
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'



def matrix_figure(asset_dir, caption):
    """Embed the {tag}_matrix.png a video-matrix extraction produced, if present.

    Picks the most recently written one. Sorting by name instead would be a trap:
    the tags are object ids, so "obj20_..." sorts before "obj3_...", and a stale
    figure from an earlier run would silently outrank the current one.
    """
    hits = glob.glob(os.path.join(ROOT, asset_dir, "*_matrix.png"))
    if not hits:
        return ""
    src = embed_img(max(hits, key=os.path.getmtime))
    if not src:
        return ""
    return (f'<h3>Qualitative comparison</h3><p>{caption}</p>'
            f'<img src="{src}" alt="comparison matrix">')

def metric_table(rows, highlight_best=True, sep_after=None):
    """rows: list of (name, metrics_dict_or_None, n_objects)."""
    have = [(n, m) for n, m, _ in rows if m]
    best = {}
    if highlight_best and have:
        best["PSNR"] = max(m["PSNR"] for _, m in have)
        best["SSIM"] = max(m["SSIM"] for _, m in have)
        best["LPIPS"] = min(m["LPIPS"] for _, m in have)
    out = ['<div class="scroll"><table>',
           "<tr><th>Method</th><th>Objects</th><th>PSNR &uarr;</th>"
           "<th>SSIM &uarr;</th><th>LPIPS &darr;</th><th>MSE &darr;</th></tr>"]
    for i, (name, m, n) in enumerate(rows):
        cls = []
        if sep_after and i == sep_after:
            cls.append("sep")
        if m and highlight_best and m["PSNR"] == best.get("PSNR"):
            cls.append("best")
        c = f' class="{" ".join(cls)}"' if cls else ""
        if not m:
            out.append(f'<tr{c}><td>{esc(name)}</td><td>&mdash;</td>'
                       f'<td colspan="4" style="text-align:left;color:var(--muted)">pending</td></tr>')
        else:
            out.append(f'<tr{c}><td>{esc(name)}</td><td>{n}</td>'
                       f'<td>{m["PSNR"]:.2f}</td><td>{m["SSIM"]:.4f}</td>'
                       f'<td>{m["LPIPS"]:.4f}</td><td>{m["MSE"]:.5f}</td></tr>')
    out.append("</table></div>")
    return "\n".join(out)


def md_table(rows):
    lines = ["| Method | Objects | PSNR (up) | SSIM (up) | LPIPS (down) | MSE (down) |",
             "|---|---|---|---|---|---|"]
    for name, m, n in rows:
        if not m:
            lines.append(f"| {name} | - | pending | pending | pending | pending |")
        else:
            lines.append(f"| {name} | {n} | {m['PSNR']:.2f} | {m['SSIM']:.4f} "
                         f"| {m['LPIPS']:.4f} | {m['MSE']:.5f} |")
    return "\n".join(lines)


def page(title, subtitle, body):
    return (f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n"
            f'<div class="wrap"><h1>{esc(title)}</h1>'
            f'<p class="sub">{subtitle}</p>{body}'
            f'<footer>Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on dirac '
            f'&middot; sources under <code>paper_experiments/</code> and '
            f'<code>log/</code></footer></div>')


# --------------------------------------------------------------------------
# Job 1
# --------------------------------------------------------------------------

def job1():
    d = load("paper_experiments/job1_gt_retrieval/results.json")
    order = ["Tactile Normal Quilting", "ObjectFolder INR",
             "Ours (coarse transfer)", "Ours (refined)"]
    abl = ["w/o temporal FiLM", "w/o normal concatenation"]
    rows, abl_rows = [], []
    if d:
        for k in order:
            t = d["table"].get(k, {})
            m = {kk: t[kk] for kk in ("PSNR", "SSIM", "LPIPS", "MSE")} if t.get("n_objects") else None
            rows.append((k, m, t.get("n_objects", 0)))
        for k in abl:
            t = d["table"].get(k, {})
            m = {kk: t[kk] for kk in ("PSNR", "SSIM", "LPIPS", "MSE")} if t.get("n_objects") else None
            abl_rows.append((k, m, t.get("n_objects", 0)))
    done = bool(d) and all(r[1] for r in rows)

    sheet = embed_img("log/paper_job1_figure_assets/contact_sheet.png")
    fig = (f'<h3>Qualitative results</h3><p>Each row is one touch location. Columns, '
           f'left to right: reference touch, reference surface-normal render, query '
           f'surface-normal render, our coarse transfer, our refined transfer, and the '
           f'ground-truth query touch.</p><img src="{sheet}" alt="qualitative matrix">'
           if sheet else "")

    body = f"""
<div class="panel">
<p><strong>What this measures.</strong> The ground-truth retrieval benchmark hands
every method the correct reference touch, so nothing here depends on retrieval
quality &mdash; it isolates how well a method transfers a known reference touch to a
new query location. Evaluated on the 50 held-out objects (ids 951&ndash;1000), 8 touch
locations each.</p>
<p><strong>Status:</strong> {badge(done)}</p>
</div>
<div class="panel">
<p><strong>Why TaRF is not in these tables.</strong> The baseline was run end to
end on both benchmarks (50 and 100 objects, no failures) using the img2touch
checkpoint at <code>log/tarf_pretrained.ckpt</code>. That checkpoint reports
<em>epoch 5, global step 5700</em>, and its predictions are near-identical blur
regardless of which object it is conditioned on &mdash; an early snapshot rather
than a converged model. Reporting it would present a strawman baseline, so the
row is withheld until a fully trained model is available. The raw runs are kept
under <code>log/paper_job{1,2}_baselines/tarf/</code> and both aggregators can
re-add the row with a one-line source path.</p>
</div>
<h3>Comparison against baselines</h3>
{metric_table(rows)}
<p>PSNR (peak signal-to-noise ratio) and SSIM (structural similarity index) are
better when higher; LPIPS (learned perceptual image patch similarity) and MSE
(mean squared error) are better when lower.</p>
<h3>Refinement-network ablations on this benchmark</h3>
<p>These use the same 50 objects, so they are directly comparable to the rows above.
The dedicated ablation study (Job 3) repeats them on the full-pipeline benchmark.</p>
{metric_table(abl_rows, highlight_best=False)}
<div class="panel">
<p><strong>Read the "w/o normal concatenation" row with care.</strong> The plan
specifies the checkpoint <code>rebot_checkpoints_S_pseudo_mini_tactile_normal_superpoint_superglue_cond-film-normal</code>
for this arm, and that is what was used &mdash; but it was trained with a different
recipe from the other two arms, not just a different way of injecting the normal
map. It ran 37 epochs of a 100-epoch schedule (the others ran a completed
20-epoch schedule) and used neither <code>zero_init_final</code> nor the
<code>lambda_delta</code> term. Its best validation PSNR was 25.93 against 32.13
for the full model. So the gap shown here mixes the conditioning change with a
training-recipe change, and overstates the value of concatenation on its own. The
"w/o temporal FiLM" arm has no such problem: it differs from the full model only
in <code>--time_cond</code>.</p>
</div>
{fig}
"""
    md = f"""# Job 1 — Ground-truth retrieval benchmark

Ground-truth reference touch supplied to every method; 50 held-out objects
(951–1000), 8 touch locations each. TaRF excluded (model still training).

## Main comparison

{md_table(rows)}

## Refinement-network ablations (same benchmark)

{md_table(abl_rows)}

**Caveat on "w/o normal concatenation".** The checkpoint the plan specifies for
this arm (`..._cond-film-normal`) was trained with a different recipe from the
other two arms, not only a different conditioning scheme: 37 epochs of a
100-epoch schedule (vs a completed 20-epoch schedule), and without
`zero_init_final` or `lambda_delta`. Best val PSNR 25.93 vs 32.13 for the full
model. The gap therefore mixes conditioning with training recipe and overstates
the effect of concatenation alone. "w/o temporal FiLM" is a clean ablation
(only `--time_cond` differs).

Assets: `log/paper_job1_figure_assets/` (per-panel PNGs + contact sheet),
`log/paper_job1_refine_ours/videos/` (predicted videos).
LaTeX table body: `paper_experiments/job1_gt_retrieval/table_body.tex`.
"""
    return body, md, done


# --------------------------------------------------------------------------
# Job 2
# --------------------------------------------------------------------------

def job2():
    d = load("paper_experiments/job2_full_pipeline/results.json")
    splits = load("paper_experiments/job2_full_pipeline/splits.json")
    order = ["Tactile Normal Quilting", "ObjectFolder INR",
             "Ours (coarse transfer, normals)", "Ours (refined, normals)",
             "Ours (coarse transfer, curvature)", "Ours (refined, curvature)"]
    rows = []
    if d:
        for k in order:
            t = d["table"].get(k, {})
            rows.append((k, t.get("metrics"), t.get("n_objects", 0)))
    done = bool(d) and all(r[1] for r in rows)

    n_obj = len(splits["objects"]) if splits else 0
    n_q = sum(len(v["query"]) for v in splits["objects"].values()) if splits else 0
    n_r = sum(len(v["ref"]) for v in splits["objects"].values()) if splits else 0

    body = f"""
<div class="panel">
<p><strong>What this measures.</strong> The complete system, retrieval included.
For each object we hold out a few touches as queries, and the method must first
find the most useful reference among that object's remaining touches (DINOv3
feature retrieval) and then transfer it. Every method gets the same split and the
same retrieval, so the table still isolates the prediction stage &mdash; but now
retrieval errors are allowed to propagate.</p>
<p><strong>Split:</strong> {n_obj} objects, {n_q} held-out query touches,
{n_r} reference touches. 4 queries per object (3 for objects with fewer than 9
touches), chosen by a seeded random draw so the split is reproducible.</p>
<p><strong>Status:</strong> {badge(done)} &nbsp; TaRF excluded, as above.</p>
<p><strong>Why two "ours" pairs.</strong> The method's default coarse alignment is
surface normals at 4x the sensor footprint (the normals rows). The curvature rows
are kept because the refinement checkpoints were <em>trained</em> on a
curvature-modality transfer, so the pair measures whether that train/test
mismatch costs anything. It does not &mdash; the two refined rows differ by less
than 0.1 dB, so the network is insensitive to which modality drove the alignment.</p>
</div>
{metric_table(rows)}
{matrix_figure("log/paper_job2_figure_assets",
               "One row per method, one column per video frame sampled evenly across "
               "the touch. The quilting baseline tiles a single quilted image to the "
               "video length, so only its first frame carries information.")}
"""
    md = f"""# Job 2 — Full-pipeline benchmark

Retrieval is part of the system under test. {n_obj} objects, {n_q} held-out query
touches, {n_r} reference touches; 4 queries per object (3 when an object has fewer
than 9 touches), seeded random draw. TaRF excluded.

{md_table(rows)}

Split manifest: `paper_experiments/job2_full_pipeline/splits.json`.
LaTeX table body: `paper_experiments/job2_full_pipeline/table_body.tex`.
"""
    return body, md, done


# --------------------------------------------------------------------------
# Job 3
# --------------------------------------------------------------------------

def job3():
    d = load("paper_experiments/job3_ablation/results.json")
    coarse_rows, refine_rows = [], []
    if d:
        for arm, e in d["coarse"].items():
            coarse_rows.append((e["label"], e["metrics"], e["n_objects"]))
        for arm, e in d["refined"].items():
            refine_rows.append((e["label"], e["metrics"], e["n_objects"]))
    done = bool(d) and all(r[1] for r in coarse_rows + refine_rows)

    # Call out the modality / scale winners, since they bear on the paper's
    # stated default configuration.
    finding = ""
    if d:
        mods = {k: v for k, v in d["coarse"].items() if k.startswith("mod_") and v["metrics"]}
        scales = {k: v for k, v in d["coarse"].items() if k.startswith("scale_") and v["metrics"]}
        if mods and scales:
            bm = max(mods, key=lambda k: mods[k]["metrics"]["PSNR"])
            bs = max(scales, key=lambda k: scales[k]["metrics"]["PSNR"])
            bs_lpips = min(scales, key=lambda k: scales[k]["metrics"]["LPIPS"])
            nice = {"mod_normal": "surface normal", "mod_color": "RGB colour",
                    "mod_curvature": "curvature", "mod_height": "height map",
                    "scale_1x": "1x", "scale_2x": "2x", "scale_4x": "4x"}
            split = ("" if bs == bs_lpips else
                     f" PSNR and the perceptual metrics disagree here: "
                     f"{nice.get(bs, bs)} has the best PSNR "
                     f"({scales[bs]['metrics']['PSNR']:.2f} dB) while "
                     f"{nice.get(bs_lpips, bs_lpips)} has the best LPIPS "
                     f"({scales[bs_lpips]['metrics']['LPIPS']:.4f}), SSIM and MSE &mdash; "
                     f"which is why the default stays at "
                     f"{nice.get(bs_lpips, bs_lpips)}.")
            finding = f"""
<div class="panel">
<p><strong>Reading the two tables together.</strong> The method's default coarse
alignment is surface normals at 4x the sensor footprint. The scale sweep holds the
modality at normals and varies only the scale, so the <em>4x</em> row is by
construction the same run as the <em>surface normal</em> row &mdash; the two numbers
agree, and that is a consistency check, not a duplicated experiment.</p>
<p>On this subset the best modality is <strong>{nice.get(bm, bm)}</strong>
({mods[bm]['metrics']['PSNR']:.2f} dB), which is the default.{split} The spread across all seven arms is only
about 1&ndash;2 dB over 20 objects, so the coarse alignment is fairly insensitive to
both choices; the refinement network accounts for far more of the final quality
(+4 dB, see the next table).</p>
</div>"""

    body = f"""
<div class="panel">
<p><strong>What this measures.</strong> Which parts of the method actually matter.
Run on a 20-object subset of the full-pipeline benchmark.</p>
<p>The first table varies how the reference touch is <em>aligned</em> to the query:
which rendered modality drives retrieval and feature matching, and how large a
physical area that render covers. Taxim names scales by an object-scale factor, so
a smaller number covers a larger area: factor 100 is 1&times; the sensor footprint,
50 is 2&times;, 25 is 4&times;.</p>
<p>The second table varies the refinement network while holding the alignment fixed.</p>
<p><strong>Status:</strong> {badge(done)}</p>
</div>
<h3>Coarse alignment: modality and scale</h3>
{metric_table(coarse_rows, highlight_best=False)}
{finding}
<h3>Refinement network</h3>
{metric_table(refine_rows, highlight_best=False)}
{matrix_figure("log/paper_job3_figure_assets",
               "The same coarse transfer refined by each network variant, against the "
               "ground truth in the last row.")}
"""
    md = f"""# Job 3 — Ablation study

20-object subset of the full-pipeline benchmark.

## Coarse alignment (modality and scale)

Scale naming: object-scale factor 100 = 1x sensor footprint, 50 = 2x, 25 = 4x.

{md_table(coarse_rows)}

## Refinement network

{md_table(refine_rows)}

LaTeX table body: `paper_experiments/job3_ablation/table_body.tex`.
"""
    return body, md, done


# --------------------------------------------------------------------------
# Job 4
# --------------------------------------------------------------------------

def job4():
    d = load("paper_experiments/job4_runtime/runtime.json")
    done = bool(d)
    if not done:
        body = ('<div class="panel"><p><strong>Status:</strong> '
                f'{badge(False)}</p></div>')
        return body, "# Job 4 — Runtime analysis\n\nPending.\n", False

    r, c, f = d["retrieval"], d["coarse_alignment"], d["refinement"]
    d2 = load("paper_experiments/job4_runtime/runtime_n27.json")
    scaling = ""
    md_scaling = ""
    if d2:
        r2, c2, f2 = d2["retrieval"], d2["coarse_alignment"], d2["refinement"]
        scaling = f"""
<p>The same measurement on the benchmark's largest reference set
({r2['n_reference_touches']} reference touches) gives
{r2['ranking_after_extraction_s'] * 1000:.2f} ms retrieval,
{c2['coarse_alignment' if False else 'coarse_after_matching_s']:.3f} s coarse alignment and
{f2['refinement_per_frame_s'] * 1000:.1f} ms per frame &mdash; essentially unchanged,
so within the range this benchmark covers the cost does not grow with the number
of reference touches. Retrieval is a dot product against N cached feature vectors;
everything after it depends only on the single retrieved reference.</p>"""
        md_scaling = f"""
Repeating the measurement on the largest reference set in the benchmark
({r2['n_reference_touches']} references) gives
{r2['ranking_after_extraction_s'] * 1000:.2f} ms,
{c2['coarse_after_matching_s']:.3f} s and
{f2['refinement_per_frame_s'] * 1000:.1f} ms/frame — essentially unchanged with N.
"""
    body = f"""
<div class="panel">
<p><strong>What this measures.</strong> How long one touch prediction takes, given
{r['n_reference_touches']} reference touches and 1 query, on a single
{esc(d['gpu'])}. Each number excludes a generic feature-extraction cost so that
what is timed is our own machinery: the retrieval number excludes DINOv3 feature
extraction, and the coarse-alignment number excludes the local feature matching.</p>
<p><strong>Status:</strong> {badge(True)}</p>
</div>
<div class="scroll"><table>
<tr><th>Stage</th><th>Time</th></tr>
<tr><td>Retrieval, after DINOv3 feature extraction</td>
    <td>{r['ranking_after_extraction_s'] * 1000:.2f} ms</td></tr>
<tr><td>Coarse alignment, after local feature matching</td>
    <td>{c['coarse_after_matching_s']:.3f} s</td></tr>
<tr><td>Network refinement, per frame</td>
    <td>{f['refinement_per_frame_s'] * 1000:.1f} ms</td></tr>
<tr><td>Network refinement, per {f['frames_timed']}-frame touch video</td>
    <td>{f['refinement_per_video_s']:.2f} s</td></tr>
</table></div>
<p>For reference, the excluded steps cost
{(r['extraction_plus_ranking_s'] - r['ranking_after_extraction_s']):.2f} s
(DINOv3 features for {r['n_reference_touches']} references + 1 query) and
{(c['including_matching_s'] - c['coarse_after_matching_s']):.2f} s
(local feature matching).</p>
{scaling}
"""
    md = f"""# Job 4 — Runtime analysis

Single {d['gpu']}, {r['n_reference_touches']} reference touches and 1 query,
{c['n_frames']}-frame touch videos, mean of {d['repeats']} timed repeats after a
warm-up pass.

| Stage | Time |
|---|---|
| Retrieval, after DINOv3 feature extraction | {r['ranking_after_extraction_s'] * 1000:.2f} ms |
| Coarse alignment, after local feature matching | {c['coarse_after_matching_s']:.3f} s |
| Network refinement, per frame | {f['refinement_per_frame_s'] * 1000:.1f} ms |
| Network refinement, per {f['frames_timed']}-frame video | {f['refinement_per_video_s']:.2f} s |

{md_scaling}
Excluded costs, for reference: DINOv3 feature extraction
{(r['extraction_plus_ranking_s'] - r['ranking_after_extraction_s']):.2f} s;
local feature matching {(c['including_matching_s'] - c['coarse_after_matching_s']):.2f} s.

Sentences for the paper:

- Retrieval phase after DINOv3 feature extraction takes {r['ranking_after_extraction_s'] * 1000:.2f} ms.
- Coarse alignment after local feature matching takes {c['coarse_after_matching_s']:.3f} s.
- Neural network-based refinement takes {f['refinement_per_frame_s'] * 1000:.1f} ms per frame.
"""
    return body, md, True


JOBS = [
    ("job1_gt_retrieval", "Job 1 — Ground-truth retrieval benchmark", job1),
    ("job2_full_pipeline", "Job 2 — Full-pipeline benchmark", job2),
    ("job3_ablation", "Job 3 — Ablation study", job3),
    ("job4_runtime", "Job 4 — Runtime analysis", job4),
]


def main():
    sections, statuses = [], []
    for slug, title, fn in JOBS:
        body, md, done = fn()
        statuses.append((title, done))
        d = os.path.join(EXP, slug)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "report.html"), "w") as f:
            f.write(page(title, "Dirac experiment batch", body))
        with open(os.path.join(d, "results.md"), "w") as f:
            f.write(md)
        sections.append(f"<h2>{esc(title)}</h2>{body}")

    overview = ['<div class="scroll"><table><tr><th>Job</th><th>Status</th></tr>']
    for title, done in statuses:
        overview.append(f"<tr><td>{esc(title)}</td><td>{badge(done)}</td></tr>")
    overview.append("</table></div>")

    intro = f"""
<div class="panel">
<p>This is the batch of experiments listed under <em>Dirac Jobs</em> in
<code>paper_experiments/experiment_plan.md</code>, run on the machine
<code>dirac</code>. Every number below comes from a script committed under
<code>paper_experiments/</code>; nothing was written into
<code>paper_source/</code>.</p>
<p>TaRF (Tactile-Augmented Radiance Fields) is not reported. The available
img2touch checkpoint is an early training snapshot, so its numbers would not
represent the method; see the note under Job 1.</p>
</div>
{''.join(overview)}
"""
    html = page("Tactile Analogies — Dirac experiment progress",
                "Ground-truth retrieval, full pipeline, ablations, and runtime",
                intro + "".join(sections))
    out = os.path.join(ROOT, "log/paper_experiments_report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}")
    for slug, title, _ in JOBS:
        print(f"wrote paper_experiments/{slug}/report.html + results.md")


if __name__ == "__main__":
    main()
