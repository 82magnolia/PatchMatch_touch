"""Build log/paper_tarf_report.html: our method against the TaRF baseline.

Reads only the per-job results.json files that the two aggregators write, plus
whatever diagnostic images already exist under log/paper_tarf_diagnostic/, so it
is safe to re-run at any time.

  python paper_experiments/gen_tarf_report.py
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_report import CSS, embed_img, esc, load, md_table, metric_table  # noqa: E402

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"
OUT_HTML = os.path.join(ROOT, "log/paper_tarf_report.html")
OUT_MD = os.path.join(ROOT, "paper_experiments/tarf_results.md")

TARF_ROWS = ["TaRF (epoch 5, finetuned)",
             "TaRF (epoch 29, from scratch)",
             "TaRF (epoch 29, finetuned)"]
JOB1_ORDER = ["Tactile Normal Quilting", "ObjectFolder INR"] + TARF_ROWS + [
    "Ours (coarse transfer, normals)", "Ours (refined, normals)"]
JOB2_ORDER = ["Tactile Normal Quilting", "ObjectFolder INR"] + TARF_ROWS + [
    "Ours (coarse transfer, normals)", "Ours (refined, normals)"]

# name -> (checkpoint file, training run, what it is)
PROVENANCE = [
    ("TaRF (epoch 5, finetuned)", "log/tarf_pretrained.ckpt",
     "2026-07-31T10-45-38 ..._upstream_finetune_ref_even_query_odd",
     "Early snapshot (epoch 5 of 30) of the run that starts from the released "
     "TaRF weights. Best validation loss so far at that point: 0.1732."),
    ("TaRF (epoch 29, from scratch)", "log/tarf_pretrained_v2.ckpt",
     "2026-07-31T10-14-41 patchmatch_sim_tactile_normal_ref_even_query_odd",
     "Completed 30-epoch run that deliberately does not import the released "
     "diffusion or conditioning weights. Best validation loss 0.1767 (epoch 10); "
     "0.1795 at the saved epoch."),
    ("TaRF (epoch 29, finetuned)", "log/tarf_pretrained_v3.ckpt",
     "2026-07-31T10-45-38 ..._upstream_finetune_ref_even_query_odd",
     "The same run as the first row, trained to completion. Best validation loss "
     "0.1373 (epoch 26); 0.1411 at the saved epoch &mdash; the lowest of the three."),
]


def rows_job1(d):
    out = []
    for k in JOB1_ORDER:
        t = (d or {}).get("table", {}).get(k, {})
        m = ({kk: t[kk] for kk in ("PSNR", "SSIM", "LPIPS", "MSE")}
             if t.get("n_objects") else None)
        out.append((k, m, t.get("n_objects", 0)))
    return out


def rows_job2(d):
    out = []
    for k in JOB2_ORDER:
        t = (d or {}).get("table", {}).get(k, {})
        out.append((k, t.get("metrics"), t.get("n_objects", 0)))
    return out


def provenance_table():
    out = ['<div class="scroll"><table>',
           "<tr><th>Row in the tables</th><th>Checkpoint file</th>"
           "<th>Training run</th><th>What it is</th></tr>"]
    for name, ckpt, run, what in PROVENANCE:
        out.append(f"<tr><td>{esc(name)}</td><td><code>{esc(ckpt)}</code></td>"
                   f"<td><code>{esc(run)}</code></td><td>{what}</td></tr>")
    out.append("</table></div>")
    return "\n".join(out)


def gap_sentences(rows, label):
    """State the our-method-vs-best-TaRF gap without hard-coding numbers."""
    have = {n: m for n, m, _ in rows if m}
    ours = have.get("Ours (refined, normals)")
    tarfs = {n: m for n, m in have.items() if n.startswith("TaRF")}
    if not ours or not tarfs:
        return "<p>Waiting for both our method and TaRF to finish on this benchmark.</p>"
    best_name = max(tarfs, key=lambda n: tarfs[n]["PSNR"])
    best = tarfs[best_name]
    return (
        f"<p>On the {label}, the strongest TaRF checkpoint is "
        f"<strong>{esc(best_name)}</strong> at {best['PSNR']:.2f} dB PSNR "
        f"(SSIM {best['SSIM']:.4f}, LPIPS {best['LPIPS']:.4f}). Our refined method "
        f"reaches {ours['PSNR']:.2f} dB (SSIM {ours['SSIM']:.4f}, LPIPS "
        f"{ours['LPIPS']:.4f}) &mdash; a gap of "
        f"<strong>{ours['PSNR'] - best['PSNR']:.2f} dB</strong>.</p>")


def diversity_section():
    """Does each method's prediction actually change with the query?"""
    d = load("paper_experiments/job1_gt_retrieval/prediction_diversity.json")
    if not d:
        return ""
    rows = ['<div class="scroll"><table>',
            "<tr><th>Method</th><th>Objects</th><th>How much predictions differ "
            "between objects</th><th>Same figure for the ground truth</th>"
            "<th>Share of the real variation</th></tr>"]
    for name, v in d["methods"].items():
        rows.append(f"<tr><td>{esc(name)}</td><td>{v['n_objects']}</td>"
                    f"<td>{v['spread']:.4f}</td>"
                    f"<td>{v['ground_truth_spread_same_objects']:.4f}</td>"
                    f"<td>{v['percent_of_ground_truth']:.0f}%</td></tr>")
    rows.append("</table></div>")
    return f"""
<h2>Do the predictions respond to the query at all?</h2>
<p>The tables above say how wrong each method is, but not <em>why</em>. This check
asks a simpler question: when you give a method 50 different touch locations, do
you get 50 different pictures back? For one touch index, we take the middle frame
of every object's prediction and measure the average pixel difference between
predictions belonging to <em>different</em> objects. A method that ignores what it
was asked returns the same picture every time, so its number collapses towards
zero. The ground truth's own value is the yardstick: that is how different these
touches genuinely are.</p>
{"".join(rows)}
<p>Our method sits at 94&ndash;96% of the real variation, and quilting at 94%: they
respond to the query about as much as the truth does. The two finetuned TaRF
checkpoints reach only 40&ndash;49%, meaning roughly half the object-to-object
variation is missing &mdash; they lean on a generic average touch. The
from-scratch checkpoint goes the other way at 129%: it varies more than reality,
which is noise rather than signal. ObjectFolder INR is the extreme case at 4%
&mdash; it returns almost the same smooth image for every object, which is a safe
bet under PSNR and explains how it scores well despite the flat predictions
visible in the figures.</p>
"""


def main():
    d1 = load("paper_experiments/job1_gt_retrieval/results.json")
    d2 = load("paper_experiments/job2_full_pipeline/results.json")
    r1, r2 = rows_job1(d1), rows_job2(d2)

    smoke = embed_img("log/paper_tarf_diagnostic/smoke_v1v2v3.png")
    vs_gt = embed_img("log/paper_tarf_diagnostic/tarf_vs_gt.png")
    cands = embed_img("log/paper_tarf_diagnostic/tarf_candidates.png")
    job2v = embed_img("log/paper_tarf_diagnostic/job2_variants.png")

    smoke_fig = (f'<h3>What the three checkpoints actually predict</h3>'
                 f'<p>One object, three touch locations. Columns: the epoch-5 '
                 f'finetuned checkpoint, the epoch-29 from-scratch checkpoint, the '
                 f'epoch-29 finetuned checkpoint, and the ground truth. This panel '
                 f'was captured while diagnosing the black-image bug, so the third '
                 f'column shows the broken output.</p>'
                 f'<img src="{smoke}" alt="three checkpoints compared">' if smoke else "")
    vs_fig = (f'<h3>TaRF against our method</h3>'
              f'<p>Each row is one touch. Columns: the query surface-normal render '
              f'that conditions the prediction, the three TaRF checkpoints, our '
              f'refined prediction, and the ground-truth touch.</p>'
              f'<img src="{vs_gt}" alt="TaRF vs ours">' if vs_gt else "")
    cand_fig = (f'<h3>All eight samples TaRF draws for one query</h3>'
                f'<p>TaRF draws eight candidate images per query and picks one with '
                f'a separate ranking network. If every candidate lacks surface '
                f'detail, the limit is the diffusion model rather than the picking '
                f'step.</p><img src="{cands}" alt="TaRF candidates">' if cands else "")

    body = f"""
<div class="panel">
<p><strong>What this page is.</strong> TaRF (Tactile-Augmented Radiance Fields) is
the diffusion-based baseline: given camera views of a location on an object, it
generates the touch image it expects there. Three trained checkpoints of its
image-to-touch diffusion model are now available, and this page runs all three
against our method on both benchmarks.</p>
<p><strong>How TaRF is run.</strong> For each query it draws 8 candidate images
with 200 DDIM (denoising diffusion implicit model) sampling steps and picks one
using the two released ranking encoders. The result is a single still image,
which is then repeated to the length of the reference video: TaRF has no
mechanism for the frame-to-frame change that a real touch sequence contains,
while our method predicts the whole sequence.</p>
</div>

<h2>The three checkpoints</h2>
{provenance_table()}

<div class="panel">
<p><strong>A silent numerical bug had to be fixed before any of this was
measurable.</strong> TaRF's inference ran the entire model in mixed precision
(16-bit floating point). Inside the encoder that turns the conditioning RGB and
depth images into a conditioning vector, the epoch-29 finetuned checkpoint
produced intermediate values above the largest number 16-bit floating point can
hold (65504). Those became infinity, then not-a-number, the not-a-number survived
all 200 sampling steps, and the final conversion to 8-bit pixels mapped every
not-a-number to 0. The result was a perfectly black image, produced without a
single warning anywhere in the logs.</p>
<p>On one test object that checkpoint scored 3.37 dB PSNR while broken and
<strong>11.29 dB after the fix</strong>. The fix is to run just that conditioning
encoder in full 32-bit precision
(<code>baselines/TaRF/patchmatch_tarf/generator.py</code>); its output is small
(largest magnitude about 18), so the rest of the model can stay in mixed
precision. Measured cost: none (7.1 s versus 7.5 s per query). Every TaRF number
below was produced after this fix.</p>
</div>

<h2>Benchmark 1 &mdash; ground-truth retrieval (50 objects)</h2>
<p>Every method is handed the correct reference touch, so nothing here depends on
retrieval quality. This isolates how well a method transfers a known reference
touch to a new query location.</p>
{metric_table(r1)}
{gap_sentences(r1, "ground-truth retrieval benchmark")}

<h2>Benchmark 2 &mdash; full pipeline (100 objects)</h2>
<p>Retrieval is part of the system under test: the method must first find a useful
reference among the object's other touches, then transfer it. All methods get the
same split and the same retrieval.</p>
{metric_table(r2)}
{gap_sentences(r2, "full-pipeline benchmark")}
<div class="panel">
<p><strong>The from-scratch checkpoint collapses on this benchmark.</strong> It
drops from 10.48 dB on the ground-truth-retrieval benchmark to 6.89 dB here, with
SSIM falling to 0.11. The figure below shows why: its predictions become
salt-and-pepper noise. This is genuine model behaviour and not a pipeline fault
&mdash; the runs completed without a single failure, and no output was blank.
The conditioning views on this benchmark come from a different rendering setup
than the one the checkpoint was trained on, and the checkpoint that never
imported the released TaRF weights has the least to fall back on when its input
drifts out of the distribution it saw.</p>
</div>
{f'<h3>The three checkpoints on the full-pipeline benchmark</h3>'
 f'<p>Three objects, one touch each. Columns: the three checkpoints and the '
 f'ground truth.</p><img src="{job2v}" alt="job2 TaRF variants">' if job2v else ""}

{diversity_section()}

{smoke_fig}
{vs_fig}
{cand_fig}
"""
    html = (f"<title>TaRF baseline vs our method</title>\n<style>{CSS}</style>\n"
            f'<div class="wrap"><h1>TaRF baseline vs our method</h1>'
            f'<p class="sub">All three trained TaRF diffusion checkpoints, on both '
            f'benchmarks, after fixing a 16-bit overflow that silently produced '
            f'black predictions</p>{body}'
            f'<footer>Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on dirac '
            f'&middot; regenerate with <code>python paper_experiments/gen_tarf_report.py</code>'
            f'</footer></div>')
    with open(OUT_HTML, "w") as f:
        f.write(html)

    md = f"""# TaRF baseline vs our method

Three trained TaRF checkpoints, both benchmarks, all run after the float32
conditioning-encoder fix (see below).

## Checkpoints

| Row | File | Training run | Notes |
|---|---|---|---|
""" + "\n".join(
        f"| {n} | `{c}` | `{r}` | {w.replace('&mdash;', '—')} |"
        for n, c, r, w in PROVENANCE) + f"""

## Benchmark 1 — ground-truth retrieval (50 objects, 951–1000)

{md_table(r1)}

## Benchmark 2 — full pipeline (100 objects)

{md_table(r2)}

## Do the predictions respond to the query?

Mean absolute difference between the middle frames of *different objects'*
predictions at one touch index, against the ground truth on the same objects.
A method that ignores its conditioning collapses towards 0%.

""" + (lambda dv: "\n".join(
        ["| Method | Objects | Spread | GT spread | % of ground truth |",
         "|---|---|---|---|---|"] +
        [f"| {n} | {v['n_objects']} | {v['spread']:.4f} | "
         f"{v['ground_truth_spread_same_objects']:.4f} | "
         f"{v['percent_of_ground_truth']:.0f}% |"
         for n, v in dv["methods"].items()])
        if dv else "_(not computed)_")(
        load("paper_experiments/job1_gt_retrieval/prediction_diversity.json")) + f"""

Ours 94–96%, quilting 94%: they vary with the query as much as the truth does.
The finetuned TaRF checkpoints reach 40–49% (half the real variation missing);
the from-scratch one hits 129% (noise, not signal); ObjectFolder INR is 4%, i.e.
almost the same image for every object.

Reproduce: `python paper_experiments/job1_gt_retrieval/prediction_diversity.py`.

## The float32 conditioning fix

TaRF inference ran fully in float16. The conditioning encoder overflowed float16
range (>65504) for the epoch-29 finetuned checkpoint, producing inf → NaN → an
all-black uint8 image with no error raised. Same object: 3.37 dB PSNR broken,
11.29 dB fixed. `baselines/TaRF/patchmatch_tarf/generator.py` now runs
`get_learned_conditioning` outside the autocast block; the UNet and decoder still
use float16. No measurable runtime cost.

Raw runs: `log/paper_job{{1,2}}_baselines/{{tarf,tarf_v2,tarf_v3}}/`.
Pre-fix runs kept for reference: `log/paper_job{{1,2}}_baselines/tarf_fp16cond_old/`.
HTML report: `log/paper_tarf_report.html`.
"""
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(f"wrote {OUT_HTML}\nwrote {OUT_MD}")


if __name__ == "__main__":
    sys.exit(main())
