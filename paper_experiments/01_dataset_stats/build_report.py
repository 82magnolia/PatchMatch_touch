"""Turn stats.json into an HTML report and a Markdown fact sheet."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/junhokim/Projects/PatchMatch_gpu"
S = json.load(open(os.path.join(HERE, "stats.json")))
D = S["datasets"]

# Rendering parameters, read off the generation scripts in
# train_refine_scripts/gen_contact_*_tactile_normal_pseudo_mini/_run.sh
GEN = {
    "ref": {
        "script": "train_refine_scripts/gen_contact_ref_tactile_normal_pseudo_mini/",
        "contact_points": "picked_points_fps.ply (farthest-point sampled)",
        "theta_jitter": "none (0 deg)",
    },
    "query": {
        "script": "train_refine_scripts/gen_contact_query_tactile_normal_pseudo_mini/",
        "contact_points": "picked_points_query.ply",
        "theta_jitter": "+/- 15 deg (0.2618 rad)",
    },
    "raw_eval": {
        "script": "train_refine_scripts/gen_contact_raw_eval_tactile_normal_pseudo_mini/",
        "contact_points": "objfolder_pts_real_eval/&lt;id&gt;/picked_points.ply (all touches pooled per object)",
        "theta_jitter": "+/- 30 deg (0.5236 rad)",
    },
}
LABEL = {
    "ref": "Reference touches (ground-truth-retrieval benchmark)",
    "query": "Query touches (ground-truth-retrieval benchmark)",
    "raw_eval": "Full-pipeline benchmark",
}
SCALE_ORDER = [("100", "1x"), ("50", "2x"), ("25", "4x")]
MODALITIES = ["color", "normal", "height", "curvature", "shapeindex"]
MOD_PLAIN = {
    "color": "RGB colour of the object surface",
    "normal": "surface normal direction",
    "height": "height (distance from the sensor plane)",
    "curvature": "curvature (how sharply the surface bends)",
    "shapeindex": "shape index (a curvature-derived surface-type descriptor)",
}


def gb(x):
    return f"{x / 1e9:.2f} GB"


def kv_rows(pairs):
    return "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in pairs)


def dataset_block(key):
    d = D[key]
    g = GEN[key]
    tv = d["tactile_video"]
    tpo = d["touches_per_object"]
    if tpo["min"] == tpo["max"]:
        touch_str = f"{tpo['min']} per object (fixed)"
    else:
        touch_str = (f"{tpo['mean']:.1f} on average per object "
                     f"(min {tpo['min']}, max {tpo['max']})")
    rows = kv_rows([
        ("Folder", f"<code>Taxim/results/{d['rel_path']}</code>"),
        ("Generation script", f"<code>{g['script']}</code>"),
        ("Objects", f"{d['n_objects']} (ObjectFolder ids {d['obj_id_min']}&ndash;{d['obj_id_max']})"),
        ("Touch locations", touch_str),
        ("Total touch locations", f"<b>{d['total_touches']}</b>"),
        ("Contact points from", g["contact_points"]),
        ("In-plane rotation jitter", g["theta_jitter"]),
        ("Tactile normal video", f"{tv['frames_mean']:.0f} frames, "
                                 f"{tv['height']}&times;{tv['width']} pixels, {tv['fps']:.0f} fps"),
        ("Total video frames", f"{d['total_touches'] * int(tv['frames_mean']):,}"),
        ("Videos per touch", ", ".join(f"<code>{k}</code>" for k in d["video_kinds"])),
        ("Disk footprint", gb(d["disk_bytes"])),
    ])
    return f"""
<h3>{LABEL[key]}</h3>
<table class="kv">{rows}</table>
"""


def render_table():
    head = "".join(f"<th>{x} ({s})</th>" for s, x in SCALE_ORDER)
    body = ""
    for mod in MODALITIES:
        cells = ""
        for s, _ in SCALE_ORDER:
            tot = sum(D[k]["modality_scale_counts"].get(f"{mod}@scale{s}", 0) for k in D)
            cells += f"<td>{tot:,}</td>"
        body += (f"<tr><th>{mod}<div class='sub'>{MOD_PLAIN[mod]}</div></th>{cells}</tr>")
    return f"""
<table class="grid"><thead><tr><th>Rendered map</th>{head}</tr></thead>
<tbody>{body}</tbody></table>
"""


totals_touches = sum(D[k]["total_touches"] for k in D)
totals_frames = sum(D[k]["total_touches"] * int(D[k]["tactile_video"]["frames_mean"]) for k in D)
totals_bytes = sum(D[k]["disk_bytes"] for k in D)
n_renders = totals_touches * len(MODALITIES) * 3

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Job 1 &mdash; Dataset generation statistics</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      max-width:960px;margin:2rem auto;padding:0 1.2rem;line-height:1.6;color:#1b1b1f}}
 h1{{border-bottom:2px solid #333;padding-bottom:.4rem}}
 h2{{margin-top:2.4rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 h3{{margin-top:1.6rem;color:#333}}
 table{{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.94rem}}
 th,td{{border:1px solid #ddd;padding:.45rem .7rem;text-align:left;vertical-align:top}}
 table.kv th{{width:230px;background:#f7f7f9;font-weight:600}}
 table.grid thead th{{background:#f0f2f5}}
 table.grid td{{text-align:right}}
 .sub{{font-weight:400;color:#666;font-size:.82rem}}
 code{{background:#f4f4f6;padding:.1rem .3rem;border-radius:3px;font-size:.88em}}
 .big{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}
 .big div{{flex:1 1 150px;background:#f7f7f9;border:1px solid #e3e3e8;border-radius:8px;
           padding:.8rem 1rem;text-align:center}}
 .big b{{display:block;font-size:1.5rem;color:#1a4f8a}}
 .note{{background:#fffbe6;border-left:4px solid #e0b400;padding:.7rem 1rem;margin:1rem 0}}
</style></head><body>
<h1>Job 1 &mdash; Dataset generation statistics</h1>
<p>This job counts what was actually generated for the tactile-analogies benchmark:
how many objects, how many touch locations, how long each tactile video is, and which
extra renderings accompany every touch. Nothing is trained or predicted here &mdash; it is a
pure inventory of the data on disk, so that the paper can state the benchmark size exactly.</p>

<div class="big">
 <div><b>{S.get('objectfolder_meshes', '?')}</b>ObjectFolder object meshes</div>
 <div><b>{totals_touches:,}</b>touch locations in total</div>
 <div><b>{totals_frames:,}</b>tactile video frames</div>
 <div><b>{n_renders:,}</b>accompanying rendered maps</div>
 <div><b>{gb(totals_bytes)}</b>on disk</div>
</div>

<h2>How a touch is simulated</h2>
<p>Every touch is produced by the tactile simulator <b>Taxim</b> pressing a virtual
GelSight-style sensor into an ObjectFolder mesh. The settings are identical across all
three datasets:</p>
<table class="kv">{kv_rows([
    ("Sensor calibration", "<code>Taxim/calibs/gelsight_pseudo_mini</code> (a scaled-down GelSight Mini)"),
    ("Sensor resolution", "240 &times; 320 pixels"),
    ("Press motion", "<code>back_forth_press</code> &mdash; the sensor presses in and then withdraws"),
    ("Press depth schedule", "0 &rarr; 10 simulator depth units, sampled at 50 steps"),
    ("Video produced", "50 frames at 5 fps (one frame per depth step)"),
    ("Recorded modality", "<code>tactile_normal</code> &mdash; the gel surface normal, colour-coded per pixel"),
])}</table>
<p>Alongside the tactile video, each touch also stores a <code>mask</code> video (which
pixels are in contact) and a <code>render_mask</code> video (which pixels the renderer
could actually cover).</p>

<h2>The three datasets</h2>
{dataset_block('ref')}
{dataset_block('query')}
{dataset_block('raw_eval')}

<div class="note">
The ground-truth-retrieval benchmark is a <b>paired</b> dataset: for the same object, the
same touch index <i>i</i> exists in both the reference set and the query set, so the correct
reference for every query is known in advance. The full-pipeline benchmark instead pools all
touches on an object into one folder, so that a subset can be held out as queries and the
retrieval step has to find the reference by itself.
</div>

<h2>Renderings that accompany every touch</h2>
<p>At each touch location the object surface is also rendered from the sensor's own
viewpoint, at three different fields of view. "1x" means the rendered patch covers exactly
the sensor footprint; "2x" and "4x" cover twice and four times that area, letting the
matching step see more surrounding context. Every rendering is 240 &times; 320 pixels
regardless of the field of view. Counts below are summed over all three datasets.</p>
{render_table()}
<p>The <code>normal</code> and <code>height</code> maps are additionally stored as raw
floating-point arrays (<code>.npz</code>), not just as JPEG previews, so they can be used
numerically: normals are {D['ref']['npz_shapes'].get('normal@scale100')} and height maps are
{D['ref']['npz_shapes'].get('height@scale100')} per touch and scale.</p>

<h2>Train / evaluation split</h2>
<p>The ground-truth-retrieval benchmark is split by object id: objects 1&ndash;950 are used
for training the refinement network and objects <b>951&ndash;1000</b> (50 objects,
{50 * 8} touch locations) are held out for evaluation. This is the split hard-coded in
<code>rebot_net/eval.py</code> and it is the split all reported numbers use.</p>

<h2>Files</h2>
<ul>
 <li><code>paper_experiments/01_dataset_stats/collect_stats.py</code> &mdash; the scanner</li>
 <li><code>paper_experiments/01_dataset_stats/stats.json</code> &mdash; raw numbers</li>
 <li><code>paper_experiments/01_dataset_stats/results.md</code> &mdash; numbers formatted for the paper</li>
</ul>
</body></html>
"""

MD = f"""# Job 1 — Dataset generation statistics

Numbers below are measured directly from the rendered data on disk
(`paper_experiments/01_dataset_stats/collect_stats.py`). Ready to be quoted in
Section "Experiments → Benchmark".

## Headline numbers

| Quantity | Value |
|---|---|
| ObjectFolder meshes used | {S.get('objectfolder_meshes')} |
| Touch locations, all datasets | {totals_touches:,} |
| Tactile video frames, all datasets | {totals_frames:,} |
| Accompanying rendered maps | {n_renders:,} |
| Total disk footprint | {gb(totals_bytes)} |

## Simulation settings (identical for all datasets)

| Setting | Value |
|---|---|
| Simulator | Taxim |
| Sensor calibration | `gelsight_pseudo_mini` |
| Sensor resolution | 240 x 320 |
| Press motion | `back_forth_press` (press in, withdraw) |
| Depth schedule | 0 -> 10 depth units over 50 steps |
| Video | 50 frames @ 5 fps |
| Modality recorded | `tactile_normal` (gel surface normal) |

## Per-dataset

| Dataset | Folder | Objects | Touches / object | Total touches | Video frames | Rotation jitter | Disk |
|---|---|---|---|---|---|---|---|
| Reference (GT-retrieval) | `gen_contact_full_tactile_normal_pseudo_mini` | {D['ref']['n_objects']} | {D['ref']['touches_per_object']['min']} (fixed) | {D['ref']['total_touches']:,} | {D['ref']['total_touches'] * 50:,} | none | {gb(D['ref']['disk_bytes'])} |
| Query (GT-retrieval) | `gen_contact_full_query_tactile_normal_pseudo_mini` | {D['query']['n_objects']} | {D['query']['touches_per_object']['min']} (fixed) | {D['query']['total_touches']:,} | {D['query']['total_touches'] * 50:,} | +/- 15 deg | {gb(D['query']['disk_bytes'])} |
| Full pipeline | `gen_contact_raw_eval_tactile_normal_pseudo_mini` | {D['raw_eval']['n_objects']} | {D['raw_eval']['touches_per_object']['mean']:.1f} (min {D['raw_eval']['touches_per_object']['min']}, max {D['raw_eval']['touches_per_object']['max']}) | {D['raw_eval']['total_touches']:,} | {D['raw_eval']['total_touches'] * 50:,} | +/- 30 deg | {gb(D['raw_eval']['disk_bytes'])} |

## Renderings per touch

Each touch also carries surface renderings from the sensor viewpoint at three fields of
view — 1x (`scale100`, exactly the sensor footprint), 2x (`scale50`), 4x (`scale25`) —
each 240 x 320 pixels. Modalities: RGB colour, surface normal, height, curvature, shape
index. That is 5 modalities x 3 scales = 15 renderings per touch location.

`normal` and `height` are also stored as float arrays: normal {D['ref']['npz_shapes'].get('normal@scale100')},
height {D['ref']['npz_shapes'].get('height@scale100')}.

## Split

Objects 1–950 train, objects 951–1000 (50 objects, 400 touch locations) evaluation.
Hard-coded in `rebot_net/eval.py` (`all_ids[950:]`).

## Sentence-ready summary

> We build our benchmark on all {S.get('objectfolder_meshes')} ObjectFolder objects. For the
> ground-truth-retrieval benchmark we simulate 8 reference and 8 query touches per object
> (16,000 touch locations, 800,000 tactile frames); for the full-pipeline benchmark we
> simulate on average {D['raw_eval']['touches_per_object']['mean']:.1f} touches on each of 100
> objects ({D['raw_eval']['total_touches']:,} touch locations). Every touch is a 50-frame,
> 240 x 320 tactile-normal video at 5 fps produced by Taxim with a scaled GelSight Mini
> calibration, pressing from 0 to 10 depth units and withdrawing, and is accompanied by
> RGB, normal, height, curvature and shape-index renderings of the surface at 1x, 2x and 4x
> the sensor footprint.
"""

open(os.path.join(HERE, "report.html"), "w").write(HTML)
open(os.path.join(HERE, "results.md"), "w").write(MD)
os.makedirs(f"{ROOT}/log/paper_job01_dataset_stats", exist_ok=True)
open(f"{ROOT}/log/paper_job01_dataset_stats/report.html", "w").write(HTML)
print("wrote report.html + results.md")
