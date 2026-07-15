"""Builds the self-contained HTML report for compare_photometric_refine.py.

Kept as its own module so a report can be regenerated against an
already-computed <out_dir>/results.pkl (e.g. log/output_photometric_sweep/
<matcher>/results.pkl) without recomputing the sweep:
`python make_report.py <out_dir>`.
"""

import html
import os
import pickle
import sys
from os import path as osp

METRIC_DIRECTIONS = {"MSE": "min", "PSNR": "max", "SSIM": "max", "LPIPS": "min"}
METRICS = list(METRIC_DIRECTIONS.keys())

# dataviz skill's validated categorical palette (first 4 slots -- CVD-safe as
# a set in both light/dark) used as fixed transform_type identity accents.
TRANSFORM_COLORS = {
    "affine":         ("#2a78d6", "#3987e5"),  # blue
    "homography":     ("#008300", "#008300"),  # green
    "rbf_affine":     ("#e87ba4", "#d55181"),  # magenta
    "rbf_homography": ("#eda100", "#c98500"),  # yellow
}
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def combo_key(row):
    return (row["transform_type"], row["refine"])


def group_key(row):
    return (row["scale"], row["modality"])


def group_label(key):
    scale, modality = key
    scale_str = f"{scale:g}" if scale is not None else "base"
    return f"scale={scale_str} / modality={modality}"


def rank_combos(rows):
    """Average, over queries, each combo's per-metric values (ok rows only),
    then rank combos per metric (1 = best) and by mean rank across metrics.
    Returns a list of dicts sorted best-first.
    """
    by_combo = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = combo_key(row)
        by_combo.setdefault(key, {m: [] for m in METRICS})
        for m in METRICS:
            by_combo[key][m].append(row[m])

    combo_means = {}
    for key, metric_lists in by_combo.items():
        combo_means[key] = {m: sum(v) / len(v) for m, v in metric_lists.items() if v}

    per_metric_rank = {m: {} for m in METRICS}
    for m in METRICS:
        keys_with_metric = [k for k in combo_means if m in combo_means[k]]
        reverse = METRIC_DIRECTIONS[m] == "max"
        ordered = sorted(keys_with_metric, key=lambda k: combo_means[k][m], reverse=reverse)
        for rank, k in enumerate(ordered, start=1):
            per_metric_rank[m][k] = rank

    summary = []
    for key, means in combo_means.items():
        ranks = [per_metric_rank[m][key] for m in METRICS if key in per_metric_rank[m]]
        summary.append({
            "transform_type": key[0], "refine": key[1],
            "means": means, "ranks": {m: per_metric_rank[m][key] for m in METRICS if key in per_metric_rank[m]},
            "mean_rank": sum(ranks) / len(ranks) if ranks else float("inf"),
        })
    summary.sort(key=lambda s: s["mean_rank"])
    return summary


def rank_groups(rows):
    """For each (scale, modality) group, rank its own combos (via
    rank_combos) and report the WINNING combo's means -- i.e. "the best
    achievable result at this scale/modality, given the ideal transform_type
    + refine choice". Groups are then ranked against each other the same way
    (per-metric rank + mean rank), so the top of this table answers "which
    scale/modality setting should I actually use".
    """
    groups = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)

    group_winners = {}
    for key, group_rows in groups.items():
        combo_summary = rank_combos(group_rows)
        if combo_summary:
            group_winners[key] = combo_summary[0]

    per_metric_rank = {m: {} for m in METRICS}
    for m in METRICS:
        keys_with_metric = [k for k in group_winners if m in group_winners[k]["means"]]
        reverse = METRIC_DIRECTIONS[m] == "max"
        ordered = sorted(keys_with_metric,
                         key=lambda k: group_winners[k]["means"][m], reverse=reverse)
        for rank, k in enumerate(ordered, start=1):
            per_metric_rank[m][k] = rank

    summary = []
    for key, winner in group_winners.items():
        ranks = [per_metric_rank[m][key] for m in METRICS if key in per_metric_rank[m]]
        summary.append({
            "group_key": key, "winner": winner,
            "ranks": {m: per_metric_rank[m][key] for m in METRICS if key in per_metric_rank[m]},
            "mean_rank": sum(ranks) / len(ranks) if ranks else float("inf"),
        })
    summary.sort(key=lambda s: s["mean_rank"])
    return summary


def render_group_overview_table(group_summary):
    rows_html = []
    for i, s in enumerate(group_summary):
        winner = s["winner"]
        color = TRANSFORM_COLORS.get(winner["transform_type"], ("#898781", "#898781"))[0]
        winner_badge = '<span class="badge badge-good">WINNER</span>' if i == 0 else ""
        cell_parts = []
        for m in METRICS:
            val = winner["means"].get(m)
            if val is None:
                cell_parts.append("<td>&mdash;</td>")
                continue
            rank_html = f' <span class="rank">#{s["ranks"][m]}</span>' if m in s["ranks"] else ""
            cell_parts.append(f"<td>{val:.5f}{rank_html}</td>")
        cells = "".join(cell_parts)
        rows_html.append(
            f'<tr class="{"winner-row" if i == 0 else ""}">'
            f'<td>{html.escape(group_label(s["group_key"]))}</td>'
            f'<td><span class="swatch" style="background:{color}"></span>'
            f'{html.escape(winner["transform_type"])} / {html.escape(winner["refine"])}</td>'
            f'{cells}'
            f'<td>{s["mean_rank"]:.2f}</td>'
            f'<td>{winner_badge}</td>'
            f'</tr>'
        )
    header = "".join(f"<th>{m}</th>" for m in METRICS)
    return f"""
    <table class="summary-table">
      <thead><tr><th>scale / modality</th><th>best combo</th>{header}<th>mean rank</th><th></th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    """


def best_per_column(rows_for_query):
    """metric -> combo_key of the best value among ok rows for one query."""
    best = {}
    for m in METRICS:
        ok_rows = [r for r in rows_for_query if r["status"] == "ok" and r[m] is not None]
        if not ok_rows:
            continue
        reverse = METRIC_DIRECTIONS[m] == "max"
        best_row = sorted(ok_rows, key=lambda r: r[m], reverse=reverse)[0]
        best[m] = combo_key(best_row)
    return best


def rel(path, out_dir):
    return osp.relpath(path, out_dir) if path else None


def render_summary_table(summary):
    rows_html = []
    for i, s in enumerate(summary):
        color = TRANSFORM_COLORS.get(s["transform_type"], ("#898781", "#898781"))[0]
        winner_badge = (
            '<span class="badge badge-good">WINNER</span>' if i == 0 else ""
        )
        cell_parts = []
        for m in METRICS:
            val = s["means"].get(m)
            if val is None:
                cell_parts.append("<td>&mdash;</td>")
                continue
            rank_html = f' <span class="rank">#{s["ranks"][m]}</span>' if m in s["ranks"] else ""
            cell_parts.append(f"<td>{val:.5f}{rank_html}</td>")
        cells = "".join(cell_parts)
        rows_html.append(
            f'<tr class="{"winner-row" if i == 0 else ""}">'
            f'<td><span class="swatch" style="background:{color}"></span>'
            f'{html.escape(s["transform_type"])}</td>'
            f'<td>{html.escape(s["refine"])}</td>'
            f'{cells}'
            f'<td>{s["mean_rank"]:.2f}</td>'
            f'<td>{winner_badge}</td>'
            f'</tr>'
        )
    header = "".join(f"<th>{m}</th>" for m in METRICS)
    return f"""
    <table class="summary-table">
      <thead><tr><th>transform_type</th><th>refine</th>{header}<th>mean rank</th><th></th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    """


def render_query_table(rows_for_query, best):
    rows_html = []
    for r in sorted(rows_for_query, key=lambda r: (r["transform_type"], r["refine"])):
        color = TRANSFORM_COLORS.get(r["transform_type"], ("#898781", "#898781"))[0]
        if r["status"] != "ok":
            cells = f'<td colspan="{len(METRICS)}" class="failed-cell">{html.escape(r["reason"])}</td>'
        else:
            cells = ""
            for m in METRICS:
                is_best = best.get(m) == combo_key(r)
                cls = "best-cell" if is_best else ""
                cells += f'<td class="{cls}">{r[m]:.5f}</td>'
        rows_html.append(
            f'<tr>'
            f'<td><span class="swatch" style="background:{color}"></span>'
            f'{html.escape(r["transform_type"])}</td>'
            f'<td>{html.escape(r["refine"])}</td>'
            f'{cells}'
            f'</tr>'
        )
    header = "".join(f"<th>{m}</th>" for m in METRICS)
    return f"""
    <table class="query-table">
      <thead><tr><th>transform_type</th><th>refine</th>{header}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    """


def render_image_grid(rows_for_query, query_info, best, out_dir):
    cards = []
    for r in sorted(rows_for_query, key=lambda r: (r["transform_type"], r["refine"])):
        color = TRANSFORM_COLORS.get(r["transform_type"], ("#898781", "#898781"))[0]
        if r["status"] != "ok" or not r["warped_png"]:
            cards.append(
                f'<div class="card failed" style="border-color:{CRITICAL}">'
                f'<div class="card-title">{html.escape(r["transform_type"])} / {html.escape(r["refine"])}</div>'
                f'<div class="card-body failed-body">failed</div>'
                f'</div>'
            )
            continue
        is_any_best = any(best.get(m) == combo_key(r) for m in METRICS)
        best_metrics = [m for m in METRICS if best.get(m) == combo_key(r)]
        badge = (
            f'<span class="badge badge-good">best: {", ".join(best_metrics)}</span>'
            if best_metrics else ""
        )
        cards.append(
            f'<div class="card" style="border-color:{color}">'
            f'<div class="card-title">{html.escape(r["transform_type"])} / {html.escape(r["refine"])} {badge}</div>'
            f'<img class="card-img" src="{rel(r["warped_png"], out_dir)}" loading="lazy">'
            f'<div class="card-metrics">MSE {r["MSE"]:.5f} · PSNR {r["PSNR"]:.2f} · '
            f'SSIM {r["SSIM"]:.4f} · LPIPS {r["LPIPS"]:.4f}</div>'
            f'</div>'
        )
    return f"""
    <div class="ref-row">
      <div class="card ref-card">
        <div class="card-title">Query (ground truth)</div>
        <img class="card-img" src="{rel(query_info['query_png'], out_dir)}">
      </div>
      <div class="card ref-card">
        <div class="card-title">Reference (source)</div>
        <img class="card-img" src="{rel(query_info['ref_png'], out_dir)}">
      </div>
    </div>
    <div class="grid">{"".join(cards)}</div>
    """


CSS = """
:root { color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; } }
:root[data-theme="dark"] { color-scheme: dark; }

body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: #f9f9f7; color: #0b0b0b;
  margin: 0; padding: 24px 32px 64px;
}
@media (prefers-color-scheme: dark) {
  body:not([data-theme="light"] body) {}
}
:root:not([data-theme="light"]) body { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) body { background: #0d0d0d; color: #ffffff; }
}
:root[data-theme="dark"] body { background: #0d0d0d; color: #ffffff; }

h1 { font-size: 1.5rem; margin-bottom: 4px; }
h2 { font-size: 1.15rem; margin-top: 40px; border-bottom: 1px solid #e1e0d9; padding-bottom: 6px; }
.subtitle { color: #52514e; margin-top: 0; }

table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: #fcfcfb; }
:root[data-theme="dark"] table, :root:not([data-theme="light"]) table { }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) table { background: #1a1a19; } }
:root[data-theme="dark"] table { background: #1a1a19; }

th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e1e0d9; font-variant-numeric: tabular-nums; font-size: 0.9rem; }
:root[data-theme="dark"] th, :root[data-theme="dark"] td { border-bottom-color: #2c2c2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) th, :root:not([data-theme="light"]) td { border-bottom-color: #2c2c2a; } }
th { color: #52514e; font-weight: 600; }
:root[data-theme="dark"] th { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) th { color: #c3c2b7; } }

.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
.rank { color: #898781; font-size: 0.78rem; }
.best-cell { background: rgba(12,163,12,0.12); font-weight: 700; }
.failed-cell { color: #d03b3b; font-style: italic; }
.winner-row { background: rgba(12,163,12,0.08); }
.badge { font-size: 0.7rem; padding: 2px 6px; border-radius: 10px; margin-left: 6px; }
.badge-good { background: rgba(12,163,12,0.15); color: #0ca30c; }

.ref-row { display: flex; gap: 12px; margin: 12px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.card { border: 2px solid #c3c2b7; border-radius: 8px; overflow: hidden; background: #fcfcfb; }
:root[data-theme="dark"] .card { background: #1a1a19; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card { background: #1a1a19; } }
.ref-card { flex: 1; border-color: #898781; }
.card-title { font-size: 0.8rem; padding: 6px 8px; font-weight: 600; }
.card-img { width: 100%; display: block; aspect-ratio: 1 / 1; object-fit: contain; background: #0d0d0d; }
.card-metrics { font-size: 0.72rem; padding: 4px 8px 8px; color: #52514e; font-variant-numeric: tabular-nums; }
:root[data-theme="dark"] .card-metrics { color: #c3c2b7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .card-metrics { color: #c3c2b7; } }
.card-body.failed-body { padding: 24px 8px; text-align: center; color: #d03b3b; }
.legend { display: flex; gap: 16px; margin: 8px 0 20px; font-size: 0.85rem; flex-wrap: wrap; }
.legend span.swatch { width: 10px; height: 10px; }
"""


def write_report(rows, cases_info, out_dir):
    legend = "".join(
        f'<span><span class="swatch" style="background:{c[0]}"></span>{t}</span>'
        for t, c in TRANSFORM_COLORS.items()
    )

    group_overview = rank_groups(rows)

    groups = {}
    for ci in cases_info:
        groups.setdefault(group_key(ci), []).append(ci)

    group_sections = []
    for key in sorted(groups, key=lambda k: (k[0] if k[0] is not None else -1, k[1])):
        group_cases = groups[key]
        group_rows = [r for r in rows if group_key(r) == key]
        group_summary = rank_combos(group_rows)

        case_sections = []
        for ci in group_cases:
            rows_for_case = [r for r in group_rows if r["case_id"] == ci["case_id"]]
            best = best_per_column(rows_for_case)
            case_sections.append(f"""
            <h3>Query {ci['query_idx']} &rarr; Reference {ci['ref_idx']}</h3>
            {render_query_table(rows_for_case, best)}
            {render_image_grid(rows_for_case, ci, best, out_dir)}
            """)

        group_sections.append(f"""
        <h2>{html.escape(group_label(key))}</h2>
        <p class="subtitle">Ranking averaged across {len(group_cases)}
        quer{"y" if len(group_cases) == 1 else "ies"} at this scale/modality.</p>
        {render_summary_table(group_summary)}
        {"".join(case_sections)}
        """)

    body = f"""
    <h1>Photometric Refinement Comparison</h1>
    <p class="subtitle">transform_type &times; refinement-loss &times; scale &times; modality
    sweep, scored on whole-image (unmasked) MSE / PSNR / SSIM / LPIPS between the warped
    reference static image and the query static image.</p>

    <h2>Scale / modality overview (best achievable combo per setting)</h2>
    {render_group_overview_table(group_overview)}

    <div class="legend">{legend}</div>

    {"".join(group_sections)}
    """

    out_html = f"<!doctype html><html><head><meta charset='utf-8'><title>Photometric Refinement Comparison</title><style>{CSS}</style></head><body>{body}</body></html>"
    report_path = osp.join(out_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(out_html)
    return report_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python make_report.py <out_dir>  "
                 "(e.g. log/output_photometric_sweep/<matcher>)")
    out_dir = sys.argv[1]
    with open(osp.join(out_dir, "results.pkl"), "rb") as f:
        data = pickle.load(f)
    path = write_report(data["rows"], data["cases"], out_dir)
    print(f"Report written to: {path}")
