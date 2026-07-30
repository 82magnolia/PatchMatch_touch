"""Build an HTML report for the temporal-conditioning sweep.

Usage: python gen_time_report.py <tag> <epoch>
  reads   log/rebot_eval_S_time_<name>_<tag>/metrics.pkl   (per arm)
          log/rebot_eval_S_time_<name>_<tag>/videos/*_grid.mp4  (qualitative)
  writes  log/tactile_normal_time_cond_<tag>_report.html
"""
import base64, html, os, pickle, sys
import cv2

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"

# name, human label, one-line description of the mechanism
ARMS = [
    ('none',      'Control — no time',        'Geometry FiLM only; the model never sees which frame it is enhancing.'),
    ('film',      'Sinusoidal → FiLM',   'Timestamp → sinusoidal embedding → MLP → per-stage scale/shift (the Stable-Diffusion recipe).'),
    ('token',     'Sinusoidal → token',  'Same embedding projected to one bias vector added to every bottleneck token.'),
    ('filmtoken', 'FiLM + token',             'Both injection points at once (time added at more layers).'),
    ('concat',    'Concat time channel',      'Timestamp broadcast as one extra constant input channel (coord-conv style).'),
]
# distinct, high-contrast swatch hues (identity only; metrics live in a table)
COLORS = {'none': '#8f8d84', 'film': '#1f6fd6', 'token': '#e08600',
          'filmtoken': '#008300', 'concat': '#c2255c'}
TOUCHES = [(951, 2), (963, 4), (975, 1), (988, 6), (1000, 3), (955, 0), (992, 5)]
METRIC_META = {'MSE': 'low', 'PSNR': 'high', 'SSIM': 'high', 'LPIPS': 'low'}


def eval_dir(name, tag):
    return f"{ROOT}/log/rebot_eval_S_time_{name}_{tag}"


def load_metrics(name, tag):
    p = os.path.join(eval_dir(name, tag), 'metrics.pkl')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)['average']


def b64img(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def read_mid_frame(path, frac=0.5):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, int(n * frac)))
    ret, frame = cap.read(); cap.release()
    return frame if ret else None


def quad(frame, q):
    h, w = frame.shape[0] // 2, frame.shape[1] // 2
    return {'tl': frame[:h, :w], 'tr': frame[:h, w:],
            'bl': frame[h:, :w], 'br': frame[h:, w:]}[q]


def png_tag_from_bgr(bgr):
    ok, buf = cv2.imencode('.png', bgr)
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'<img src="data:image/png;base64,{b64}" alt="">'


def main():
    tag, epoch = sys.argv[1], sys.argv[2]
    metrics = {name: load_metrics(name, tag) for name, _, _ in ARMS}
    present = [(n, lbl, d) for (n, lbl, d) in ARMS for _ in [0]
               if metrics[n] is not None for lbl2 in [lbl] for d2 in [d]] \
        if False else [(n, lbl, d) for (n, lbl, d) in ARMS if metrics[n] is not None]

    # best per metric (for bolding)
    best = {}
    for m, better in METRIC_META.items():
        vals = [(metrics[n][m], n) for n, _, _ in present]
        best[m] = (min if better == 'low' else max)(vals)[1]

    # metrics table
    rows = []
    for name, label, desc in present:
        mv = metrics[name]
        cells = []
        for m in ['MSE', 'PSNR', 'SSIM', 'LPIPS']:
            fmt = '{:.5f}' if m == 'MSE' else ('{:.2f}' if m == 'PSNR' else '{:.4f}')
            val = fmt.format(mv[m])
            cls = ' class="best"' if best[m] == name else ''
            cells.append(f'<td{cls}>{val}</td>')
        rows.append(
            f'<tr><td class="arm"><span class="sw" style="background:{COLORS[name]}"></span>'
            f'{html.escape(label)}</td>{"".join(cells)}</tr>')
    table = '\n'.join(rows)

    mech_rows = '\n'.join(
        f'<tr><td class="arm"><span class="sw" style="background:{COLORS[n]}"></span>{html.escape(lbl)}</td>'
        f'<td class="desc">{html.escape(d)}</td></tr>' for n, lbl, d in present)

    # qualitative sections
    sections = []
    for obj, pair in TOUCHES:
        # context (reference / transferred / gt) from any present arm's grid
        ctx_html, preds_html = '', ''
        ctx_done = False
        for name, label, _ in present:
            grid = os.path.join(eval_dir(name, tag), 'videos', f'{obj}_{pair}_grid.mp4')
            if not os.path.exists(grid):
                continue
            frame = read_mid_frame(grid)
            if frame is None:
                continue
            if not ctx_done:
                ctx_html = ''.join(
                    f'<figure>{png_tag_from_bgr(quad(frame, q))}<figcaption>{cap}</figcaption></figure>'
                    for q, cap in [('tl', 'Reference (donor)'),
                                   ('bl', 'Transferred (input)'),
                                   ('tr', 'Ground truth (target)')])
                ctx_done = True
            preds_html += (f'<figure>{png_tag_from_bgr(quad(frame, "br"))}'
                           f'<figcaption><span class="dot" style="background:{COLORS[name]}"></span>'
                           f'{html.escape(label)}</figcaption></figure>')
        if not ctx_done:
            continue
        sections.append(
            f'<section class="touch"><h3>Object {obj}, contact {pair}</h3>'
            f'<div class="ctx">{ctx_html}</div><div class="preds">{preds_html}</div></section>')

    out = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Temporal conditioning sweep — epoch {epoch}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{{color-scheme:light;--bg:#fff;--s1:#fcfcfb;--s2:#f3f2ef;--bd:#e3e1da;--tx:#0b0b0b;--t2:#52514e;--tm:#85837c;--best:#e7f4e7;}}
 @media (prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{color-scheme:dark;--bg:#121211;--s1:#1a1a19;--s2:#212120;--bd:#33322f;--tx:#fff;--t2:#c3c2b7;--tm:#8f8d84;--best:#173417;}}}}
 :root[data-theme=dark]{{color-scheme:dark;--bg:#121211;--s1:#1a1a19;--s2:#212120;--bd:#33322f;--tx:#fff;--t2:#c3c2b7;--tm:#8f8d84;--best:#173417;}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 .wrap{{max-width:820px;margin:0 auto;padding:40px 24px 80px}}
 h1{{font-size:23px;margin:0 0 4px}} .meta{{color:var(--tm);font-size:13px;margin-bottom:22px}}
 h2{{font-size:15px;margin:34px 0 10px;text-transform:uppercase;letter-spacing:.04em;color:var(--t2)}}
 .callout{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0 8px;font-size:14px;color:var(--t2)}}
 .callout b{{color:var(--tx)}}
 table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 8px}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--bd);white-space:nowrap}}
 th{{color:var(--tm);font-weight:600;font-size:12px;text-transform:uppercase}}
 td.best{{background:var(--best);font-weight:700}}
 td.arm{{white-space:normal}} td.desc{{white-space:normal;color:var(--t2);font-size:13px}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:middle}}
 .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}}
 .touch{{margin:26px 0;padding-top:22px;border-top:1px solid var(--bd)}}
 h3{{font-size:13px;color:var(--tm);font-weight:600;margin:0 0 12px;text-transform:uppercase;letter-spacing:.03em}}
 .ctx{{display:flex;gap:10px;margin-bottom:12px}}
 .preds{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}
 @media(max-width:640px){{.ctx{{flex-wrap:wrap}}.preds{{grid-template-columns:repeat(2,1fr)}}}}
 figure{{margin:0;flex:1;min-width:0}}
 img{{width:100%;height:auto;display:block;border-radius:6px;border:1px solid var(--bd);background:var(--s2)}}
 figcaption{{font-size:11px;color:var(--t2);margin-top:5px;text-align:center}}
 footer{{margin-top:44px;color:var(--tm);font-size:12px}}
</style></head><body><div class="wrap">
 <h1>Conditioning on the frame's timestamp</h1>
 <div class="meta">Temporal-conditioning sweep &middot; rebot_S &middot; tactile-normal domain &middot; superpoint_superglue transfer &middot;
 base recipe held fixed (charbonnier + zero-init + delta penalty + FiLM-normal, bottleneck 24) &middot; snapshot at <b>epoch {epoch}/20</b></div>

 <div class="callout">
  <b>The question.</b> Geometry FiLM (surface-normal / curvature) alone was not enough. A touch video runs
  no-press &rarr; contact &rarr; take-off, so we give the network the frame's <b>normalized timestamp</b>
  (frame index divided by length, a number in [0,1]) and test different ways to feed it in. Every arm shares
  the exact same base recipe; the <b>only</b> thing that changes is how time is injected. Lower MSE / LPIPS is
  better; higher PSNR / SSIM is better. Best value in each column is highlighted.
 </div>

 <h2>How each arm feeds in time</h2>
 <table><tbody>{mech_rows}</tbody></table>

 <h2>Test-set metrics (50 held-out objects)</h2>
 <table>
  <thead><tr><th>arm</th><th>MSE&darr;</th><th>PSNR&uarr;</th><th>SSIM&uarr;</th><th>LPIPS&darr;</th></tr></thead>
  <tbody>{table}</tbody>
 </table>

 <h2>Qualitative comparison (one mid-touch frame per object)</h2>
 <div class="callout" style="margin-top:0">Top row is shared context: the reference (where texture is copied
 from), the transferred input the network refines, and the ground-truth target. The bottom row is each arm's
 output, in the same order as the table.</div>
 {''.join(sections)}

 <footer>Generated from per-arm eval at epoch {epoch} &middot; tag "{tag}" &middot;
 log/tactile_normal_time_cond_{tag}_report.html</footer>
</div></body></html>"""

    out_path = f"{ROOT}/log/tactile_normal_time_cond_{tag}_report.html"
    with open(out_path, 'w') as f:
        f.write(out)
    print("written:", out_path, "| arms:", ",".join(n for n, _, _ in present))


if __name__ == '__main__':
    main()
