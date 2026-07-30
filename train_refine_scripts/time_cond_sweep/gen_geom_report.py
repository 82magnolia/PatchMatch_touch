"""HTML report for the geom-concat sweep (normals concatenated as input).

Usage: python gen_geom_report.py <tag> <epoch>
  reads   log/rebot_eval_S_geomcat_<name>_<tag>/metrics.pkl  (per arm)
          log/rebot_eval_S_geomcat_<name>_<tag>/videos/*_grid.mp4
  writes  log/tactile_normal_geomcat_<tag>_report.html
"""
import base64, html, os, pickle, sys
import cv2

ROOT = "/data1/junhokim/Projects/PatchMatch_touch"

ARMS = [
    ('none',  'Normals concat — no time',
     'Query normal render concatenated as 3 aligned input channels (FiLM off); no timestamp.'),
    ('film',  'Normals concat + sinusoidal FiLM',
     'Same normal concatenation, plus the timestamp injected as sinusoidal → per-stage FiLM.'),
    ('token', 'Normals concat + sinusoidal token',
     'Same normal concatenation, plus the timestamp injected as sinusoidal → bottleneck-token bias.'),
]
COLORS = {'none': '#8f8d84', 'film': '#1f6fd6', 'token': '#e08600'}
TOUCHES = [(951, 2), (963, 4), (975, 1), (988, 6), (1000, 3), (955, 0), (992, 5)]
METRIC_META = {'MSE': 'low', 'PSNR': 'high', 'SSIM': 'high', 'LPIPS': 'low'}

# For context: the winning arms from the first sweep (FiLM-normal, not concat),
# so the reader can see whether concatenation beats FiLM. Filled if available.
FILM_REF = {  # name in first sweep -> (label, eval tag/dir)
    'control (FiLM-normal, no time)': 'none',
    'FiLM-normal + sinusoidal token': 'token',
}


def eval_dir(name, tag):
    return f"{ROOT}/log/rebot_eval_S_geomcat_{name}_{tag}"


def load_avg(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)['average']


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


def png_tag(bgr):
    ok, buf = cv2.imencode('.png', bgr)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.tobytes()).decode("ascii")}" alt="">'


def main():
    tag, epoch = sys.argv[1], sys.argv[2]
    metrics = {n: load_avg(os.path.join(eval_dir(n, tag), 'metrics.pkl')) for n, _, _ in ARMS}
    present = [(n, lbl, d) for n, lbl, d in ARMS if metrics[n] is not None]

    # reference row: first sweep's FiLM-normal + token at the same interim epoch,
    # if that eval exists (tag 'interim' from the first-sweep report).
    ref_rows = []
    for reflbl, refname in [('FiLM-normal, no time (1st sweep, ep~9)', 'none'),
                            ('FiLM-normal + token (1st sweep, ep~9)', 'token')]:
        avg = load_avg(f"{ROOT}/log/rebot_eval_S_time_{refname}_interim/metrics.pkl")
        if avg:
            ref_rows.append((reflbl, avg))

    best = {}
    for m, better in METRIC_META.items():
        vals = [(metrics[n][m], n) for n, _, _ in present]
        best[m] = (min if better == 'low' else max)(vals)[1] if vals else None

    def cells(mv, name=None):
        out = []
        for m in ['MSE', 'PSNR', 'SSIM', 'LPIPS']:
            fmt = '{:.5f}' if m == 'MSE' else ('{:.2f}' if m == 'PSNR' else '{:.4f}')
            cls = ' class="best"' if (name and best[m] == name) else ''
            out.append(f'<td{cls}>{fmt.format(mv[m])}</td>')
        return ''.join(out)

    rows = ''.join(
        f'<tr><td class="arm"><span class="sw" style="background:{COLORS[n]}"></span>{html.escape(lbl)}</td>{cells(metrics[n], n)}</tr>'
        for n, lbl, _ in present)
    ref_html = ''.join(
        f'<tr class="ref"><td class="arm">{html.escape(lbl)}</td>{cells(avg)}</tr>'
        for lbl, avg in ref_rows)
    mech_rows = ''.join(
        f'<tr><td class="arm"><span class="sw" style="background:{COLORS[n]}"></span>{html.escape(lbl)}</td>'
        f'<td class="desc">{html.escape(d)}</td></tr>' for n, lbl, d in present)

    sections = []
    for obj, pair in TOUCHES:
        ctx_html, preds_html, ctx_done = '', '', False
        for n, lbl, _ in present:
            grid = os.path.join(eval_dir(n, tag), 'videos', f'{obj}_{pair}_grid.mp4')
            if not os.path.exists(grid):
                continue
            fr = read_mid_frame(grid)
            if fr is None:
                continue
            if not ctx_done:
                ctx_html = ''.join(
                    f'<figure>{png_tag(quad(fr, q))}<figcaption>{cap}</figcaption></figure>'
                    for q, cap in [('tl', 'Reference (donor)'), ('bl', 'Transferred (input)'),
                                   ('tr', 'Ground truth (target)')])
                ctx_done = True
            preds_html += (f'<figure>{png_tag(quad(fr, "br"))}<figcaption>'
                           f'<span class="dot" style="background:{COLORS[n]}"></span>{html.escape(lbl)}</figcaption></figure>')
        if ctx_done:
            sections.append(f'<section class="touch"><h3>Object {obj}, contact {pair}</h3>'
                            f'<div class="ctx">{ctx_html}</div><div class="preds">{preds_html}</div></section>')

    out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Normals-as-input sweep — epoch {epoch}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
 :root{{color-scheme:light;--bg:#fff;--s2:#f3f2ef;--bd:#e3e1da;--tx:#0b0b0b;--t2:#52514e;--tm:#85837c;--best:#e7f4e7;}}
 @media(prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{color-scheme:dark;--bg:#121211;--s2:#212120;--bd:#33322f;--tx:#fff;--t2:#c3c2b7;--tm:#8f8d84;--best:#173417;}}}}
 :root[data-theme=dark]{{color-scheme:dark;--bg:#121211;--s2:#212120;--bd:#33322f;--tx:#fff;--t2:#c3c2b7;--tm:#8f8d84;--best:#173417;}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 .wrap{{max-width:820px;margin:0 auto;padding:40px 24px 80px}}
 h1{{font-size:23px;margin:0 0 4px}} .meta{{color:var(--tm);font-size:13px;margin-bottom:22px}}
 h2{{font-size:15px;margin:34px 0 10px;text-transform:uppercase;letter-spacing:.04em;color:var(--t2)}}
 .callout{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0 8px;font-size:14px;color:var(--t2)}} .callout b{{color:var(--tx)}}
 table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 8px}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--bd);white-space:nowrap}}
 th{{color:var(--tm);font-weight:600;font-size:12px;text-transform:uppercase}}
 td.best{{background:var(--best);font-weight:700}} td.arm{{white-space:normal}} td.desc{{white-space:normal;color:var(--t2);font-size:13px}}
 tr.ref td{{color:var(--tm);font-style:italic}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:middle}}
 .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}}
 .touch{{margin:26px 0;padding-top:22px;border-top:1px solid var(--bd)}}
 h3{{font-size:13px;color:var(--tm);font-weight:600;margin:0 0 12px;text-transform:uppercase;letter-spacing:.03em}}
 .ctx{{display:flex;gap:10px;margin-bottom:12px}} .preds{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
 @media(max-width:640px){{.ctx{{flex-wrap:wrap}}}}
 figure{{margin:0;flex:1;min-width:0}} img{{width:100%;height:auto;display:block;border-radius:6px;border:1px solid var(--bd);background:var(--s2)}}
 figcaption{{font-size:11px;color:var(--t2);margin-top:5px;text-align:center}} footer{{margin-top:44px;color:var(--tm);font-size:12px}}
</style></head><body><div class="wrap">
 <h1>Feeding the query normals by concatenation instead of FiLM</h1>
 <div class="meta">rebot_S &middot; tactile-normal domain &middot; superpoint_superglue transfer &middot; base recipe fixed
 (charbonnier + zero-init + delta penalty, bottleneck 24) &middot; snapshot at <b>epoch {epoch}/20</b></div>
 <div class="callout"><b>What changed.</b> Instead of injecting the query surface-normal render globally via FiLM,
 we <b>concatenate it as 3 aligned input channels</b> (broadcast to both frames) — the same way the render_mask was
 concatenated — and turn FiLM off. On top of that we test three time settings: none, sinusoidal&rarr;FiLM, and
 sinusoidal&rarr;token. Lower MSE/LPIPS and higher PSNR/SSIM are better; best value per column is highlighted.
 The italic rows are the first sweep's FiLM-normal results at a comparable epoch, for reference.</div>
 <h2>How each arm feeds in the geometry + time</h2>
 <table><tbody>{mech_rows}</tbody></table>
 <h2>Test-set metrics (50 held-out objects)</h2>
 <table><thead><tr><th>arm</th><th>MSE&darr;</th><th>PSNR&uarr;</th><th>SSIM&uarr;</th><th>LPIPS&darr;</th></tr></thead>
 <tbody>{rows}{ref_html}</tbody></table>
 <h2>Qualitative comparison (one mid-touch frame per object)</h2>
 <div class="callout" style="margin-top:0">Top row: shared context (reference / transferred input / ground truth).
 Bottom row: each arm's output, in table order.</div>
 {''.join(sections)}
 <footer>Generated from per-arm eval at epoch {epoch} &middot; tag "{tag}" &middot; log/tactile_normal_geomcat_{tag}_report.html</footer>
</div></body></html>"""
    out_path = f"{ROOT}/log/tactile_normal_geomcat_{tag}_report.html"
    with open(out_path, 'w') as f:
        f.write(out)
    print("written:", out_path, "| arms:", ",".join(n for n, _, _ in present))


if __name__ == '__main__':
    main()
