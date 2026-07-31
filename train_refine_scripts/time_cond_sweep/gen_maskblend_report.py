"""HTML report for the mask-blended geom-concat run (normals concatenated as
input, no temporal conditioning), trained on the superpoint+superglue transfer
regenerated from the mask-blended sim tactile normals.

Usage: python gen_maskblend_report.py <eval_dir> <epoch_label> <out_html>
  reads   <eval_dir>/metrics.pkl
          <eval_dir>/videos/<obj>_<pair>_grid.mp4   (tl=ref, tr=GT, bl=input, br=output)
  writes  <out_html>
"""
import base64, glob, html, os, pickle, sys
import cv2

METRIC_META = {'MSE': 'lower is better', 'PSNR': 'higher is better',
               'SSIM': 'higher is better', 'LPIPS': 'lower is better'}


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


def png_tag(bgr, width=220):
    if bgr is None:
        return "<span style='color:#999'>n/a</span>"
    h, w = bgr.shape[:2]
    scale = width / w
    bgr = cv2.resize(bgr, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', bgr)
    b64 = base64.b64encode(buf).decode()
    return f"<img src='data:image/png;base64,{b64}'>"


def main():
    eval_dir, epoch_label, out_html = sys.argv[1], sys.argv[2], sys.argv[3]
    metrics = None
    mpath = os.path.join(eval_dir, 'metrics.pkl')
    if os.path.exists(mpath):
        with open(mpath, 'rb') as f:
            metrics = pickle.load(f)
    avg = metrics['average'] if metrics else None
    nobj = len(metrics['per_object']) if metrics else 0

    grids = sorted(glob.glob(os.path.join(eval_dir, 'videos', '*_grid.mp4')))[:8]

    metric_rows = ""
    if avg:
        for k in ['MSE', 'PSNR', 'SSIM', 'LPIPS']:
            metric_rows += (f"<tr><td>{k}</td><td>{avg[k]:.4f}</td>"
                            f"<td>{METRIC_META[k]}</td></tr>")

    cards = ""
    for g in grids:
        base = os.path.basename(g)[:-len('_grid.mp4')]
        frame = read_mid_frame(g)
        if frame is None:
            continue
        cards += f"""
        <div class="case">
          <div class="cap">test object/touch {html.escape(base)}</div>
          <div class="triptych">
            <figure>{png_tag(quad(frame,'bl'))}<figcaption>Input (transferred)</figcaption></figure>
            <figure>{png_tag(quad(frame,'br'))}<figcaption>Model output</figcaption></figure>
            <figure>{png_tag(quad(frame,'tr'))}<figcaption>Ground truth (sim)</figcaption></figure>
          </div>
        </div>"""

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Mask-blended normals-concat ReBotNet — epoch {html.escape(str(epoch_label))}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1050px;margin:0 auto;padding:24px 20px 80px;color:#1a1a1a;line-height:1.55}}
 h1{{font-size:24px;margin-bottom:2px}} .sub{{color:#666;margin-top:0}}
 h2{{margin-top:30px;border-bottom:2px solid #eee;padding-bottom:6px}}
 code{{background:#f2f2f4;padding:1px 5px;border-radius:4px;font-size:90%}}
 table{{border-collapse:collapse;margin-top:8px}} td,th{{border:1px solid #ddd;padding:6px 12px;text-align:left}}
 th{{background:#f5f5f7}}
 .case{{margin:16px 0;padding:12px;border:1px solid #e6e6ea;border-radius:10px;background:#fafafb}}
 .cap{{font-size:13px;color:#444;margin-bottom:8px}}
 .triptych{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
 figure{{margin:0;text-align:center}} figure img{{width:100%;border-radius:6px;border:1px solid #ddd;background:#000}}
 figcaption{{font-size:12px;color:#555;margin-top:5px}}
 .note{{background:#fff8e6;border:1px solid #f0e0a8;padding:10px 14px;border-radius:8px}}
</style></head><body>
<h1>Mask-blended normals-concat ReBotNet — initial results</h1>
<p class="sub">Checkpoint at epoch {html.escape(str(epoch_label))} &middot; training still in progress</p>

<h2>What this run is</h2>
<p>
This is a fresh ReBotNet (the tactile-video enhancement/refiner network) trained the same way as
the kept GPU 5&ndash;7 runs &mdash; <b>normal concatenation</b> (the simulated surface-normal render is
concatenated onto the input as extra image channels, flag <code>--geom_concat</code>) with
<b>no temporal conditioning</b> (no timestamp embedding, token, or FiLM; <code>--time_cond none</code>).
The difference is the <b>data</b>: both the training pairs (superpoint+superglue transfer) and the
conditioning normals were regenerated from the new <b>mask-blended</b> simulated tactile normals.
</p>
<div class="note">
FiLM = Feature-wise Linear Modulation (a way of injecting side information into a network).
MSE = Mean Squared Error. PSNR = Peak Signal-to-Noise Ratio. SSIM = Structural Similarity.
LPIPS = Learned Perceptual Image Patch Similarity (a perceptual difference score).
</div>

<h2>Test-set metrics (50 held-out objects, 951&ndash;1000)</h2>
<p>Averaged over {nobj} evaluated test objects. These measure the <b>model output vs. the simulated
ground truth</b>.</p>
<table><tr><th>Metric</th><th>Value</th><th>Meaning</th></tr>{metric_rows}</table>

<h2>Example test touches</h2>
<p>Each row: the degraded <b>input</b> the network receives (the transferred video), the network's
<b>output</b>, and the <b>ground-truth</b> simulated normals it is trying to match. Middle frame of each
touch shown.</p>
{cards or '<p><i>No eval videos found.</i></p>'}
</body></html>"""
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, 'w') as f:
        f.write(doc)
    print("wrote", out_html)


if __name__ == '__main__':
    main()
