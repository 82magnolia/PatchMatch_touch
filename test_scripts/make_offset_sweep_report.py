"""Multi-column overlay report comparing offset-estimation variants.

One ROW per example (dataset / session / query), one COLUMN per variant. Each
cell crossfades that variant's transferred video over the query video, and two
global trackbars drive every cell at once:

  frame  -- scrubs all videos to the same normalised position
  blend  -- crossfades transferred over query everywhere

Because all columns of a row show the same query at the same frame, sweeping
blend makes the variants directly comparable: whichever column ghosts least is
the better offset estimate.

Expects the layout the offset sweep writes:
    <root>/<session>/<variant>/{idx}_transferred.mp4
                              /{idx}_query_<video_type>.mp4
                              /decomposition.pkl

Videos are transcoded to H.264 (OpenCV writes mp4v, which browsers cannot
decode) and lazily attached as each row scrolls into view, so a report with a
few hundred clips stays responsive.

Example usage:
    python test_scripts/make_offset_sweep_report.py \
        --run real:log/offset_sweep/real --run sim:log/offset_sweep/sim \
        --out log/offset_sweep_overlay.html
"""

import argparse
import html
import os
import pickle
import re
import subprocess
from os import path as osp

ASSET_DIRNAME = "offset_sweep_assets"

# (dir name, column label, short description) in display order.
VARIANTS = [
    ("none_none",  "i. no offset",      "zero-offset linear warp only"),
    ("med_dinov3", "ii. median dinov3", "median displacement, DINOv3"),
    ("med_disk",   "iii. median disk",  "median displacement, disk+lightglue"),
    ("med_spsg",   "iii. median sp+sg", "median displacement, superpoint+superglue"),
    ("ran_dinov3", "iv. ransac dinov3", "translation RANSAC, DINOv3"),
    ("ran_disk",   "iv. ransac disk",   "translation RANSAC, disk+lightglue"),
    ("ran_spsg",   "iv. ransac sp+sg",  "translation RANSAC, superpoint+superglue"),
]


def _transcode(src, dst):
    """Re-encode to browser-playable H.264; skip if already up to date."""
    if osp.exists(dst) and osp.getmtime(dst) >= osp.getmtime(src):
        return True
    os.makedirs(osp.dirname(dst), exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
            check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ffmpeg failed on {src}: {e}")
        return False


def _info(variant_dir):
    path = osp.join(variant_dir, "decomposition.pkl")
    if not osp.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _rows(group, root):
    """Yield (session, idx, {variant: dir}, {variant: info})."""
    if not osp.isdir(root):
        return
    for session in sorted(os.listdir(root),
                          key=lambda x: int(x) if x.isdigit() else x):
        sdir = osp.join(root, session)
        if not osp.isdir(sdir):
            continue
        dirs = {v: osp.join(sdir, v) for v, _, _ in VARIANTS
                if osp.isdir(osp.join(sdir, v))}
        if not dirs:
            continue
        infos = {v: _info(d) for v, d in dirs.items()}
        idxs = sorted({int(m.group(1))
                       for d in dirs.values() for m in
                       (re.match(r"^(\d+)_transferred\.mp4$", f) for f in os.listdir(d)) if m})
        for idx in idxs:
            yield session, idx, dirs, infos


def _find_query(vdir, idx):
    for vt in ("shadow", "sim"):
        p = osp.join(vdir, f"{idx}_query_{vt}.mp4")
        if osp.exists(p):
            return p
    return None


def _cell(group, session, idx, variant, label, vdir, info, out_dir, uid):
    src = osp.join(vdir, f"{idx}_transferred.mp4")
    if not osp.exists(src):
        return f'<div class="cell empty"><div class="lbl">{html.escape(label)}</div>—</div>'
    web = osp.join(out_dir, ASSET_DIRNAME, f"{group}_{session}_{idx}_{variant}.mp4")
    if not _transcode(src, web):
        return f'<div class="cell empty"><div class="lbl">{html.escape(label)}</div>!</div>'

    d = info.get(idx, {})
    tx, ty = d.get("offset", (float("nan"),) * 2)
    status = d.get("offset_status", "")
    warn = "" if status.startswith("ok") or status.startswith("none") else " warn"
    nin, ntot = d.get("offset_inliers", "?"), d.get("offset_matches", "?")
    sub = (f"({tx:+.1f}, {ty:+.1f}) &middot; {nin}/{ntot} &middot; "
           f"{100 * d.get('valid_fraction', float('nan')):.0f}% in")
    return f"""<div class="cell{warn}">
  <div class="lbl">{html.escape(label)}</div>
  <div class="stack">
    <video class="qv" data-src="{html.escape(osp.relpath(_QUERY_WEB[uid], out_dir))}"
           muted playsinline></video>
    <video class="tv" data-src="{html.escape(osp.relpath(web, out_dir))}"
           muted playsinline style="opacity:.5"></video>
  </div>
  <div class="sub">{sub}</div>
</div>"""


_QUERY_WEB = {}   # uid -> transcoded query path shared across a row's cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL:ROOT")
    ap.add_argument("--out", default="log/offset_sweep_overlay.html")
    args = ap.parse_args()

    out_dir = osp.dirname(osp.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    sections, uid, n_rows = [], 0, 0
    for spec in args.run:
        group, _, root = spec.partition(":")
        body = []
        for session, idx, dirs, infos in _rows(group, root):
            # The query clip is identical across variants; transcode it once.
            qsrc = next((q for v in dirs.values() if (q := _find_query(v, idx))), None)
            if qsrc is None:
                continue
            qweb = osp.join(out_dir, ASSET_DIRNAME, f"{group}_{session}_{idx}_query.mp4")
            if not _transcode(qsrc, qweb):
                continue
            _QUERY_WEB[uid] = qweb

            cells = "".join(
                _cell(group, session, idx, v, label, dirs[v], infos.get(v, {}), out_dir, uid)
                if v in dirs else
                f'<div class="cell empty"><div class="lbl">{html.escape(label)}</div>—</div>'
                for v, label, _ in VARIANTS)
            body.append(f"""<div class="row">
  <h3>{html.escape(group)} / session {html.escape(session)} / query {idx}</h3>
  <div class="cols">{cells}</div>
</div>""")
            uid += 1
            n_rows += 1
        if body:
            sections.append(f"<h2>{html.escape(group)}</h2>" + "".join(body))

    legend = "".join(f"<li><b>{html.escape(l)}</b> — {html.escape(d)}</li>"
                     for _, l, d in VARIANTS)

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Offset estimation variants — overlay comparison</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0 auto 4rem; max-width: 1500px;
         padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #8886; padding-bottom: .3rem; }}
  .controls {{ position: sticky; top: 0; z-index: 10; background: Canvas;
              border-bottom: 1px solid #8886; padding: .8rem 0; display: flex;
              gap: 2rem; align-items: center; flex-wrap: wrap; }}
  .controls label {{ font-size: .85rem; }}
  .controls input[type=range] {{ width: 260px; vertical-align: middle; }}
  .row {{ margin: 1.4rem 0; }}
  .row h3 {{ font-size: .95rem; margin: 0 0 .4rem; }}
  .cols {{ display: flex; gap: .5rem; overflow-x: auto; padding-bottom: .4rem; }}
  .cell {{ flex: 0 0 auto; width: 200px; }}
  .cell.warn .lbl {{ color: #b06a00; }}
  .cell.empty {{ opacity: .4; }}
  .lbl {{ font-size: .75rem; font-weight: 600; margin-bottom: .2rem;
         white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .sub {{ font-size: .7rem; opacity: .7; margin-top: .2rem; }}
  .stack {{ position: relative; width: 200px; height: 150px; background: #0002; }}
  .stack video {{ position: absolute; inset: 0; width: 200px; height: 150px; }}
  .intro {{ background: #8881; border-radius: 8px; padding: .8rem 1rem; }}
  .intro ul {{ margin: .4rem 0 0; padding-left: 1.2rem; }}
  .intro li {{ font-size: .85rem; }}
</style>
<h1>Offset estimation variants — overlay comparison</h1>
<div class="intro">
  <p>Warp = <b>offset &compfn; linear</b>. The linear part is identical in every
  column (homography, disk+lightglue, fitted at <code>--match_scale</code>); only
  the offset stage differs. Each cell shows that variant's transferred video
  crossfaded over the query.</p>
  <ul>{legend}</ul>
  <p>Per-cell numbers: estimated <code>(tx, ty)</code> &middot; offset
  inliers/matches &middot; fraction of the NNF landing in bounds.</p>
</div>
<div class="controls">
  <label>frame <input type="range" id="gf" min="0" max="1000" value="0"></label>
  <label>blend <input type="range" id="gb" min="0" max="100" value="50">
         <span id="gbv">50%</span></label>
  <span style="font-size:.8rem;opacity:.7">sliders drive every visible cell</span>
</div>
{"".join(sections) if sections else "<p>No runs found.</p>"}
<script>
const gf = document.getElementById("gf"), gb = document.getElementById("gb");
const gbv = document.getElementById("gbv");
const loaded = new Set();

function seek(v) {{
  if (!v.duration) return;
  v.currentTime = (gf.value / 1000) * v.duration;
}}

// Attach sources only as a row scrolls into view: the full report holds a few
// hundred clips, and eagerly decoding them all is what would make it crawl.
const io = new IntersectionObserver((entries) => {{
  for (const e of entries) {{
    if (!e.isIntersecting || loaded.has(e.target)) continue;
    loaded.add(e.target);
    e.target.querySelectorAll("video[data-src]").forEach(v => {{
      v.src = v.dataset.src;
      v.addEventListener("loadedmetadata", () => {{
        // A video that has never been seeked paints nothing, so nudge it.
        v.currentTime = Math.max(0.01, (gf.value / 1000) * v.duration);
        if (v.classList.contains("tv")) v.style.opacity = gb.value / 100;
      }}, {{ once: true }});
    }});
  }}
}}, {{ rootMargin: "300px" }});
document.querySelectorAll(".row").forEach(r => io.observe(r));

gf.addEventListener("input", () => document.querySelectorAll("video").forEach(seek));
gb.addEventListener("input", () => {{
  document.querySelectorAll("video.tv").forEach(v => v.style.opacity = gb.value / 100);
  gbv.textContent = gb.value + "%";
}});
</script>
"""
    with open(args.out, "w") as f:
        f.write(page)
    print(f"Wrote {args.out} ({n_rows} rows x {len(VARIANTS)} variants)")


if __name__ == "__main__":
    main()
