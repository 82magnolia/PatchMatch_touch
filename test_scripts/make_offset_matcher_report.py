"""Mid-frame overlay report comparing the OFFSET matcher: disk+lightglue vs DINOv3.

For each query it shows two overlays side by side -- query + transferred, once
with the offset estimated by disk+lightglue and once by DINOv3 -- using the
single MID frame of each clip (a still image, not the video: 135 rows x 4 clips
overwhelms the browser's video-decoder budget and the page stops painting).
A single blend slider per row crossfades the transferred still over the query.

Both columns share the same linear part (homography, disk+lightglue, fit at
--match_scale); only the OFFSET stage's matcher differs, so sweeping blend
isolates which matcher localises the shift better -- whichever ghosts less wins.

Each group pairs a disk root with a dinov3 root, both laid out as the decomposed
transfer writes them:
    <root>/<session>/transfer/{idx}_transferred.mp4
                             /{idx}_query_<video_type>.mp4
                             /decomposition.pkl

Mid frames are extracted to the report's own asset dir with ffmpeg.

Example usage:
    python test_scripts/make_offset_matcher_report.py \
        --group sim:log/offset_matcher_report/disk/sim:log/offset_matcher_report/dinov3/sim \
        --group real:log/offset_matcher_report/disk/real:log/offset_matcher_report/dinov3/real \
        --out log/offset_matcher_overlay.html
"""

import argparse
import html
import os
import pickle
import re
import subprocess
from os import path as osp

ASSET_DIRNAME = "offset_matcher_assets"


def _duration(src):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", src],
            check=True, capture_output=True, text=True).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def _midframe(src, dst):
    """Extract the middle frame of src as a JPG; skip if already up to date."""
    if osp.exists(dst) and osp.getmtime(dst) >= osp.getmtime(src):
        return True
    os.makedirs(osp.dirname(dst), exist_ok=True)
    mid = _duration(src) / 2.0
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.3f}",
             "-i", src, "-frames:v", "1", "-q:v", "3", dst],
            check=True, capture_output=True)
        return osp.exists(dst)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ffmpeg failed on {src}: {e}")
        return False


def _load_info(transfer_dir):
    path = osp.join(transfer_dir, "decomposition.pkl")
    if not osp.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _sessions(root):
    """root/<session>/transfer/{idx}_transferred.mp4 -> {session: (tdir, idxs, info)}."""
    out = {}
    if not osp.isdir(root):
        return out
    for session in sorted(os.listdir(root),
                          key=lambda x: int(x) if x.isdigit() else x):
        tdir = osp.join(root, session, "transfer")
        if not osp.isdir(tdir):
            continue
        idxs = sorted(int(m.group(1))
                      for m in (re.match(r"^(\d+)_transferred\.mp4$", f)
                                for f in os.listdir(tdir)) if m)
        if idxs:
            out[session] = (tdir, idxs, _load_info(tdir))
    return out


def _query(tdir, idx):
    for vt in ("shadow", "sim"):
        p = osp.join(tdir, f"{idx}_query_{vt}.mp4")
        if osp.exists(p):
            return p
    return None


def _sub(info, idx):
    d = info.get(idx, {})
    tx, ty = d.get("offset", (float("nan"),) * 2)
    status = str(d.get("offset_status", "n/a"))
    warn = not (status.startswith("ok") or status.startswith("none"))
    txt = (f"({tx:+.1f}, {ty:+.1f}) &middot; "
           f"{d.get('offset_inliers', '?')}/{d.get('offset_matches', '?')} matches "
           f"&middot; {100 * d.get('valid_fraction', float('nan')):.0f}% in")
    if warn:
        txt += f" &middot; {html.escape(status)}"
    return warn, txt


def _stack(uid, tag, sub, warn, query_img, transferred_img, out_dir):
    rel = lambda p: html.escape(osp.relpath(p, out_dir))
    cls = "tag warn" if warn else "tag"
    return f"""<div class="col">
  <div class="stack">
    <img class="q" src="{rel(query_img)}" loading="lazy" alt="query">
    <img class="t" data-row="{uid}" src="{rel(transferred_img)}" loading="lazy" alt="transferred"
         style="opacity:0.5">
  </div>
  <div class="{cls}">{html.escape(tag)}</div>
  <div class="sub">{sub}</div>
</div>"""


def _card(group, session, idx, d_tdir, d_info, v_tdir, v_info, out_dir, uid):
    q_src = _query(d_tdir, idx)
    d_src = osp.join(d_tdir, f"{idx}_transferred.mp4")
    v_src = osp.join(v_tdir, f"{idx}_transferred.mp4")
    if not (q_src and osp.exists(d_src) and osp.exists(v_src)):
        return ""

    stem = f"{group}_{session}_{idx}"
    q_img = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_query.jpg")
    d_img = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_disk.jpg")
    v_img = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_dinov3.jpg")
    if not (_midframe(q_src, q_img) and _midframe(d_src, d_img) and _midframe(v_src, v_img)):
        return ""

    lin = d_info.get(idx, {})
    head = (f"linear {lin.get('linear_inliers', '?')}/{lin.get('linear_matches', '?')} inliers "
            f"&middot; ratio {lin.get('ratio', '?')}")
    d_warn, d_sub = _sub(d_info, idx)
    v_warn, v_sub = _sub(v_info, idx)

    return f"""
<div class="card">
  <h3>{html.escape(group)} / session {html.escape(session)} / query {idx}</h3>
  <p class="meta">{head} &nbsp;(linear stage identical in both columns)</p>
  <div class="cols">
    {_stack(uid, "offset: disk+lightglue", d_sub, d_warn, q_img, d_img, out_dir)}
    {_stack(uid, "offset: dinov3", v_sub, v_warn, q_img, v_img, out_dir)}
  </div>
  <label>blend <input type="range" class="b" data-row="{uid}" min="0" max="100" value="50">
         <span class="bv" data-row="{uid}">50%</span></label>
</div>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True,
                    metavar="LABEL:DISK_ROOT:DINOV3_ROOT",
                    help="Group label, its disk-offset root, and its dinov3-offset root. "
                         "Repeatable.")
    ap.add_argument("--out", default="log/offset_matcher_overlay.html")
    args = ap.parse_args()

    out_dir = osp.dirname(osp.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    sections, uid, n_cards = [], 0, 0
    for spec in args.group:
        label, d_root, v_root = spec.split(":", 2)
        d_sessions = _sessions(d_root)
        v_sessions = _sessions(v_root)
        cards = []
        for session, (d_tdir, idxs, d_info) in d_sessions.items():
            if session not in v_sessions:
                continue
            v_tdir, _, v_info = v_sessions[session]
            for idx in idxs:
                c = _card(label, session, idx, d_tdir, d_info, v_tdir, v_info, out_dir, uid)
                uid += 1
                if c:
                    cards.append(c)
                    n_cards += 1
        if cards:
            sections.append(f"<h2>{html.escape(label)} "
                            f"<span class='count'>({len(cards)} queries)</span></h2>"
                            + "".join(cards))

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Offset matcher — disk+lightglue vs DINOv3 (mid frame)</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 780px;
         padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #8886; padding-bottom: .3rem; }}
  h2 .count {{ font-size: .8rem; font-weight: 400; opacity: .6; }}
  .card {{ border: 1px solid #8884; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .card h3 {{ margin: 0 0 .2rem; font-size: 1rem; }}
  .meta {{ margin: 0 0 .8rem; opacity: .75; font-size: .85rem; }}
  .cols {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .col {{ flex: 0 0 auto; }}
  .tag {{ font-size: .78rem; font-weight: 600; margin-top: .3rem; }}
  .tag.warn {{ color: #b06a00; }}
  .sub {{ font-size: .72rem; opacity: .7; }}
  .stack {{ position: relative; width: 320px; height: 240px; background: #0002; }}
  .stack img {{ position: absolute; inset: 0; width: 320px; height: 240px; object-fit: fill; }}
  label {{ display: block; margin-top: .6rem; font-size: .85rem; }}
  input[type=range] {{ width: 320px; vertical-align: middle; }}
  .intro {{ background: #8881; border-radius: 8px; padding: .8rem 1rem; }}
</style>
<h1>Offset matcher — disk+lightglue vs DINOv3 <span style="font-size:.9rem;opacity:.6">(mid frame)</span></h1>
<div class="intro">
  <p>Warp = <b>offset &compfn; linear</b>. Both columns share the same linear part
  (homography, disk+lightglue, fit at <code>--match_scale</code>); only the
  <b>offset</b> stage's matcher differs &mdash; <b>left</b> uses disk+lightglue
  (the current default), <b>right</b> uses DINOv3. Each image is the clip's
  <b>middle frame</b>.</p>
  <p><b>blend</b> crossfades the transferred still over the query. Misalignment
  reads as ghosting/doubling as you sweep blend &mdash; compare columns to see
  which matcher localises the shift better. Per-column numbers: estimated
  <code>(tx, ty)</code> &middot; offset inliers/matches &middot; fraction of the
  NNF landing in bounds (plus the offset status when it is not a clean
  <code>ok</code>).</p>
</div>
{"".join(sections) if sections else "<p>No runs found.</p>"}
<script>
document.querySelectorAll("input.b").forEach(b => b.addEventListener("input", () => {{
  const row = b.dataset.row;
  document.querySelectorAll(`img.t[data-row="${{row}}"]`).forEach(
    v => v.style.opacity = b.value / 100);
  document.querySelector(`.bv[data-row="${{row}}"]`).textContent = b.value + "%";
}}));
</script>
"""
    with open(args.out, "w") as f:
        f.write(page)
    print(f"Wrote {args.out} ({n_cards} cards)")


if __name__ == "__main__":
    main()
