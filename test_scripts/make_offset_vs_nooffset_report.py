"""Overlay report comparing WITH-offset vs NO-offset decomposed transfers.

For each query it shows two overlays side by side -- query + transferred, once
for the full decomposition (offset o linear) and once for the zero-offset linear
warp only -- driven by a single pair of trackbars per row:
  frame  -- scrubs every video in the row together
  blend  -- crossfades the transferred videos over the query everywhere

Sweeping blend makes the offset stage's effect obvious: whichever overlay ghosts
less is the better alignment, so this isolates what the offset stage buys you.

Each group pairs a WITH-offset root with a NO-offset root, both laid out as the
decomposed transfer writes them:
    <root>/<session>/transfer/{idx}_transferred.mp4
                             /{idx}_query_<video_type>.mp4
                             /decomposition.pkl

Videos are transcoded to H.264 (OpenCV writes mp4v, which browsers cannot
decode) into the report's own asset dir.

Example usage:
    python test_scripts/make_offset_vs_nooffset_report.py \
        --group sim:log/initial_transfer_report/sim:log/initial_transfer_report_nooffset/sim \
        --group real:log/initial_transfer_report/real:log/initial_transfer_report_nooffset/real \
        --out log/initial_transfer_offset_vs_nooffset.html
"""

import argparse
import html
import os
import pickle
import re
import subprocess
from os import path as osp

ASSET_DIRNAME = "offset_vs_nooffset_assets"


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


def _meta(info, idx):
    d = info.get(idx, {})
    tx, ty = d.get("offset", (float("nan"),) * 2)
    status = d.get("offset_status", "n/a")
    badge = "ok" if str(status).startswith("ok") or str(status).startswith("none") else "warn"
    line = (f"offset ({tx:+.1f}, {ty:+.1f}) &middot; "
            f"{d.get('offset_inliers', '?')}/{d.get('offset_matches', '?')} matches "
            f"&middot; in-bounds {100 * d.get('valid_fraction', float('nan')):.0f}%")
    return badge, status, line


def _stack(uid, tag, query_web, transferred_web, out_dir):
    rel = lambda p: html.escape(osp.relpath(p, out_dir))
    return f"""<div class="col">
  <div class="stack">
    <video class="q" data-vid="{uid}" src="{rel(query_web)}" muted playsinline preload="auto"></video>
    <video class="t" data-vid="{uid}" src="{rel(transferred_web)}" muted playsinline preload="auto"
           style="opacity:0.5"></video>
  </div>
  <div class="tag">{html.escape(tag)}</div>
</div>"""


def _card(group, session, idx, w_tdir, w_info, n_tdir, n_info, out_dir, uid):
    q_src = _query(w_tdir, idx)
    w_src = osp.join(w_tdir, f"{idx}_transferred.mp4")
    n_src = osp.join(n_tdir, f"{idx}_transferred.mp4")
    if not (q_src and osp.exists(w_src) and osp.exists(n_src)):
        return ""

    stem = f"{group}_{session}_{idx}"
    q_web = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_query.mp4")
    w_web = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_with.mp4")
    n_web = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_none.mp4")
    if not (_transcode(q_src, q_web) and _transcode(w_src, w_web) and _transcode(n_src, n_web)):
        return ""

    badge, status, line = _meta(w_info, idx)
    d = w_info.get(idx, {})
    head = (f"linear {d.get('linear_inliers', '?')}/{d.get('linear_matches', '?')} inliers "
            f"&middot; ratio {d.get('ratio', '?')}")

    return f"""
<div class="card" data-row="{uid}">
  <h3>{html.escape(group)} / session {html.escape(session)} / query {idx}
      <span class="badge {badge}">{html.escape(str(status))}</span></h3>
  <p class="meta">{head} &middot; {line}</p>
  <div class="cols">
    {_stack(uid, "with offset (offset o linear)", q_web, w_web, out_dir)}
    {_stack(uid, "no offset (linear only)", q_web, n_web, out_dir)}
  </div>
  <label>frame <input type="range" class="f" data-row="{uid}" min="0" max="1000" value="0"></label>
  <label>blend <input type="range" class="b" data-row="{uid}" min="0" max="100" value="50">
         <span class="bv" data-row="{uid}">50%</span></label>
</div>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True,
                    metavar="LABEL:WITH_ROOT:NO_ROOT",
                    help="Group label, its with-offset root, and its no-offset root. "
                         "Repeatable.")
    ap.add_argument("--out", default="log/initial_transfer_offset_vs_nooffset.html")
    args = ap.parse_args()

    out_dir = osp.dirname(osp.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    sections, uid, n_cards = [], 0, 0
    for spec in args.group:
        label, w_root, n_root = spec.split(":", 2)
        w_sessions = _sessions(w_root)
        n_sessions = _sessions(n_root)
        cards = []
        for session, (w_tdir, idxs, w_info) in w_sessions.items():
            if session not in n_sessions:
                continue
            n_tdir, _, n_info = n_sessions[session]
            for idx in idxs:
                c = _card(label, session, idx, w_tdir, w_info, n_tdir, n_info, out_dir, uid)
                uid += 1
                if c:
                    cards.append(c)
                    n_cards += 1
        if cards:
            sections.append(f"<h2>{html.escape(label)}</h2>" + "".join(cards))

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Initial transfer — offset vs no offset</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 780px;
         padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #8886; padding-bottom: .3rem; }}
  .card {{ border: 1px solid #8884; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .card h3 {{ margin: 0 0 .2rem; font-size: 1rem; }}
  .meta {{ margin: 0 0 .8rem; opacity: .75; font-size: .85rem; }}
  .cols {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .col {{ flex: 0 0 auto; }}
  .tag {{ font-size: .78rem; font-weight: 600; margin-top: .3rem; opacity: .85; }}
  .stack {{ position: relative; width: 320px; height: 240px; background: #0002; }}
  .stack video {{ position: absolute; inset: 0; width: 320px; height: 240px; }}
  label {{ display: block; margin-top: .6rem; font-size: .85rem; }}
  input[type=range] {{ width: 320px; vertical-align: middle; }}
  .badge {{ font-size: .7rem; padding: .1rem .4rem; border-radius: 4px; margin-left: .4rem;
           vertical-align: middle; }}
  .badge.ok {{ background: #1baf7a33; color: #0d7a52; }}
  .badge.warn {{ background: #eda10033; color: #8a5f00; }}
  @media (prefers-color-scheme: dark) {{
    .badge.ok {{ color: #4fd3a0; }} .badge.warn {{ color: #f0c04a; }}
  }}
  .intro {{ background: #8881; border-radius: 8px; padding: .8rem 1rem; }}
</style>
<h1>Initial transfer — offset vs no offset</h1>
<div class="intro">
  <p>Warp = <b>offset &compfn; linear</b>. Both columns share the same linear part
  (homography, disk+lightglue, fit at <code>--match_scale</code>); the
  <b>left</b> column adds the re-estimated offset, the <b>right</b> column keeps
  the centred zero-offset linear warp only.</p>
  <p><b>frame</b> scrubs both overlays together; <b>blend</b> crossfades the
  transferred videos over the query. Misalignment reads as ghosting while you
  sweep blend &mdash; compare the two columns to see what the offset stage buys.
  The badge/metadata describe the with-offset (left) estimate; <code>offset
  zeroed</code> means the estimated shift left the query region and was dropped
  (so left == right for that row).</p>
</div>
{"".join(sections) if sections else "<p>No runs found.</p>"}
<script>
// A <video> that has never been seeked paints nothing, so nudge each off zero.
document.querySelectorAll(".stack video").forEach(v =>
  v.addEventListener("loadedmetadata", () => {{
    v.currentTime = Math.min(0.01, (v.duration || 1) / 2);
  }}, {{ once: true }}));

document.querySelectorAll("input.f").forEach(f => f.addEventListener("input", () => {{
  const row = f.dataset.row, p = f.value / 1000;
  document.querySelectorAll(`.stack video[data-vid="${{row}}"]`).forEach(v => {{
    if (v.duration) v.currentTime = p * v.duration;
  }});
}}));
document.querySelectorAll("input.b").forEach(b => b.addEventListener("input", () => {{
  const row = b.dataset.row;
  document.querySelectorAll(`.stack video.t[data-vid="${{row}}"]`).forEach(
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
