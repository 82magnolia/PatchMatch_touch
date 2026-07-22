"""Builds an HTML overlay report for decomposed (offset o linear) transfers.

For each query it stacks the query touch video and the transferred video and
gives you two trackbars:
  frame  -- scrubs both videos together (they are seeked to the same time)
  blend  -- crossfades the transferred video over the query video

Misalignment shows up as ghosting/doubling as you sweep the blend slider, which
is what the decomposition's offset stage is supposed to remove.

Videos are referenced by relative path rather than inlined, so the report stays
small and always reflects whatever is currently on disk.

Example usage:
    python test_scripts/make_decomposed_overlay_report.py \
        --run real:log/decomposed_eval/real \
        --run sim:log/decomposed_eval/sim \
        --out log/decomposed_transfer_overlay.html
"""

import argparse
import html
import os
import pickle
import re
import subprocess
from os import path as osp

ASSET_DIRNAME = "decomposed_overlay_assets"


def _transcode(src, dst):
    """Re-encode to browser-playable H.264.

    The pipeline writes videos with OpenCV's "mp4v" fourcc, i.e. MPEG-4 Part 2,
    which no browser decodes in an HTML5 <video> element -- the element loads
    but never paints, so the report shows blank boxes. H.264 (yuv420p, with the
    moov atom moved to the front) plays everywhere.

    Skips work when the existing output is newer than its source.
    """
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


def _discover(group_root):
    """group_root/<session>/transfer/{idx}_transferred.mp4 -> nested dict."""
    sessions = {}
    if not osp.isdir(group_root):
        return sessions
    for session in sorted(os.listdir(group_root),
                          key=lambda x: int(x) if x.isdigit() else x):
        tdir = osp.join(group_root, session, "transfer")
        if not osp.isdir(tdir):
            continue
        idxs = sorted(int(m.group(1))
                      for m in (re.match(r"^(\d+)_transferred\.mp4$", f)
                                for f in os.listdir(tdir)) if m)
        if idxs:
            sessions[session] = (tdir, idxs, _load_info(tdir))
    return sessions


def _card(group, session, idx, tdir, info, out_dir, uid):
    rel = lambda p: html.escape(osp.relpath(p, out_dir))
    transferred = osp.join(tdir, f"{idx}_transferred.mp4")
    query = osp.join(tdir, f"{idx}_query_shadow.mp4")
    if not osp.exists(query):                       # --video_type sim
        query = osp.join(tdir, f"{idx}_query_sim.mp4")
    if not osp.exists(query) or not osp.exists(transferred):
        return ""

    # Transcode both into the report's own asset dir (see _transcode).
    stem = f"{group}_{session}_{idx}"
    q_web = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_query.mp4")
    t_web = osp.join(out_dir, ASSET_DIRNAME, f"{stem}_transferred.mp4")
    if not (_transcode(query, q_web) and _transcode(transferred, t_web)):
        return ""
    query, transferred = q_web, t_web

    d = info.get(idx, {})
    tx, ty = d.get("offset", (float("nan"),) * 2)
    status = d.get("offset_status", "n/a")
    badge = "ok" if status == "ok" else "warn"
    meta = (f"linear {d.get('linear_inliers', '?')}/{d.get('linear_matches', '?')} inliers "
            f"&middot; ratio {d.get('ratio', '?')} &middot; "
            f"offset ({tx:+.1f}, {ty:+.1f}) from {d.get('offset_matches', '?')} matches "
            f"&middot; in-bounds {100 * d.get('valid_fraction', float('nan')):.1f}%")

    return f"""
<div class="card">
  <h3>{html.escape(group)} / session {html.escape(session)} / query {idx}
      <span class="badge {badge}">{html.escape(status)}</span></h3>
  <p class="meta">{meta}</p>
  <div class="stack" id="stack{uid}">
    <video id="q{uid}" src="{rel(query)}" muted playsinline preload="auto"></video>
    <video id="t{uid}" src="{rel(transferred)}" muted playsinline preload="auto"
           style="opacity:0.5"></video>
  </div>
  <label>frame <input type="range" id="f{uid}" min="0" max="1000" value="0"></label>
  <label>blend <input type="range" id="b{uid}" min="0" max="100" value="50">
         <span id="bv{uid}">50%</span></label>
  <script>
  (function() {{
    const q = document.getElementById("q{uid}"), t = document.getElementById("t{uid}");
    const f = document.getElementById("f{uid}"), b = document.getElementById("b{uid}");
    const bv = document.getElementById("bv{uid}");
    // A <video> that has never been played or seeked paints nothing, so nudge
    // both off zero once metadata is in to force the first frame to render.
    [q, t].forEach(v => v.addEventListener("loadedmetadata", () => {{
      v.currentTime = Math.min(0.01, (v.duration || 1) / 2);
    }}, {{ once: true }}));
    // Seek both to the same normalised position. Slider is 0-1000 rather than
    // per-frame because frame count is not known until metadata loads.
    f.addEventListener("input", () => {{
      if (!q.duration) return;
      const p = f.value / 1000;
      q.currentTime = p * q.duration;
      if (t.duration) t.currentTime = p * t.duration;
    }});
    b.addEventListener("input", () => {{
      t.style.opacity = b.value / 100;
      bv.textContent = b.value + "%";
    }});
  }})();
  </script>
</div>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL:ROOT",
                    help="Group label and its root dir (root/<session>/transfer/...). "
                         "Repeatable.")
    ap.add_argument("--out", default="log/decomposed_transfer_overlay.html")
    args = ap.parse_args()

    out_dir = osp.dirname(osp.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    sections, uid, n_cards = [], 0, 0
    for spec in args.run:
        label, _, root = spec.partition(":")
        cards = []
        for session, (tdir, idxs, info) in _discover(root).items():
            for idx in idxs:
                c = _card(label, session, idx, tdir, info, out_dir, uid)
                uid += 1
                if c:
                    cards.append(c)
                    n_cards += 1
        if cards:
            sections.append(f"<h2>{html.escape(label)}</h2>" + "".join(cards))

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Decomposed transfer overlays</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 900px;
         padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #8886; padding-bottom: .3rem; }}
  .card {{ border: 1px solid #8884; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .card h3 {{ margin: 0 0 .2rem; font-size: 1rem; }}
  .meta {{ margin: 0 0 .8rem; opacity: .75; font-size: .85rem; }}
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
<h1>Decomposed transfer overlays</h1>
<div class="intro">
  <p>Warp = <b>offset &compfn; linear</b>. The linear part (affine/homography) is
  RANSAC-fit at <code>--match_scale</code>, conjugated into the video's coordinate
  space, then stripped of its offset; the offset is re-estimated at
  <code>--scale</code> as the median match displacement between the
  zero-offset-warped reference and the query.</p>
  <p><b>frame</b> scrubs both videos together; <b>blend</b> crossfades the
  transferred video over the query. Misalignment reads as ghosting while you
  sweep blend. <code>in-bounds</code> is the fraction of the NNF landing inside
  the reference canvas &mdash; the rest is clamped to the border.</p>
</div>
{"".join(sections) if sections else "<p>No runs found.</p>"}
"""
    with open(args.out, "w") as f:
        f.write(page)
    print(f"Wrote {args.out} ({n_cards} cards)")


if __name__ == "__main__":
    main()
