import os
import random
import cv2
import torch
import torch.utils.data as data
import numpy as np


def _read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _to_tensor(frame):
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)


def _resolve_lq_path(obj_dir, pair_idx):
    """Find the transferred video for pair_idx, trying both backend namings.

    '_transferred_em' is main_retrieval_transfer_accel.py's (patchmatch/EM
    backend) output naming; plain '_transferred' is
    main_retrieval_transfer_feat_match.py's (dinov3_feat_match backend).
    Returns None if neither exists.
    """
    for suffix in ("_transferred_em", "_transferred"):
        path = os.path.join(obj_dir, f"{pair_idx}{suffix}.mp4")
        if os.path.exists(path):
            return path
    return None


# Flat/no-contact tactile_normal encoding for the surface normal (0, 0, 1):
# Taxim's height_map_to_normals + uint8((n+1)/2*255) encoding maps a flat gel
# (normal exactly (0,0,1) everywhere) to this exact RGB triple regardless of
# object or video (see Taxim/OpticalSimulation/simOptical.py). Used as a fixed
# residual baseline for tactile_normal videos instead of a frame-0-derived
# blank, since the physically correct no-contact reading is a universal
# constant here (unlike the shadow/appearance domain, where it varies per
# object/lighting and must be read from frame 0).
_FLAT_NORMAL_RGB = torch.tensor([127.0, 127.0, 255.0]) / 255.0


def _normal_blank(h, w):
    return _FLAT_NORMAL_RGB.view(3, 1, 1).expand(3, h, w).clone()


def _load_blank(video_path, normal_blank=False):
    """Load the no-contact blank frame for residual mode.

    Default: frame 0 of the given video. When normal_blank=True, the video's
    frame-0 content is discarded (only used for its shape) and the constant
    flat-normal encoding is returned instead -- see _FLAT_NORMAL_RGB.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read blank frame from {video_path}")
    if normal_blank:
        h, w = frame.shape[:2]
        return _normal_blank(h, w)
    return _to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# --- Conditioning helpers (render_mask concat + static-geometry FiLM) ---

def _read_gray_frame(cap, idx, h, w):
    """Read frame idx of a mask video as a normalised (h,w,1) float array.

    Clamps to the mask's frame count and falls back to zeros if the frame can't
    be read, so a mask video shorter than the transferred video is tolerated.
    """
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > 0:
        idx = min(idx, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return np.zeros((h, w, 1), np.float32)
    g = frame.astype(np.float32).mean(axis=2) / 255.0
    if g.shape != (h, w):
        g = cv2.resize(g, (w, h))
    return g[..., None]


def _read_film_image(path, h, w):
    """Read a static conditioning image as a normalised (h,w,3) float array."""
    im = cv2.imread(path)
    if im is None:
        raise RuntimeError(f"Failed to read film image {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if im.shape[:2] != (h, w):
        im = cv2.resize(im, (w, h))
    return im


def _chw(arr):
    """(H,W,C) float array -> (C,H,W) float tensor (already normalised)."""
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()


class TactileTransferDataset(data.Dataset):
    """Paired dataset of PatchMatch-transferred and ground-truth tactile videos.

    Each sample is a frame pair (t-1, t) from the transferred video as input
    and frame t from the query (GT) video as target.

    When residual=True, inputs and targets are expressed as contact residuals:
        residual[t] = video[t] - blank
    where blank is, by default, frame 0 of the transferred video itself
    ({pair_idx}_transferred_em.mp4 or {pair_idx}_transferred.mp4, see
    _resolve_lq_path) — always available at real deployment (unlike the
    query's own touch video, which only exists for paired train/eval data)
    and already in query coordinate space (unlike the raw reference video,
    which lives in reference coordinate space). When normal_blank=True
    (tactile_normal-modality videos), blank is instead the fixed encoding of
    the flat surface normal (0,0,1) — see _FLAT_NORMAL_RGB — since that is
    the true, video-independent no-contact reading in that domain.
    The returned dict includes a 'blank' key for reconstruction.

    video_type selects which appearance domain's videos to load, matching the
    {idx}_{video_type}.mp4 / {idx}_query_{video_type}.mp4 /
    {idx}_ref_{video_type}.mp4 naming written by
    main_retrieval_transfer_feat_match.py (default 'shadow'; e.g.
    'tactile_normal' for the surface-normal-encoded domain).

    Directory layout expected:
        transfer_dir/
            {obj_id}/
                {pair_idx}_transferred_em.mp4        # network input, blank source
                                                      # (patchmatch/EM backend), or
                {pair_idx}_transferred.mp4            # (dinov3_feat_match backend)
                {pair_idx}_query_{video_type}.mp4     # ground truth
                {pair_idx}_ref_{video_type}.mp4       # reference (viz only)
    """

    NUM_PAIRS = 8

    def __init__(self, transfer_dir, object_ids, split='train', use_hflip=True,
                 residual=False, cond_dir=None, mask_cond=False,
                 film_modality=None, film_scale=100, video_type='shadow',
                 normal_blank=False):
        super().__init__()
        self.transfer_dir = transfer_dir
        self.split = split
        self.use_hflip = use_hflip and (split == 'train')
        self.residual = residual
        self.video_type = video_type
        self.normal_blank = normal_blank
        # Query conditioning (both sourced from cond_dir/{obj}/...):
        #   mask_cond   -> concat the per-frame render_mask (1ch, aligned) to lq
        #   film_modality -> load a static geometry render (3ch) returned as 'film'
        # These are query-side signals available at deployment; do NOT use the
        # tactile-video normals, which are derived from the GT.
        self.cond_dir = cond_dir
        self.mask_cond = mask_cond
        self.film_modality = film_modality
        self.film_scale = film_scale

        # Build flat sample index: list of (obj_id, pair_idx, frame_idx)
        self.samples = []
        for obj_id in object_ids:
            obj_dir = self._obj_dir(obj_id)
            for pair_idx in range(self.NUM_PAIRS):
                vid_path = _resolve_lq_path(obj_dir, pair_idx)
                if vid_path is None:
                    continue
                cap = cv2.VideoCapture(vid_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if n_frames <= 0:
                    continue
                for t in range(n_frames):
                    self.samples.append((obj_id, pair_idx, t))

    def _obj_dir(self, obj_id):
        """Directory holding this object's videos. Overridable for other layouts."""
        return os.path.join(self.transfer_dir, str(obj_id))

    def _cond_base(self, obj_id):
        """Directory holding this object's query conditioning signals."""
        return os.path.join(self.cond_dir, str(obj_id))

    def _mask_path(self, obj_id, pair_idx):
        return os.path.join(self._cond_base(obj_id), f"{pair_idx}_render_mask.mp4")

    def _film_path(self, obj_id, pair_idx):
        return os.path.join(self._cond_base(obj_id),
                            f"{pair_idx}_scale{self.film_scale}_{self.film_modality}.jpg")

    def _load_film(self, obj_id, pair_idx, h, w, do_flip):
        film = _read_film_image(self._film_path(obj_id, pair_idx), h, w)
        if do_flip:
            film = np.fliplr(film).copy()
        return _chw(film)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        obj_id, pair_idx, t = self.samples[index]
        obj_dir = self._obj_dir(obj_id)

        lq_path = _resolve_lq_path(obj_dir, pair_idx)
        gt_path = os.path.join(obj_dir, f"{pair_idx}_query_{self.video_type}.mp4")

        cap_lq = cv2.VideoCapture(lq_path)
        cap_gt = cv2.VideoCapture(gt_path)

        try:
            frame_t = _read_frame(cap_lq, t)
            frame_t_minus_1 = _read_frame(cap_lq, max(0, t - 1))
            frame_gt = _read_frame(cap_gt, t)
        finally:
            cap_lq.release()
            cap_gt.release()

        do_flip = self.use_hflip and random.random() < 0.5
        if do_flip:
            frame_t = np.fliplr(frame_t).copy()
            frame_t_minus_1 = np.fliplr(frame_t_minus_1).copy()
            frame_gt = np.fliplr(frame_gt).copy()

        h, w = frame_t.shape[:2]
        lq = torch.stack([_to_tensor(frame_t_minus_1), _to_tensor(frame_t)], dim=0)
        gt = _to_tensor(frame_gt)

        if self.residual:
            blank = _load_blank(lq_path, self.normal_blank)
            lq = lq - blank.unsqueeze(0)   # (2, 3, H, W); only RGB is residualised
            gt = gt - blank

        # Concat the per-frame render_mask (aligned, raw — never residualised).
        if self.mask_cond:
            cap_m = cv2.VideoCapture(self._mask_path(obj_id, pair_idx))
            try:
                m_t = _read_gray_frame(cap_m, t, h, w)
                m_tm1 = _read_gray_frame(cap_m, max(0, t - 1), h, w)
            finally:
                cap_m.release()
            if do_flip:
                m_t = np.fliplr(m_t).copy(); m_tm1 = np.fliplr(m_tm1).copy()
            mask_pair = torch.stack([_chw(m_tm1), _chw(m_t)], dim=0)  # (2,1,H,W)
            lq = torch.cat([lq, mask_pair], dim=1)                    # (2,3+1,H,W)

        out = {'lq': lq, 'gt': gt, 'meta': (obj_id, pair_idx, t)}
        if self.residual:
            out['blank'] = blank
        if self.film_modality:
            out['film'] = self._load_film(obj_id, pair_idx, h, w, do_flip)
        return out

    def lq_video_exists(self, obj_id, pair_idx):
        obj_dir = self._obj_dir(obj_id)
        return _resolve_lq_path(obj_dir, pair_idx) is not None

    def iter_video_pairs(self, obj_id, pair_idx):
        """Yield (lq_pair, gt_frame, blank_or_None, film_or_None) per frame.

        blank is a (3,H,W) tensor when residual=True else None. film is a
        (film_chans,H,W) tensor when film_modality is set else None. lq gains the
        concatenated render_mask channel when mask_cond is set. lq/gt are in
        residual space (RGB only) when residual=True.
        """
        obj_dir = self._obj_dir(obj_id)
        lq_path = _resolve_lq_path(obj_dir, pair_idx)
        gt_path = os.path.join(obj_dir, f"{pair_idx}_query_{self.video_type}.mp4")

        blank = None
        if self.residual:
            blank = _load_blank(lq_path, self.normal_blank)

        cap_lq = cv2.VideoCapture(lq_path)
        cap_gt = cv2.VideoCapture(gt_path)
        cap_m = cv2.VideoCapture(self._mask_path(obj_id, pair_idx)) if self.mask_cond else None
        n_frames = int(cap_lq.get(cv2.CAP_PROP_FRAME_COUNT))
        film = None

        try:
            prev_frame = None
            prev_mask = None
            for t in range(n_frames):
                ret_lq, f_lq = cap_lq.read()
                ret_gt, f_gt = cap_gt.read()
                if not ret_lq or not ret_gt:
                    break
                f_lq = cv2.cvtColor(f_lq, cv2.COLOR_BGR2RGB)
                f_gt = cv2.cvtColor(f_gt, cv2.COLOR_BGR2RGB)
                h, w = f_lq.shape[:2]
                if prev_frame is None:
                    prev_frame = f_lq
                lq = torch.stack([_to_tensor(prev_frame), _to_tensor(f_lq)], dim=0)
                gt = _to_tensor(f_gt)
                prev_frame = f_lq

                if self.residual:
                    lq = lq - blank.unsqueeze(0)
                    gt = gt - blank

                if self.mask_cond:
                    ret_m, f_m = cap_m.read()
                    if ret_m:
                        g = f_m.astype(np.float32).mean(axis=2) / 255.0
                        if g.shape != (h, w):
                            g = cv2.resize(g, (w, h))
                        cur_mask = g[..., None]
                    else:
                        cur_mask = prev_mask if prev_mask is not None else np.zeros((h, w, 1), np.float32)
                    if prev_mask is None:
                        prev_mask = cur_mask
                    mask_pair = torch.stack([_chw(prev_mask), _chw(cur_mask)], dim=0)
                    lq = torch.cat([lq, mask_pair], dim=1)
                    prev_mask = cur_mask

                if self.film_modality and film is None:
                    film = self._load_film(obj_id, pair_idx, h, w, do_flip=False)

                yield lq, gt, blank, film
        finally:
            cap_lq.release()
            cap_gt.release()
            if cap_m is not None:
                cap_m.release()
