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


def _load_blank(ref_path):
    """Load frame 0 of the reference video as the no-contact blank frame."""
    cap = cv2.VideoCapture(ref_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read blank frame from {ref_path}")
    return _to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class TactileTransferDataset(data.Dataset):
    """Paired dataset of PatchMatch-transferred and ground-truth tactile videos.

    Each sample is a frame pair (t-1, t) from the transferred video as input
    and frame t from the query (GT) video as target.

    When residual=True, inputs and targets are expressed as contact residuals:
        residual[t] = video[t] - blank
    where blank is frame 0 of the reference video ({pair_idx}_ref_shadow.mp4).
    The returned dict includes a 'blank' key for reconstruction.

    Directory layout expected:
        transfer_dir/
            {obj_id}/
                {pair_idx}_transferred_em.mp4   # network input
                {pair_idx}_query_shadow.mp4      # ground truth
                {pair_idx}_ref_shadow.mp4        # reference (blank source)
    """

    NUM_PAIRS = 8

    def __init__(self, transfer_dir, object_ids, split='train', use_hflip=True,
                 residual=False):
        super().__init__()
        self.transfer_dir = transfer_dir
        self.split = split
        self.use_hflip = use_hflip and (split == 'train')
        self.residual = residual

        # Build flat sample index: list of (obj_id, pair_idx, frame_idx)
        self.samples = []
        for obj_id in object_ids:
            obj_dir = os.path.join(transfer_dir, str(obj_id))
            for pair_idx in range(self.NUM_PAIRS):
                vid_path = os.path.join(obj_dir, f"{pair_idx}_transferred_em.mp4")
                if not os.path.exists(vid_path):
                    continue
                if residual:
                    ref_path = os.path.join(obj_dir, f"{pair_idx}_ref_shadow.mp4")
                    if not os.path.exists(ref_path):
                        continue
                cap = cv2.VideoCapture(vid_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if n_frames <= 0:
                    continue
                for t in range(n_frames):
                    self.samples.append((obj_id, pair_idx, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        obj_id, pair_idx, t = self.samples[index]
        obj_dir = os.path.join(self.transfer_dir, str(obj_id))

        lq_path = os.path.join(obj_dir, f"{pair_idx}_transferred_em.mp4")
        gt_path = os.path.join(obj_dir, f"{pair_idx}_query_shadow.mp4")

        cap_lq = cv2.VideoCapture(lq_path)
        cap_gt = cv2.VideoCapture(gt_path)

        try:
            frame_t = _read_frame(cap_lq, t)
            frame_t_minus_1 = _read_frame(cap_lq, max(0, t - 1))
            frame_gt = _read_frame(cap_gt, t)
        finally:
            cap_lq.release()
            cap_gt.release()

        if self.use_hflip and random.random() < 0.5:
            frame_t = np.fliplr(frame_t).copy()
            frame_t_minus_1 = np.fliplr(frame_t_minus_1).copy()
            frame_gt = np.fliplr(frame_gt).copy()

        lq = torch.stack([_to_tensor(frame_t_minus_1), _to_tensor(frame_t)], dim=0)
        gt = _to_tensor(frame_gt)

        if self.residual:
            ref_path = os.path.join(obj_dir, f"{pair_idx}_ref_shadow.mp4")
            blank = _load_blank(ref_path)
            lq = lq - blank.unsqueeze(0)   # (2, 3, H, W)
            gt = gt - blank
            return {'lq': lq, 'gt': gt, 'blank': blank, 'meta': (obj_id, pair_idx, t)}

        return {'lq': lq, 'gt': gt, 'meta': (obj_id, pair_idx, t)}

    def lq_video_exists(self, obj_id, pair_idx):
        path = os.path.join(self.transfer_dir, str(obj_id),
                            f"{pair_idx}_transferred_em.mp4")
        return os.path.exists(path)

    def iter_video_pairs(self, obj_id, pair_idx):
        """Yield (lq_pair, gt_frame, blank_or_None) for every frame in order.

        blank_or_None is a (3,H,W) float tensor when residual=True, else None.
        lq and gt are in residual space when residual=True.
        """
        obj_dir = os.path.join(self.transfer_dir, str(obj_id))
        lq_path = os.path.join(obj_dir, f"{pair_idx}_transferred_em.mp4")
        gt_path = os.path.join(obj_dir, f"{pair_idx}_query_shadow.mp4")

        blank = None
        if self.residual:
            ref_path = os.path.join(obj_dir, f"{pair_idx}_ref_shadow.mp4")
            blank = _load_blank(ref_path)

        cap_lq = cv2.VideoCapture(lq_path)
        cap_gt = cv2.VideoCapture(gt_path)
        n_frames = int(cap_lq.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            prev_frame = None
            for t in range(n_frames):
                ret_lq, f_lq = cap_lq.read()
                ret_gt, f_gt = cap_gt.read()
                if not ret_lq or not ret_gt:
                    break
                f_lq = cv2.cvtColor(f_lq, cv2.COLOR_BGR2RGB)
                f_gt = cv2.cvtColor(f_gt, cv2.COLOR_BGR2RGB)
                if prev_frame is None:
                    prev_frame = f_lq
                lq = torch.stack([_to_tensor(prev_frame), _to_tensor(f_lq)], dim=0)
                gt = _to_tensor(f_gt)
                prev_frame = f_lq

                if self.residual:
                    lq = lq - blank.unsqueeze(0)
                    gt = gt - blank

                yield lq, gt, blank
        finally:
            cap_lq.release()
            cap_gt.release()
