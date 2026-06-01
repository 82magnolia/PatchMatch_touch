import os
import random
import cv2
import torch
import torch.utils.data as data
import numpy as np


def _read_frame_gt(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _to_tensor(frame):
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)


class NormalBaselineDataset(data.Dataset):
    """Baseline dataset using a static surface normal image as input.

    Both frames of the lq pair are the same normal image; the target is the
    ground-truth query tactile video frame.

    Directory layout expected:
        query_normal_dir/
            {obj_id}/
                {pair_idx}_scale100_normal.jpg   # static normal image (input)
        transfer_dir/
            {obj_id}/
                {pair_idx}_query_shadow.mp4      # ground truth
    """

    NUM_PAIRS = 8

    def __init__(self, query_normal_dir, transfer_dir, object_ids,
                 split='train', use_hflip=True):
        super().__init__()
        self.query_normal_dir = query_normal_dir
        self.transfer_dir = transfer_dir
        self.split = split
        self.use_hflip = use_hflip and (split == 'train')

        # Build flat sample index: list of (obj_id, pair_idx, frame_idx)
        self.samples = []
        for obj_id in object_ids:
            for pair_idx in range(self.NUM_PAIRS):
                normal_path = os.path.join(query_normal_dir, str(obj_id),
                                           f"{pair_idx}_scale100_normal.jpg")
                gt_path = os.path.join(transfer_dir, str(obj_id),
                                       f"{pair_idx}_query_shadow.mp4")
                if not os.path.exists(normal_path) or not os.path.exists(gt_path):
                    continue
                cap = cv2.VideoCapture(gt_path)
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

        normal_path = os.path.join(self.query_normal_dir, str(obj_id),
                                   f"{pair_idx}_scale100_normal.jpg")
        gt_path = os.path.join(self.transfer_dir, str(obj_id),
                                f"{pair_idx}_query_shadow.mp4")

        normal_bgr = cv2.imread(normal_path)
        normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)

        cap_gt = cv2.VideoCapture(gt_path)
        cap_gt.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, frame_gt = cap_gt.read()
        cap_gt.release()
        if not ret:
            raise RuntimeError(f"Failed to read GT frame {t} from {gt_path}")
        frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)

        if self.use_hflip and random.random() < 0.5:
            normal_rgb = np.fliplr(normal_rgb).copy()
            frame_gt = np.fliplr(frame_gt).copy()

        normal_t = _to_tensor(normal_rgb)
        gt = _to_tensor(frame_gt)
        lq = torch.stack([normal_t, normal_t], dim=0)

        return {'lq': lq, 'gt': gt, 'meta': (obj_id, pair_idx, t)}

    def lq_video_exists(self, obj_id, pair_idx):
        path = os.path.join(self.query_normal_dir, str(obj_id),
                            f"{pair_idx}_scale100_normal.jpg")
        return os.path.exists(path)

    def iter_video_pairs(self, obj_id, pair_idx):
        """Yield (lq_pair, gt_frame) tensors for every GT frame; lq is static normal."""
        normal_path = os.path.join(self.query_normal_dir, str(obj_id),
                                   f"{pair_idx}_scale100_normal.jpg")
        gt_path = os.path.join(self.transfer_dir, str(obj_id),
                               f"{pair_idx}_query_shadow.mp4")

        normal_bgr = cv2.imread(normal_path)
        normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
        normal_t = _to_tensor(normal_rgb)
        lq = torch.stack([normal_t, normal_t], dim=0)

        cap_gt = cv2.VideoCapture(gt_path)
        try:
            while True:
                ret, frame = cap_gt.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gt = _to_tensor(frame_rgb)
                yield lq, gt
        finally:
            cap_gt.release()
