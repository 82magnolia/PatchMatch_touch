"""Efros/Freeman-style image quilting with SSD candidates and seam blending."""

from __future__ import annotations


def _min_cut(error):
    """Return a top-to-bottom minimum-cost cut as one column index per row."""
    import numpy as np

    cost = error.astype(np.float64).copy()
    parent = np.zeros(cost.shape, dtype=np.int32)
    for row in range(1, cost.shape[0]):
        previous = cost[row - 1]
        for col in range(cost.shape[1]):
            lo, hi = max(0, col - 1), min(cost.shape[1], col + 2)
            offset = int(np.argmin(previous[lo:hi]))
            parent[row, col] = lo + offset
            cost[row, col] += previous[parent[row, col]]
    cut = np.zeros(cost.shape[0], dtype=np.int32)
    cut[-1] = int(np.argmin(cost[-1]))
    for row in range(cost.shape[0] - 2, -1, -1):
        cut[row] = parent[row + 1, cut[row + 1]]
    return cut


def _candidate_patches(image, block, max_candidates, rng):
    import numpy as np

    height, width = image.shape[:2]
    if height < block or width < block:
        raise ValueError(
            f"Quilting block size {block} exceeds input patch {width}x{height}"
        )
    positions = [
        (y, x)
        for y in range(height - block + 1)
        for x in range(width - block + 1)
    ]
    if max_candidates is not None and len(positions) > max_candidates:
        chosen = rng.choice(len(positions), size=max_candidates, replace=False)
        positions = [positions[int(index)] for index in chosen]
    return np.stack([image[y : y + block, x : x + block] for y, x in positions])


def quilt(
    image,
    output_shape,
    block=30,
    overlap=6,
    tolerance=0.1,
    seed=0,
    max_candidates=1024,
):
    """Synthesize an RGB image of ``output_shape=(height, width)``."""
    import numpy as np

    if not 0 < overlap < block:
        raise ValueError("overlap must be greater than zero and smaller than block")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    rng = np.random.default_rng(seed)
    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 2:
        source = np.repeat(source[..., None], 3, axis=2)
    candidates = _candidate_patches(source, block, max_candidates, rng)
    out_h, out_w = output_shape
    canvas = np.zeros((out_h, out_w, source.shape[2]), dtype=np.float32)
    filled = np.zeros((out_h, out_w), dtype=bool)
    stride = block - overlap

    for top in range(0, out_h, stride):
        for left in range(0, out_w, stride):
            height, width = min(block, out_h - top), min(block, out_w - left)
            current = canvas[top : top + height, left : left + width]
            occupied = filled[top : top + height, left : left + width]
            pool = candidates[:, :height, :width]
            if occupied.any():
                difference = (pool - current[None, ...]) ** 2
                errors = (difference * occupied[None, ..., None]).sum(axis=(1, 2, 3))
                threshold = errors.min() * (1.0 + tolerance) + 1e-12
                eligible = np.flatnonzero(errors <= threshold)
                patch = pool[int(rng.choice(eligible))]
            else:
                patch = pool[int(rng.integers(len(pool)))]

            take_new = ~occupied
            if left > 0:
                ov = min(overlap, width)
                error = ((current[:, :ov] - patch[:, :ov]) ** 2).sum(axis=2)
                cut = _min_cut(error)
                for row, column in enumerate(cut):
                    take_new[row, column:ov] = True
            if top > 0:
                ov = min(overlap, height)
                error = ((current[:ov] - patch[:ov]) ** 2).sum(axis=2).T
                cut = _min_cut(error)
                for column, row in enumerate(cut):
                    take_new[row:ov, column] = True

            current[take_new] = patch[take_new]
            filled[top : top + height, left : left + width] = True
    return np.clip(canvas, 0, 255).astype(np.uint8)
