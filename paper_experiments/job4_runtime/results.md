# Job 4 — Runtime analysis

Single NVIDIA GeForce RTX 3090, 6 reference touches and 1 query,
50-frame touch videos, mean of 10 timed repeats after a
warm-up pass.

| Stage | Time |
|---|---|
| Retrieval, after DINOv3 feature extraction | 0.08 ms |
| Coarse alignment, after local feature matching | 0.021 s |
| Network refinement, per frame | 8.5 ms |
| Network refinement, per 50-frame video | 0.42 s |


Repeating the measurement on the largest reference set in the benchmark
(27 references) gives
0.07 ms,
0.019 s and
8.6 ms/frame — essentially unchanged with N.

Excluded costs, for reference: DINOv3 feature extraction
0.06 s;
local feature matching 0.08 s.

Sentences for the paper:

- Retrieval phase after DINOv3 feature extraction takes 0.08 ms.
- Coarse alignment after local feature matching takes 0.021 s.
- Neural network-based refinement takes 8.5 ms per frame.
