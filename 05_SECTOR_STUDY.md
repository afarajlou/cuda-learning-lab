# Lesson 05 — Memory sectors and stride

A small CPU-only model of 32 CUDA lanes loading float32 values. It counts unique 32-byte sectors touched, assuming an aligned array base and one load per lane. This is an analytical exercise, not a GPU measurement.

Run on Windows: `py 05_sector_model.py`

| Stride | Sectors | Useful bytes / fetched bytes |
| --- | ---: | ---: |
| 1 | 4 | 100% |
| 2 | 8 | 50% |
| 4 | 16 | 25% |
| 8 | 32 | 12.5% |

These values describe one idealized access instruction; cache reuse, alignment, and device architecture can change real performance.

Read: [NVIDIA CUDA Best Practices — Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).

Questions:
1. Why does stride 8 touch more sectors than stride 1?
2. Why is the useful-byte fraction not a prediction of speedup?
3. What changes if the first address is offset by one float?
