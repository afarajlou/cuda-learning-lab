# Lesson 4 — Coalesced vs. strided global-memory access

**Research question:** Why does the mapping of neighboring threads to neighboring addresses matter on a GPU?

## Read the kernel

In `04_memory_access.cu`, thread `tid` loads either `x[tid]` or `x[tid * 8]`. Both produce correct results, but a warp of neighboring threads accesses very different address patterns.

| Pattern | Addresses used by threads 0,1,2,3 | Interpretation |
|---|---|---|
| Contiguous | 0, 1, 2, 3 | Consecutive elements often make efficient memory transactions |
| Strided by 8 | 0, 8, 16, 24 | Wider spread may require more memory transactions |

CUDA coalescing depends on device architecture, element size, alignment, caching, and addresses touched. **This program only checks correctness; it does not measure GPU bandwidth or claim an observed speedup.**

## Study steps

1. Read [NVIDIA CUDA Best Practices: Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).
2. Draw addresses touched by 32 adjacent threads in both kernels.
3. Identify `blockIdx.x`, `blockDim.x`, `threadIdx.x`, and the bounds guard.
4. Predict the memory-transaction difference and state the assumptions.
5. If you later get an NVIDIA GPU, add CUDA-event timing with many repetitions and a warm-up, check correctness and report hardware/compiler versions.

**Connection to LLM/HPC:** Layout choices affect memory efficiency in matrix products, embedding gathers, tensor operations, and attention kernels.

**Question for your notes:** Why might the same stride be more or less costly with different element sizes or caching?
