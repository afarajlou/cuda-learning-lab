# CUDA Learning Lab

Three small, readable **CUDA C++** examples connecting GPU programming to AI hardware and modern ML workloads. This is a learning lab, **not a claim of optimized kernels or GPU benchmark results**.

| Lesson | CUDA source | Learn |
|---|---|---|
| 1. Tiled matrix multiplication | [01_matmul.cu](01_matmul.cu) | thread/block coordinates, global vs. shared memory, coalesced reads, barriers |
| 2. Two-layer MLP inference | [02_mlp.cu](02_mlp.cu) | neuron-level parallelism, ReLU, kernel launches, intermediate activations |
| 3. Single-head attention | [03_attention.cu](03_attention.cu) | query/key dot products, softmax, weighted values, attention bottlenecks |

**Recommended reading order:** Run the Python CPU reference → read CUDA source line by line → identify each thread's output → compile and compare GPU results when a suitable GPU is available.

## Windows setup

- **Without an NVIDIA GPU:** use Python 3: `py cpu_reference.py`. This exercises the math but does **not** execute CUDA.
- **With a supported NVIDIA GPU:** install a compatible [CUDA Toolkit and driver](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/). In a Visual Studio Developer PowerShell with `nvcc` available:
  ```powershell
  nvcc -std=c++17 01_matmul.cu -o matmul.exe
  .\matmul.exe
  nvcc -std=c++17 02_mlp.cu -o mlp.exe
  .\mlp.exe
  nvcc -std=c++17 03_attention.cu -o attention.exe
  .\attention.exe
  ```
Each CUDA program prints a simple correctness result against a host-side expectation. This lab is educational and uses tiny sizes.

## What to notice

**Matrix multiplication:** A 16×16 thread block cooperatively loads tiles into `__shared__` arrays. All threads reach `__syncthreads()` before using and replacing each tile. What changes for larger matrices?

**MLP:** Each thread computes one output neuron; the hidden layer is written to global memory between launches. How could fusion reduce memory traffic? What would be lost or complicated?

**Attention:** We compute scores, normalize with a deliberately serial single-thread softmax, then combine values. Why would this normalization become a bottleneck as sequence length grows? How do fused attention implementations reduce intermediate storage?

## Read next

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [NVIDIA shared-memory explanation](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)

**Scope and limitations:** Demonstrations are single precision, tiny fixed examples, not optimized, and exclude batching, tensor cores, multihead attention, numerical stability at production scale, training/backpropagation, and performance measurement. CUDA examples require a supported NVIDIA GPU to execute. CPU reference output is not GPU validation.
