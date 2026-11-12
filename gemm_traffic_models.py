"""Compare ideal, naive, and tiled global-memory traffic for square FP32 GEMM.

Analytical model only: no CUDA execution, caching, or hardware measurements.
Naive: each of N^2 output elements reads N pairs of operands.
Tiled: square tiles of width T; assume N divisible by T and each
thread block loads two T x T input tiles for each K tile.
Output is written once in both cases.
Run: python gemm_traffic_models.py
"""
def traffic(n, tile):
    if n <= 0 or tile <= 0 or n % tile:
        raise ValueError("n and tile must be positive and n divisible by tile")
    flops = 2 * n**3
    output_bytes = 4 * n**2
    ideal = 3 * output_bytes
    naive = 8 * n**3 + output_bytes
    tiled = 8 * n**3 // tile + output_bytes
    return flops, {"ideal_lower_bound": ideal, "naive_loads": naive, "tiled_loads": tiled}


def main():
    print("N=256, FP32, output writes included; estimated global-memory traffic")
    for tile in (1, 8, 16, 32):
        flops, cases = traffic(256, tile)
        print(f"tile={tile:2d} naive={cases['naive_loads']/2**20:6.2f} MiB "
              f"tiled={cases['tiled_loads']/2**20:6.2f} MiB "
              f"tiled_AI={flops/cases['tiled_loads']:6.2f} FLOP/byte")
    flops, cases = traffic(256, 16)
    print(f"ideal lower bound={cases['ideal_lower_bound']/2**20:.2f} MiB; "
          f"ideal_AI={flops/cases['ideal_lower_bound']:.2f}")
    assert traffic(256, 1)[1]["naive_loads"] == traffic(256, 1)[1]["tiled_loads"]
    assert cases["ideal_lower_bound"] <= cases["tiled_loads"] < cases["naive_loads"]


if __name__ == "__main__":
    main()
