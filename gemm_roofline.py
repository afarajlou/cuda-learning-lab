"""CPU-only analytical estimate of arithmetic intensity for tiled GEMM.

Square float32 C=A@B, n x n, idealized compulsory global traffic:
read A and B once and write C once. Actual GPU traffic may be higher
due to caching, tile boundaries, spills, and repeated loads.
Run: python gemm_roofline.py
"""
def gemm_intensity(n):
    if n <= 0:
        raise ValueError("matrix size must be positive")
    flops = 2 * n**3
    minimum_bytes = 3 * n**2 * 4
    return flops / minimum_bytes


def main():
    for n in (64, 256, 1024):
        print(f"n={n:4d} ideal intensity={gemm_intensity(n):.2f} FLOP/byte")
    assert abs(gemm_intensity(256) - (256 / 6)) < 1e-9
    print("Lower-bound traffic estimate only; no GPU timing.")


if __name__ == "__main__":
    main()
