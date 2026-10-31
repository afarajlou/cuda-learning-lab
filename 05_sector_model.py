"""CPU-only model of sectors touched by a warp's float32 loads.

Assumes 32 lanes, aligned base address, 32-byte sectors and one load/lane.
Not a GPU benchmark or simulator. Run: python 05_sector_model.py
"""
LANES = 32
ELEMENT_BYTES = 4
SECTOR_BYTES = 32


def sector_count(stride: int, offset_elements: int = 0) -> int:
    if stride < 1 or offset_elements < 0:
        raise ValueError("stride >= 1 and offset_elements >= 0 required")
    return len({
        ((offset_elements + lane * stride) * ELEMENT_BYTES) // SECTOR_BYTES
        for lane in range(LANES)
    })


def main() -> None:
    assert [sector_count(s) for s in (1, 2, 4, 8)] == [4, 8, 16, 32]
    assert sector_count(1, 1) == 5
    print("stride | sectors | useful sector-byte fraction")
    for stride in (1, 2, 4, 8):
        count = sector_count(stride)
        fraction = LANES * ELEMENT_BYTES / (count * SECTOR_BYTES)
        print(f"{stride:>6} | {count:>7} | {fraction:>27.1%}")
    print("Aligned-base analytical model; no GPU timing performed.")


if __name__ == "__main__":
    main()
