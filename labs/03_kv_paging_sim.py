#!/usr/bin/env python3
"""Compare max-length reservation with paged KV allocation.

This simulates token slots, not runtime metadata or kernel performance.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--median", type=float, default=900)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lengths = [
        max(1, min(args.max_length, int(rng.lognormvariate(math.log(args.median), args.sigma))))
        for _ in range(args.requests)
    ]

    used = sum(lengths)
    static_reserved = args.requests * args.max_length
    print(f"requests: {args.requests}")
    print(
        f"length p50/p95/max: {int(statistics.median(lengths))}/"
        f"{percentile(lengths, 0.95)}/{max(lengths)}"
    )
    print(f"actual token slots used: {used:,}")
    print(
        f"static max-length reservation: {static_reserved:,} slots "
        f"({100 * (static_reserved - used) / static_reserved:.1f}% unused)"
    )
    print()
    print(f"{'block':>7} {'allocated slots':>18} {'tail waste':>13} {'waste %':>10}")
    print("-" * 55)
    for block in args.blocks:
        allocated = sum(math.ceil(length / block) * block for length in lengths)
        waste = allocated - used
        print(f"{block:>7} {allocated:>18,} {waste:>13,} {100 * waste / allocated:>9.2f}%")

    print()
    print("Paging removes max-length reservation and external fragmentation in the block pool.")
    print("It still has tail waste, block-table metadata, and attention indirection.")


if __name__ == "__main__":
    main()
