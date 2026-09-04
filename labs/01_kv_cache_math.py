#!/usr/bin/env python3
"""Calculate logical KV-cache payload for MHA, GQA, and MQA models.

This uses the invariant formula:
    bytes = 2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_element

It does not include allocator blocks, alignment, scale metadata, temporary buffers,
or tensor-parallel replication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    bytes_per_element: float

    def kv_bytes(self, tokens: int, batch: int = 1) -> float:
        return (
            2
            * batch
            * self.layers
            * tokens
            * self.kv_heads
            * self.head_dim
            * self.bytes_per_element
        )


def gib(value: float) -> float:
    return value / 1024**3


def mib(value: float) -> float:
    return value / 1024**2


def requests_that_fit(pool_gib: float, per_request_bytes: float) -> int:
    if per_request_bytes <= 0:
        raise ValueError("per-request bytes must be positive")
    return int(pool_gib * 1024**3 // per_request_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--bytes-per-element",
        type=float,
        default=2,
        help="2 for FP16/BF16, 1 for idealized FP8 payload",
    )
    parser.add_argument("--kv-pool-gib", type=float, default=16)
    args = parser.parse_args()

    if not 1 <= args.kv_heads <= args.query_heads:
        raise ValueError("kv-heads must be between 1 and query-heads")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query-heads must be divisible by kv-heads for standard GQA")

    variants = {
        "MHA": ModelShape(
            args.layers,
            args.query_heads,
            args.query_heads,
            args.head_dim,
            args.bytes_per_element,
        ),
        f"GQA-{args.kv_heads}": ModelShape(
            args.layers,
            args.query_heads,
            args.kv_heads,
            args.head_dim,
            args.bytes_per_element,
        ),
        "MQA": ModelShape(
            args.layers,
            args.query_heads,
            1,
            args.head_dim,
            args.bytes_per_element,
        ),
    }

    print(
        f"layers={args.layers}, query_heads={args.query_heads}, "
        f"head_dim={args.head_dim}, tokens={args.tokens}, batch={args.batch}, "
        f"bytes/element={args.bytes_per_element:g}"
    )
    print()
    print(f"{'variant':<12} {'MiB/request':>14} {'GiB/batch':>12} {'fits in pool':>13}")
    print("-" * 55)
    for name, shape in variants.items():
        one_request = shape.kv_bytes(args.tokens)
        batch_bytes = shape.kv_bytes(args.tokens, args.batch)
        fits = requests_that_fit(args.kv_pool_gib, one_request)
        print(f"{name:<12} {mib(one_request):>14.2f} {gib(batch_bytes):>12.3f} {fits:>13}")

    per_token = variants[f"GQA-{args.kv_heads}"].kv_bytes(tokens=1)
    print()
    print(f"Selected GQA payload per sequence-token: {per_token:,.0f} bytes")
    print("Physical runtime memory will be larger than this logical payload.")


if __name__ == "__main__":
    main()
