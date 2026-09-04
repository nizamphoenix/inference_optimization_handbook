#!/usr/bin/env python3
"""Demonstrate that cached autoregressive attention matches full recomputation.

This is intentionally small and uses NumPy so it runs without PyTorch.
It models a single causal self-attention head. It is an educational reference,
not an optimized kernel.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def full_last_token(
    x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = q[-1:] @ k.T / np.sqrt(q.shape[-1])
    out = softmax(scores) @ v
    return out, k, v


def cached_step(
    x_new: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_new = x_new @ wq
    k_new = x_new @ wk
    v_new = x_new @ wv
    k_cache = np.concatenate([k_cache, k_new], axis=0)
    v_cache = np.concatenate([v_cache, v_new], axis=0)
    scores = q_new @ k_cache.T / np.sqrt(q_new.shape[-1])
    out = softmax(scores) @ v_cache
    return out, k_cache, v_cache


def benchmark(fn, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    x = rng.normal(size=(args.tokens, args.dim)).astype(np.float32)
    x_new = rng.normal(size=(1, args.dim)).astype(np.float32)
    scale = 1 / np.sqrt(args.dim)
    wq = (rng.normal(size=(args.dim, args.dim)) * scale).astype(np.float32)
    wk = (rng.normal(size=(args.dim, args.dim)) * scale).astype(np.float32)
    wv = (rng.normal(size=(args.dim, args.dim)) * scale).astype(np.float32)

    _, k_cache, v_cache = full_last_token(x, wq, wk, wv)
    full_out, _, _ = full_last_token(np.concatenate([x, x_new]), wq, wk, wv)
    cached_out, _, _ = cached_step(x_new, k_cache, v_cache, wq, wk, wv)

    np.testing.assert_allclose(full_out, cached_out, rtol=2e-5, atol=2e-5)

    full_time = benchmark(
        lambda: full_last_token(np.concatenate([x, x_new]), wq, wk, wv),
        args.repeats,
    )
    cached_time = benchmark(
        lambda: cached_step(x_new, k_cache, v_cache, wq, wk, wv),
        args.repeats,
    )

    full_projection_rows = args.tokens + 1
    cached_projection_rows = 1
    print("Correctness: cached output matches full recomputation")
    print(f"Full path projects {full_projection_rows} token rows per step")
    print(f"Cached path projects {cached_projection_rows} new token row per step")
    print(f"Full recomputation: {full_time * 1e3:.3f} ms")
    print(f"Cached step:        {cached_time * 1e3:.3f} ms")
    print(f"Observed demo speedup: {full_time / cached_time:.2f}x")
    print()
    print("Important: cached attention still reads all prior K and V rows.")
    print("The cache removes repeated projections; it does not make attention constant-time.")


if __name__ == "__main__":
    main()
