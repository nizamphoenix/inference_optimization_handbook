#!/usr/bin/env python3
"""Profile a small cached Transformer-like workload on Apple MPS.

This avoids downloading model weights and makes prefill/decode shapes explicit.
It is an educational workload, not a text-generation benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
import torch.nn.functional as F


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden * 4, hidden),
        )

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, tokens, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, tokens, self.hidden)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        y = self.norm1(x)
        q = self._heads(self.q_proj(y))
        k = self._heads(self.k_proj(y))
        v = self._heads(self.v_proj(y))
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(self._merge(attention))
        x = x + self.ffn(self.norm2(x))
        return x, (k, v)

    def decode(
        self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        y = self.norm1(x)
        q = self._heads(self.q_proj(y))
        k_new = self._heads(self.k_proj(y))
        v_new = self._heads(self.v_proj(y))
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.o_proj(self._merge(attention))
        x = x + self.ffn(self.norm2(x))
        return x, (k, v)


def synchronize() -> None:
    torch.mps.synchronize()


def timed(fn, repeats: int) -> float:
    synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable. Use Apple Silicon with MPS-enabled PyTorch.")

    device = torch.device("mps")
    blocks = torch.nn.ModuleList(
        [TinyBlock(args.hidden, args.heads) for _ in range(args.layers)]
    ).to(device).eval()
    prompt = torch.randn(1, args.prompt, args.hidden, device=device)
    decode_token = torch.randn(1, 1, args.hidden, device=device)

    def prefill() -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = prompt
        caches = []
        with torch.inference_mode():
            for block in blocks:
                x, cache = block.prefill(x)
                caches.append(cache)
        return x, caches

    _, prompt_caches = prefill()

    def decode() -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = decode_token
        next_caches = []
        with torch.inference_mode():
            for block, cache in zip(blocks, prompt_caches, strict=True):
                x, next_cache = block.decode(x, cache)
                next_caches.append(next_cache)
        return x, next_caches

    for _ in range(3):
        prefill()
        decode()
    synchronize()

    prefill_ms = timed(prefill, args.repeats) * 1e3
    token_ms = timed(decode, args.repeats) * 1e3

    print(f"device: {device}")
    print(f"prefill shape: {tuple(prompt.shape)}, average: {prefill_ms:.3f} ms")
    print(
        f"cached decode: one query token over {args.prompt} cached positions, "
        f"average: {token_ms:.3f} ms"
    )
    print(f"current tensor memory: {torch.mps.current_allocated_memory() / 1024**2:.1f} MiB")
    print(f"driver memory: {torch.mps.driver_allocated_memory() / 1024**2:.1f} MiB")
    print(f"recommended max: {torch.mps.recommended_max_memory() / 1024**3:.2f} GiB")

    if args.profile:
        profiler = getattr(torch.mps, "profiler", None)
        profile_context = (
            profiler.profile(mode="interval,event", wait_until_completed=False)
            if profiler is not None
            else contextlib.nullcontext()
        )
        with profile_context:
            for _ in range(10):
                prefill()
                decode()
            synchronize()
        print("MPS signpost region complete. Record the process with Xcode Instruments.")

    print()
    print("Caveat: this block omits RoPE, GQA, paging, and optimized decode kernels.")
    print("It does use prompt KV state, unlike an isolated sequence-length-one test.")


if __name__ == "__main__":
    main()
