#!/usr/bin/env python3
"""CUDA-only lab: observe pageable/pinned HtoD copies and overlap.

Use PyTorch Profiler or run this process under Nsight Systems.
"""

from __future__ import annotations

import argparse
import contextlib

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--trace", default="cuda_transfer_trace.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this lab on an NVIDIA CUDA machine.")

    device = torch.device("cuda")
    elements = args.mib * 1024 * 1024 // 4
    pageable = torch.randn(elements, dtype=torch.float32)
    pinned = torch.randn(elements, dtype=torch.float32, pin_memory=True)
    compute_input = torch.randn(4096, 4096, device=device)
    copy_stream = torch.cuda.Stream()

    def run_copy(source: torch.Tensor, separate_stream: bool) -> None:
        stream_context = (
            torch.cuda.stream(copy_stream) if separate_stream else contextlib.nullcontext()
        )
        with stream_context:
            copied = source.to(device, non_blocking=True)
            copied.record_stream(copy_stream if separate_stream else torch.cuda.current_stream())
        with record_function("independent_compute"):
            _ = compute_input @ compute_input

    cases = (
        ("pageable_default", pageable, False),
        ("pinned_default", pinned, False),
        ("pageable_separate", pageable, True),
        ("pinned_separate", pinned, True),
    )

    for _, source, separate_stream in cases:
        run_copy(source, separate_stream)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(args.steps):
            for name, source, separate_stream in cases:
                with record_function(name), torch.cuda.nvtx.range(name):
                    run_copy(source, separate_stream)
                    torch.cuda.synchronize()
        torch.cuda.synchronize()

    prof.export_chrome_trace(args.trace)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print(f"Wrote {args.trace}")
    print("Compare all four pinning/stream combinations after warmup.")
    print("Only pinned + separate stream can overlap when a copy engine is free.")


if __name__ == "__main__":
    main()
