#!/usr/bin/env python3
"""Toy token-budget scheduler for prefill/decode trade-offs.

This is a discrete-time model, not a production serving simulator. A request cannot
finish prefill and decode in the same tick. Prefill/decode tokens have equal toy cost.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
from dataclasses import dataclass, field


@dataclass
class Request:
    request_id: int
    arrival: int
    prompt: int
    output: int
    prefilled: int = 0
    decoded: int = 0
    first_token_tick: int | None = None
    completed_tick: int | None = None
    decode_ticks: list[int] = field(default_factory=list)

    @property
    def prefill_left(self) -> int:
        return self.prompt - self.prefilled

    @property
    def decode_left(self) -> int:
        return self.output - self.decoded


def make_workload(count: int, seed: int, max_interarrival: int) -> list[Request]:
    rng = random.Random(seed)
    requests = []
    arrival = 0
    for request_id in range(count):
        arrival += rng.randint(0, max_interarrival)
        prompt = max(16, min(4096, int(rng.lognormvariate(5.5, 1.0))))
        output = max(8, min(512, int(rng.lognormvariate(3.7, 0.8))))
        requests.append(Request(request_id, arrival, prompt, output))
    return requests


def run(
    source: list[Request], policy: str, token_budget: int, prefill_chunk: int
) -> list[Request]:
    requests = copy.deepcopy(source)
    tick = 0
    completed: list[Request] = []

    while len(completed) < len(requests):
        available = [
            req for req in requests if req.arrival <= tick and req.completed_tick is None
        ]
        budget = token_budget
        decoders = sorted(
            (req for req in available if req.prefill_left == 0 and req.decode_left > 0),
            key=lambda req: (req.arrival, req.request_id),
        )
        prefills = sorted(
            (req for req in available if req.prefill_left > 0),
            key=lambda req: (req.arrival, req.request_id),
        )

        def schedule_decodes(limit: int) -> int:
            used = 0
            for req in decoders:
                if used >= limit:
                    break
                req.decoded += 1
                req.decode_ticks.append(tick + 1)
                used += 1
                if req.first_token_tick is None:
                    req.first_token_tick = tick + 1
                if req.decode_left == 0:
                    req.completed_tick = tick + 1
                    completed.append(req)
            return used

        scheduled_prefills: set[int] = set()

        def schedule_prefills(limit: int, one_request: bool) -> int:
            used = 0
            for req in prefills:
                if req.request_id in scheduled_prefills:
                    continue
                if used >= limit:
                    break
                chunk = req.prefill_left if policy == "fcfs" else min(req.prefill_left, prefill_chunk)
                consumed = min(chunk, limit - used)
                req.prefilled += consumed
                used += consumed
                scheduled_prefills.add(req.request_id)
                if one_request or (policy == "fcfs" and req.prefill_left > 0):
                    break
            return used

        if policy == "decode-first":
            budget -= schedule_decodes(budget)
            budget -= schedule_prefills(budget, one_request=False)
        elif policy == "chunked":
            # One bounded prefill chunk admits new work; active decoders then run.
            used = schedule_prefills(min(budget, prefill_chunk), one_request=True)
            budget -= used
            budget -= schedule_decodes(budget)
            budget -= schedule_prefills(budget, one_request=False)
        else:
            # Strict request-level FCFS: only the oldest unfinished request advances.
            oldest = min(available, key=lambda req: (req.arrival, req.request_id), default=None)
            if oldest is not None and oldest.prefill_left > 0:
                consumed = min(oldest.prefill_left, budget)
                oldest.prefilled += consumed
                budget -= consumed
            elif oldest is not None and oldest.decode_left > 0 and budget > 0:
                oldest.decoded += 1
                oldest.decode_ticks.append(tick + 1)
                if oldest.first_token_tick is None:
                    oldest.first_token_tick = tick + 1
                if oldest.decode_left == 0:
                    oldest.completed_tick = tick + 1
                    completed.append(oldest)

        tick += 1
        if tick > 1_000_000:
            raise RuntimeError("simulation did not converge")

    return completed


def pct(values: list[int], p: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))
    return float(values[index])


def report(policy: str, completed: list[Request]) -> None:
    ttft = [req.first_token_tick - req.arrival for req in completed if req.first_token_tick]
    e2e = [req.completed_tick - req.arrival for req in completed if req.completed_tick]
    itl = [
        later - earlier
        for req in completed
        for earlier, later in zip(req.decode_ticks, req.decode_ticks[1:])
    ]
    makespan = max(req.completed_tick for req in completed if req.completed_tick)
    print(
        f"{policy:<14} "
        f"TTFT mean/p95={statistics.mean(ttft):6.1f}/{pct(ttft, 0.95):6.1f}  "
        f"ITL mean/p95={statistics.mean(itl):5.2f}/{pct(itl, 0.95):4.1f}  "
        f"E2E mean/p95={statistics.mean(e2e):6.1f}/{pct(e2e, 0.95):6.1f}  "
        f"makespan={makespan:5d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument(
        "--max-interarrival",
        type=int,
        default=2,
        help="maximum ticks between arrivals; use 0 for a burst",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    workload = make_workload(args.requests, args.seed, args.max_interarrival)
    print(
        f"requests={args.requests}, token_budget={args.token_budget}, "
        f"prefill_chunk={args.prefill_chunk}, max_interarrival={args.max_interarrival}"
    )
    for policy in ("fcfs", "chunked", "decode-first"):
        report(policy, run(workload, policy, args.token_budget, args.prefill_chunk))
    print()
    print("A tick ends after scheduled work; arrivals are eligible at tick start.")
    print("Real prefill and decode tokens do not have equal cost. Benchmark a real engine.")


if __name__ == "__main__":
    main()
