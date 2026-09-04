# Inference Optimization Founder Handbook

This package is a six-week path from computer-architecture refresh to a defensible inference-optimization startup experiment.

## Files

- `handbook.md` - canonical technical text, equations, trade-offs, and citations
- `handbook.html` - interactive visual companion with animated diagrams and calculators
- `labs/` - runnable exercises for Apple MPS and optional NVIDIA CUDA
- `requirements-mps.txt` - local Apple Silicon environment
- `requirements-cuda.txt` - optional NVIDIA environment

## Start

1. Open `handbook.html` in a browser.
2. Use `handbook.md` when you need the precise explanation or source.
3. Complete the labs in numerical order.

Your machine is an Apple M2 Pro with 16 GB unified memory. The local track therefore uses PyTorch MPS. CUDA-only topics such as PCIe copies, pinned host memory, Nsight Systems, and Nsight Compute require an NVIDIA machine and are marked clearly.

Full Xcode is also required for Instruments/Metal traces. This machine currently has Command Line Tools only, so install Xcode before that lab.

## Learning Rule

Do not optimize from folklore.

1. Define the workload and SLO.
2. Measure a baseline.
3. Classify the bottleneck.
4. Change one variable.
5. Measure quality, latency, throughput, memory, and cost again.

## Environment

Local MPS:

```bash
python3 -m venv .venv-mps
source .venv-mps/bin/activate
python -m pip install --upgrade pip
python -m pip install -r inference_optimization_handbook/requirements-mps.txt
```

Optional NVIDIA CUDA:

```bash
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
python -m pip install -r inference_optimization_handbook/requirements-cuda.txt
```

Package versions are deliberately bounded but not locked to exact patch versions. Record exact versions with every benchmark.

## Accuracy Policy

- **Invariant** labels mean model math or general architecture principles.
- **Typical** labels mean common behavior that must be measured on the target workload.
- **Version-sensitive** labels mean runtime or hardware details that may change.
- Vendor speedups are examples, not promises for another workload.
- The references were checked on 2026-09-01. Recheck version-sensitive documentation before product decisions.
