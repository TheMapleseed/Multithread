# Multithread — Adaptive Thread Pool for Python 3.14 Free-Threading

**Repository:** [github.com/TheMapleseed/Multithread](https://github.com/TheMapleseed/Multithread) · **Version:** 0.0.1 (Alpha)

**Multithread** is a high-performance C extension that implements an adaptive thread pool for Python 3.14's free-threading mode (no-GIL). It provides CPU-bound parallelism with load-aware scaling: thread count adjusts automatically from system resources and queue depth, then shuts down cleanly when you call `shutdown(wait=True)`.

## Overview

The pool scales worker threads up and down based on workload, CPU utilization, queue depth, and a configurable policy (conservative, balanced, aggressive). Resource monitoring runs in a dedicated C-level thread; scaling uses soft and hard limits so the pool stays within bounds.

## Key Features

- **Adaptive scaling:** Three policies (conservative, balanced, aggressive) control how quickly the pool scales up or down.
- **Priority queue:** Tasks can set a priority (lower number = higher priority).
- **Metrics:** `get_metrics()` returns CPU, memory, thread counts, queue depth, task counts, and average task duration.
- **Clean shutdown:** `shutdown(wait=True)` drains the queue and returns when all tasks are done; `submit()` after shutdown raises `RuntimeError`.

## Requirements

- **Python 3.14** with free-threading (no-GIL). Use the free-threaded build, often installed as `python3.14t`. You need **setuptools** for that interpreter (e.g. `python3.14t -m pip install setuptools wheel`) to build from source.
- **Clang** (C23). The project is built with Clang only.
- **Linux:** `/proc/stat` for CPU metrics. **macOS:** Mach APIs. Tested on x86_64 and ARM64.

## Installation

**From PyPI (when published):** `pip install multithread`

**From source (current instructions):**

1. Ensure Python 3.14 with free-threading and Clang are installed. If your only Python is the free-threaded build, use it as `python3.14t` or set `PYTHON=python3` when running `make`.
2. Clone and build:

```bash
git clone https://github.com/TheMapleseed/Multithread
cd Multithread

# Verify (optional)
python3.14t --version
python3.14t -c "import sys; print('Free-threading:', getattr(sys, '_is_gil_disabled', lambda: False)())"
clang --version

# Build and validate (recommended)
make validate
```

**Make targets:**

| Target | Description |
|--------|-------------|
| `make build` | Build the C extension in-place. |
| `make validate` | Build, then runtime test (scaling + shutdown), then full test suite. |
| `make validate-quick` | Build and runtime test only. |
| `make test` | Run full test suite (requires prior build). |
| `make install` | Install system-wide (optional). |
| `make clean` | Remove build artifacts. |

**Validation** runs the runtime test, which confirms: import, pool creation, task execution, scaling under load, clean shutdown (queue drained), and that `submit()` after shutdown raises. Direct commands: `python3.14t validate.py` or `python3.14t runtime_test.py` (after building).

**Other:** `make dev-build` (debug build), `make sanitize` (ASan/UBSan). Uninstall: `pip uninstall multithread`. Build problems: Clang 14+; Linux: `python3.14-dev` / `-devel`; macOS: `xcode-select --install`.

## Usage

```python
import multithread
import time

pool = multithread.AdaptiveThreadPool(
    min_threads=2,
    max_threads=16,
    soft_limit=8,
    hard_limit=16,
    policy=multithread.POLICY_BALANCED,
    monitor_interval_ms=250,
    scale_up_threshold=0.75,
    scale_down_threshold=0.25,
    cpu_threshold=0.80,
)

def compute_fibonacci(n):
    if n <= 1:
        return n
    return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)

for i in range(100):
    pool.submit(compute_fibonacci, args=(30,))

pool.submit(compute_fibonacci, args=(35,), priority=1)
pool.submit(compute_fibonacci, args=(25,), priority=10)

metrics = pool.get_metrics()
print(f"Active threads: {metrics['active_threads']}, Queue: {metrics['queue_depth']}")

pool.shutdown(wait=True)
```

## Configuration Parameters

- **min_threads / max_threads:** Bounds on worker count.
- **soft_limit / hard_limit:** Target and absolute cap; pool can exceed soft under load but never hard.
- **policy:** `multithread.POLICY_CONSERVATIVE`, `POLICY_BALANCED`, or `POLICY_AGGRESSIVE`.
- **monitor_interval_ms:** How often the monitor thread checks and may scale (default 250).
- **scale_up_threshold / scale_down_threshold:** Activity ratios that trigger scale up/down (defaults 0.75 / 0.25).
- **cpu_threshold:** Do not scale up if system CPU is above this (default 0.80).

## Testing

```bash
make test
# or: python3.14t test_multithread.py
```

Individual test classes: `python3.14t -m unittest test_multithread.TestCorrectnessBasic`, `test_multithread.TestPerformanceBenchmarks`, `test_multithread.TestResourceBehavior`.

## Troubleshooting

- **Tasks not running:** Check `get_metrics()` for thread count and queue depth; ensure the queue is not full.
- **Low CPU despite queue depth:** Consider lowering `cpu_threshold`.
- **Thread count thrashing:** Increase `monitor_interval_ms` or widen the gap between scale-up and scale-down thresholds.

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Contributions welcome. Keep the C code C23-compliant and the test suite passing.
