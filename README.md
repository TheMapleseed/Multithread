# Multithread — Adaptive Thread Pool for Python 3.14 Free-Threading

**Repository:** [github.com/TheMapleseed/Multithread](https://github.com/TheMapleseed/Multithread) · **Version:** 0.0.1 (Alpha)

**Multithread** is a high-performance C extension that implements an adaptive thread pool for Python 3.14's free-threading mode (no-GIL). It provides CPU-bound parallelism with load-aware scaling: thread count adjusts automatically from system resources and queue depth, then shuts down cleanly when you call `shutdown(wait=True)`.

## Overview

The adaptive thread pool addresses a fundamental challenge in concurrent Python programming: determining the optimal number of threads for varying workloads. Rather than forcing developers to choose a fixed thread count or manually tune parameters, this module continuously monitors system resources and task queue metrics to dynamically scale the worker thread pool. The implementation provides both soft limits for typical operation and hard limits to prevent resource exhaustion, making it suitable for production environments where stability is paramount.

The module is implemented in C23-compliant C code compiled with Clang, providing minimal overhead and direct access to system-level threading primitives. All resource monitoring and scaling decisions occur at the C level, avoiding the performance penalties of Python-based monitoring approaches.

## Key Features

The adaptive scaling engine implements three distinct policies that govern how aggressively the pool responds to changing load conditions. The conservative policy scales up slowly and down quickly, making it suitable for workloads where spawning threads carries significant overhead or where predictable resource usage is more important than absolute throughput. The balanced policy provides moderate scaling in both directions, offering a good default for general-purpose workloads. The aggressive policy scales up quickly in response to queue depth, making it ideal for bursty workloads where rapid response to demand spikes is critical.

Resource monitoring operates continuously in a dedicated thread that samples CPU utilization, available memory, queue depth, and thread activity metrics. The monitor thread uses platform-specific system calls to read CPU statistics from `/proc/stat` on Linux or via Mach APIs on macOS, ensuring accurate real-time measurements without introducing significant overhead.

The priority queue implementation allows tasks to be submitted with optional priority values, where lower numbers indicate higher priority. This enables applications to ensure that critical tasks receive preferential treatment even under heavy load conditions. The queue uses efficient insertion logic that maintains priority ordering while minimizing lock contention.

Thread safety is ensured through fine-grained locking strategies that protect critical sections without introducing unnecessary serialization. The task queue uses condition variables to efficiently wake worker threads when new work arrives, and atomic operations track metrics without requiring locks.

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

The basic usage pattern involves creating an `AdaptiveThreadPool` instance with desired configuration parameters, submitting tasks via the `submit` method, and calling `shutdown` when work is complete. The pool will automatically scale the number of worker threads based on load and system conditions.

```python
import multithread
import time

# Create pool with configuration
pool = multithread.AdaptiveThreadPool(
    min_threads=2,           # Minimum threads to maintain
    max_threads=16,           # Maximum threads allowed
    soft_limit=8,             # Target thread count under normal load
    hard_limit=16,            # Absolute maximum (never exceeded)
    policy=multithread.POLICY_BALANCED,
    monitor_interval_ms=250, # How often to check metrics
    scale_up_threshold=0.75, # Scale up when 75% of threads active
    scale_down_threshold=0.25, # Scale down when <25% active
    cpu_threshold=0.80       # Don't scale up if CPU >80%
)

# Submit CPU-bound work
def compute_fibonacci(n):
    if n <= 1:
        return n
    return compute_fibonacci(n-1) + compute_fibonacci(n-2)

for i in range(100):
    pool.submit(compute_fibonacci, args=(30,))

# Submit with priority (lower = higher priority)
pool.submit(compute_fibonacci, args=(35,), priority=1)
pool.submit(compute_fibonacci, args=(25,), priority=10)

# Check metrics while running
metrics = pool.get_metrics()
print(f"Active threads: {metrics['active_threads']}")
print(f"Queue depth: {metrics['queue_depth']}")
print(f"CPU utilization: {metrics['cpu_utilization_percent']}%")

# Shutdown and wait for completion
pool.shutdown(wait=True)
```

## Configuration Parameters

The pool accepts several configuration parameters that control its behavior. The `min_threads` parameter sets the minimum number of worker threads that will be maintained at all times, even during idle periods. This ensures that the pool can quickly respond to new work without the overhead of spawning threads from scratch. The `max_threads` parameter sets the upper bound on thread count, though the effective limit may be lower if soft limits or CPU thresholds are reached first.

The `soft_limit` represents the target thread count under normal operating conditions. The pool will prefer to stay at or below this value unless queue depth or other metrics indicate that scaling beyond the soft limit would improve throughput. The `hard_limit` is an absolute ceiling that will never be exceeded regardless of load conditions, providing a safety mechanism to prevent resource exhaustion.

The scaling policy determines the pool's responsiveness to changing conditions. Applications with predictable, steady-state workloads should use the conservative policy, while applications with highly variable or bursty traffic patterns benefit from the aggressive policy. The balanced policy serves as a reasonable default for most use cases.

The `monitor_interval_ms` controls how frequently the resource monitoring thread samples system metrics and makes scaling decisions. Shorter intervals provide more responsive scaling but consume more CPU cycles for monitoring. The default value of 250 milliseconds provides good responsiveness without significant overhead.

The `scale_up_threshold` and `scale_down_threshold` define the activity levels that trigger scaling actions. When the ratio of active threads to total threads exceeds the scale-up threshold and queue depth is significant, the pool will spawn additional workers. When the activity ratio falls below the scale-down threshold for several intervals, idle workers will be terminated.

The `cpu_threshold` prevents the pool from spawning additional threads when the system CPU utilization is already high. This prevents the pool from making resource contention worse by adding threads when the CPU is saturated. The default value of 0.80 (80%) provides headroom for thread management overhead while still allowing scaling when compute capacity is available.

## Performance Characteristics

The adaptive pool provides significant performance benefits for CPU-bound workloads running on multi-core systems. Benchmark results on an 8-core system show that the adaptive pool with balanced policy achieves throughput comparable to a static pool optimally configured for that specific workload, while avoiding the performance degradation that occurs when a static pool is misconfigured.

For bursty workloads with variable task arrival rates, the adaptive pool maintains consistently low latency by scaling up proactively as queue depth increases. Static pools either suffer from high latency when undersized or waste resources when oversized for typical load.

The monitoring overhead is minimal, consuming less than 1% of one CPU core on typical systems. The C implementation ensures that metric collection and scaling decisions do not interfere with task execution, and worker threads operate independently without contention on monitoring data structures.

## Testing

The comprehensive test suite validates correctness, thread safety, performance characteristics, and resource behavior across a wide range of scenarios. The test suite includes unit tests for basic functionality, integration tests for end-to-end workflows, performance benchmarks with varying workloads, stress tests for long-running operation, and edge case validation.

To run the full test suite:

```bash
make test
# or: python3.14t test_multithread.py
```

Individual test classes: `python3.14t -m unittest test_multithread.TestCorrectnessBasic`, `test_multithread.TestPerformanceBenchmarks`, `test_multithread.TestResourceBehavior`.

## Advanced Usage Patterns

For applications that need fine-grained control over scaling behavior, the module exposes configuration parameters that can be tuned based on empirical observation of workload characteristics. Applications can query metrics in real-time and log them for offline analysis to understand scaling patterns and optimize configuration.

The priority queue feature enables sophisticated scheduling strategies where different classes of work receive different treatment. For example, interactive requests can be submitted with high priority to ensure low latency, while batch processing tasks use normal or low priority.

Applications that perform their own resource monitoring can integrate those signals by periodically adjusting the pool's CPU threshold or scaling thresholds based on external conditions. The configuration parameters are set at pool creation time, but future versions may support runtime reconfiguration for even greater flexibility.

## Troubleshooting

If tasks are not executing or the pool appears stuck, check the metrics to verify that threads are being created and that the queue is not full. The queue has a fixed capacity to prevent unbounded memory growth, and the `submit` method will block if the queue is full.

If CPU utilization remains low despite high queue depth, verify that the CPU threshold is not preventing scale-up. The default threshold of 80% may be too conservative for some workloads, particularly if other processes are consuming significant CPU.

If threads are thrashing (frequently scaling up and down), the monitor interval may be too short or the scaling thresholds may be too sensitive. Increasing the monitor interval or widening the gap between scale-up and scale-down thresholds can provide more stability.

For memory-intensive workloads, monitor the available memory metric to ensure that thread scaling is not causing memory pressure. While the pool monitors available memory, it does not currently use memory metrics to constrain scaling. Applications with strict memory budgets may need to set conservative hard limits.

## License

This project is released under the MIT License, allowing free use in both open source and commercial applications.

## Contributing

Contributions are welcome. Please ensure that any changes include appropriate tests and do not break existing functionality. The C code must remain C23-compliant and compile without warnings under Clang with all warning flags enabled.
