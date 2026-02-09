The adaptive_threadpool module provides a Python class called AdaptiveThreadPool that implements a dynamic, self-adjusting thread pool for task execution. It's not "static" in the sense that it automatically scales the number of worker threads up or down based on workload, system resources (like CPU utilization and queue depth), and the configured scaling policy. However, it does expose methods ("commands") for interacting with it, and it's highly configurable during initialization. There's no explicit "ruleset" API for custom rules (e.g., no way to define arbitrary conditional logic at runtime), but the scaling behavior can be tuned via parameters to approximate different rule-like behaviors.Initialization and ConfigurationYou create an instance of AdaptiveThreadPool with optional parameters to customize its behavior. These are passed to the constructor (e.g., pool = adaptive_threadpool.AdaptiveThreadPool(min_threads=4, ...)). Here's a breakdown of the configurable options:min_threads (int, default: 2): Minimum number of worker threads to maintain.
max_threads (int, default: 32): Maximum number of worker threads (also used as the default hard limit).
soft_limit (int, default: 8): Soft cap on threads; can be exceeded under high load but prefers to stay below this.
hard_limit (int, default: 32): Absolute maximum threads; cannot be exceeded.
policy (int, default: POLICY_BALANCED): Scaling strategy. Use one of the module constants:POLICY_CONSERVATIVE (0): Scales up slowly (e.g., +1 thread when load is high) but down quickly (e.g., -2 when idle).
POLICY_BALANCED (1): Moderate scaling in both directions based on queue depth and load.
POLICY_AGGRESSIVE (2): Scales up quickly (up to +4 threads) but down slowly (-1 when very idle).

monitor_interval_ms (int, default: 250): How often (in milliseconds) the monitoring thread checks metrics and decides to scale.
scale_up_threshold (float, default: 0.75): Load ratio (active threads / total threads) above which scaling up may trigger.
scale_down_threshold (float, default: 0.25): Load ratio below which scaling down may trigger.
cpu_threshold (float, default: 0.80): CPU utilization (0.0-1.0) above which scaling is paused to avoid overload.

These parameters allow you to "configure" the pool's automatic behavior to fit different workloads (e.g., set a conservative policy for resource-constrained environments). Once initialized, the configuration is fixed—you can't change it at runtime without creating a new pool instance.Available Methods (Commands)The pool operates automatically after initialization (e.g., submitting tasks triggers execution and potential scaling), but you can interact with it via these methods:submit(callable, args=None, kwargs=None, priority=UINT64_MAX): Submits a task (a callable Python function) to the queue for execution. args: Tuple of positional arguments for the callable.
kwargs: Dict of keyword arguments for the callable.
priority: Optional uint64 for priority queuing (lower number = higher priority; default is max value for lowest priority).
Returns None on success; raises RuntimeError if the pool is shut down or enqueue fails.
This is the primary way to "send commands" (i.e., work) to the pool.

shutdown(wait=True): Shuts down the pool.wait: If True (default), waits for the task queue to drain before returning.
After shutdown, no new tasks can be submitted.

get_metrics(): Returns a dict of current runtime metrics for monitoring:cpu_utilization_percent: System CPU usage (0-100).
memory_available_mb: Free system memory in MB.
active_threads: Number of busy worker threads.
idle_threads: Number of idle worker threads.
current_threads: Total worker threads.
queue_depth: Number of pending tasks.
avg_task_duration_ms: Average task execution time in ms (simple moving average).
total_submitted: Total tasks submitted since creation.
total_completed: Total tasks completed since creation.

get_config(): Returns a dict of the pool's configuration parameters (as set during init).

Usage Example

python:
import Multithread

# Create a pool with custom config (e.g., aggressive scaling for bursty workloads)
pool = adaptive_threadpool.AdaptiveThreadPool(
    min_threads=4,
    max_threads=64,
    policy=adaptive_threadpool.POLICY_AGGRESSIVE,
    scale_up_threshold=0.8
)

# Submit tasks (the pool will automatically scale based on load)
def my_task(x):
    print(f"Processing {x}")

for i in range(100):
    pool.submit(my_task, args=(i,))

# Check status
print(pool.get_metrics())

# Shut down when done
pool.shutdown()

The scaling is fully automatic and driven by internal monitoring—no manual "commands" to scale threads. If you need more custom rules, you'd have to extend the C code or implement logic in Python (e.g., monitor get_metrics() and create/destroy pools dynamically, though that's inefficient). If this doesn't match what you meant by "module," provide more context!

