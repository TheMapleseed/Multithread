"""
example_usage.py - Demonstration of adaptive thread pool features

This script demonstrates the key features of the adaptive thread pool module,
including basic task submission, priority handling, different scaling policies,
and real-time metrics monitoring. Run this script to see the pool in action
and understand how it adapts to varying workload conditions.
"""

import time
import math
import sys
import threading
from collections import defaultdict

try:
    import multithread
except ImportError:
    print("ERROR: multithread module not found.")
    print("Build it first with: python setup.py build_ext --inplace")
    sys.exit(1)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demonstrate_basic_usage():
    """Demonstrate basic task submission and execution"""
    print_section("BASIC USAGE")
    
    print("\nCreating an adaptive thread pool with default configuration...")
    pool = multithread.AdaptiveThreadPool(
        min_threads=2,
        max_threads=8,
        soft_limit=4,
        hard_limit=8
    )
    
    config = pool.get_config()
    print(f"Pool configuration: {config}")
    
    # Submit simple tasks
    print("\nSubmitting 10 simple computational tasks...")
    results = []
    lock = threading.Lock()
    
    def compute_factorial(n):
        result = math.factorial(n)
        with lock:
            results.append((n, result))
        return result
    
    for i in range(10, 20):
        pool.submit(compute_factorial, args=(i,))
    
    # Wait a moment and check progress
    time.sleep(0.5)
    metrics = pool.get_metrics()
    print(f"\nProgress: {metrics['total_completed']}/{metrics['total_submitted']} tasks completed")
    print(f"Active threads: {metrics['active_threads']}, Idle: {metrics['idle_threads']}")
    
    # Shutdown and verify completion
    pool.shutdown(wait=True)
    
    print(f"\nAll tasks completed. Total results: {len(results)}")
    print("Sample results:")
    for n, fact in sorted(results)[:3]:
        print(f"  {n}! = {fact}")


def demonstrate_priority_queue():
    """Demonstrate priority-based task execution"""
    print_section("PRIORITY QUEUE")
    
    print("\nCreating pool with single thread to show priority ordering...")
    pool = multithread.AdaptiveThreadPool(
        min_threads=1,
        max_threads=1,
        soft_limit=1,
        hard_limit=1
    )
    
    execution_order = []
    lock = threading.Lock()
    
    def priority_task(task_id, priority_level):
        time.sleep(0.05)  # Simulate work
        with lock:
            execution_order.append((task_id, priority_level))
    
    print("\nSubmitting tasks with different priorities:")
    print("  Task A - Priority 10 (low)")
    print("  Task B - Priority 5 (medium)")
    print("  Task C - Priority 1 (high)")
    print("  Task D - Priority 15 (very low)")
    
    pool.submit(priority_task, args=("A", 10), priority=10)
    pool.submit(priority_task, args=("B", 5), priority=5)
    pool.submit(priority_task, args=("C", 1), priority=1)
    pool.submit(priority_task, args=("D", 15), priority=15)
    
    pool.shutdown(wait=True)
    
    print("\nExecution order:")
    for task_id, priority in execution_order:
        print(f"  Task {task_id} (priority {priority})")


def demonstrate_scaling_behavior():
    """Demonstrate how the pool scales with load"""
    print_section("SCALING BEHAVIOR")
    
    print("\nCreating pool with aggressive scaling policy...")
    pool = multithread.AdaptiveThreadPool(
        min_threads=2,
        max_threads=12,
        soft_limit=4,
        hard_limit=12,
        policy=multithread.POLICY_AGGRESSIVE,
        monitor_interval_ms=200
    )
    
    def cpu_intensive_task(iterations):
        """Simulate CPU-bound work"""
        result = 0
        for i in range(iterations):
            result += math.sqrt(i)
        return result
    
    print("\nPhase 1: Light load (10 tasks)...")
    for i in range(10):
        pool.submit(cpu_intensive_task, args=(50000,))
    
    time.sleep(0.5)
    metrics1 = pool.get_metrics()
    print(f"  Threads: {metrics1['current_threads']}, Queue: {metrics1['queue_depth']}")
    
    print("\nPhase 2: Heavy load (100 tasks)...")
    for i in range(100):
        pool.submit(cpu_intensive_task, args=(50000,))
    
    time.sleep(1.0)
    metrics2 = pool.get_metrics()
    print(f"  Threads: {metrics2['current_threads']}, Queue: {metrics2['queue_depth']}")
    print(f"  CPU utilization: {metrics2['cpu_utilization_percent']}%")
    
    print("\nPhase 3: Load decreasing (waiting for completion)...")
    time.sleep(2.0)
    metrics3 = pool.get_metrics()
    print(f"  Threads: {metrics3['current_threads']}, Queue: {metrics3['queue_depth']}")
    
    pool.shutdown(wait=True)
    
    print("\nScaling demonstration complete.")
    print(f"Thread count progression: {metrics1['current_threads']} → "
          f"{metrics2['current_threads']} → {metrics3['current_threads']}")


def demonstrate_policy_comparison():
    """Compare different scaling policies"""
    print_section("POLICY COMPARISON")
    
    def medium_workload():
        """Medium-duration CPU task"""
        total = 0
        for i in range(100000):
            total += math.sqrt(i)
        return total
    
    policies = [
        ("Conservative", multithread.POLICY_CONSERVATIVE),
        ("Balanced", multithread.POLICY_BALANCED),
        ("Aggressive", multithread.POLICY_AGGRESSIVE)
    ]
    
    results = {}
    
    for policy_name, policy_const in policies:
        print(f"\nTesting {policy_name} policy...")
        
        pool = multithread.AdaptiveThreadPool(
            min_threads=2,
            max_threads=16,
            soft_limit=8,
            hard_limit=16,
            policy=policy_const,
            monitor_interval_ms=200
        )
        
        start_time = time.perf_counter()
        
        # Submit burst of tasks
        for i in range(50):
            pool.submit(medium_workload)
        
        # Let pool adapt
        time.sleep(0.8)
        
        metrics = pool.get_metrics()
        
        pool.shutdown(wait=True)
        
        duration = time.perf_counter() - start_time
        
        results[policy_name] = {
            'duration': duration,
            'threads': metrics['current_threads'],
            'throughput': 50 / duration
        }
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Peak threads: {metrics['current_threads']}")
        print(f"  Throughput: {50/duration:.2f} tasks/sec")
    
    print("\nPolicy comparison summary:")
    for policy_name, data in results.items():
        print(f"  {policy_name:12s}: {data['duration']:.2f}s, "
              f"{data['threads']:2d} threads, {data['throughput']:.2f} tasks/sec")


def demonstrate_metrics_monitoring():
    """Demonstrate real-time metrics monitoring"""
    print_section("REAL-TIME METRICS")
    
    print("\nCreating pool and monitoring metrics during execution...")
    
    pool = multithread.AdaptiveThreadPool(
        min_threads=4,
        max_threads=12,
        soft_limit=8,
        hard_limit=12,
        policy=multithread.POLICY_BALANCED
    )
    
    def variable_workload():
        """Task with variable duration"""
        import random
        iterations = random.randint(10000, 100000)
        result = 0
        for i in range(iterations):
            result += math.sqrt(i)
        return result
    
    print("\nSubmitting 100 tasks with variable duration...")
    for i in range(100):
        pool.submit(variable_workload)
    
    print("\nMonitoring metrics every 0.5 seconds:")
    print(f"{'Time':>6s} {'Threads':>8s} {'Active':>7s} {'Queue':>6s} "
          f"{'CPU%':>5s} {'Completed':>10s} {'Avg Task (ms)':>13s}")
    print("-" * 70)
    
    start_time = time.time()
    for _ in range(8):
        time.sleep(0.5)
        elapsed = time.time() - start_time
        metrics = pool.get_metrics()
        
        print(f"{elapsed:6.1f} {metrics['current_threads']:8d} "
              f"{metrics['active_threads']:7d} {metrics['queue_depth']:6d} "
              f"{metrics['cpu_utilization_percent']:5d} "
              f"{metrics['total_completed']:10d} "
              f"{metrics['avg_task_duration_ms']:13.2f}")
    
    pool.shutdown(wait=True)
    
    final_metrics = pool.get_metrics()
    print("\nFinal statistics:")
    print(f"  Total tasks submitted: {final_metrics['total_submitted']}")
    print(f"  Total tasks completed: {final_metrics['total_completed']}")
    print(f"  Average task duration: {final_metrics['avg_task_duration_ms']:.2f}ms")


def demonstrate_error_handling():
    """Demonstrate exception handling in tasks"""
    print_section("ERROR HANDLING")
    
    print("\nCreating pool and submitting tasks that may fail...")
    
    pool = multithread.AdaptiveThreadPool(
        min_threads=2,
        max_threads=4
    )
    
    successful_count = [0]
    failed_count = [0]
    lock = threading.Lock()
    
    def task_that_might_fail(task_id):
        """Task that fails on certain inputs"""
        if task_id % 3 == 0:
            raise ValueError(f"Task {task_id} intentionally failed")
        
        # Simulate work
        result = sum(i * i for i in range(1000))
        
        with lock:
            successful_count[0] += 1
        
        return result
    
    print("Submitting 15 tasks (every 3rd task will fail)...")
    for i in range(15):
        pool.submit(task_that_might_fail, args=(i,))
    
    pool.shutdown(wait=True)
    
    print(f"\nResults:")
    print(f"  Successful tasks: {successful_count[0]}")
    print(f"  Failed tasks: {15 - successful_count[0]}")
    print("  (Exceptions were caught and printed to stderr)")


def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("  Multithread - Feature Demonstration")
    print("  Python", sys.version)
    print("  Module version:", multithread.__version__)
    print("=" * 70)
    
    try:
        demonstrate_basic_usage()
        demonstrate_priority_queue()
        demonstrate_scaling_behavior()
        demonstrate_policy_comparison()
        demonstrate_metrics_monitoring()
        demonstrate_error_handling()
        
        print("\n" + "=" * 70)
        print("  All demonstrations completed successfully!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
