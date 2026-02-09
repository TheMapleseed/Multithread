"""
benchmark.py - Comprehensive performance benchmarking suite

This script provides detailed performance benchmarks for the adaptive thread pool,
comparing it against static configurations and measuring overhead, throughput,
latency, and scaling efficiency under various workload patterns.
"""

import time
import math
import sys
import threading
import multiprocessing
from dataclasses import dataclass
from typing import List, Dict, Callable
import statistics

try:
    import multithread
except ImportError:
    print("ERROR: multithread module not found.")
    print("Build it first with: python setup.py build_ext --inplace")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    duration: float
    tasks_completed: int
    throughput: float
    avg_latency: float
    p50_latency: float
    p99_latency: float
    thread_count: int
    cpu_utilization: float


class WorkloadGenerator:
    """Generates various types of workloads for benchmarking"""
    
    @staticmethod
    def cpu_bound_light(iterations: int = 10000) -> Callable:
        """Lightweight CPU-bound task"""
        def task():
            result = 0
            for i in range(iterations):
                result += math.sqrt(i)
            return result
        return task
    
    @staticmethod
    def cpu_bound_medium(iterations: int = 50000) -> Callable:
        """Medium CPU-bound task"""
        def task():
            result = 0
            for i in range(iterations):
                result += math.sqrt(i) + math.log(i + 1)
            return result
        return task
    
    @staticmethod
    def cpu_bound_heavy(iterations: int = 200000) -> Callable:
        """Heavy CPU-bound task"""
        def task():
            result = 0
            for i in range(iterations):
                result += math.sqrt(i) + math.log(i + 1) + math.sin(i)
            return result
        return task
    
    @staticmethod
    def mixed_workload() -> Callable:
        """Mixed CPU and short sleep"""
        def task():
            result = sum(math.sqrt(i) for i in range(5000))
            time.sleep(0.001)
            return result
        return task


class BenchmarkRunner:
    """Manages benchmark execution and result collection"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, name: str, pool_config: Dict, workload: Callable, 
                     task_count: int, monitor_latency: bool = False) -> BenchmarkResult:
        """Execute a single benchmark configuration"""
        
        print(f"\nRunning: {name}")
        print(f"  Configuration: {pool_config}")
        print(f"  Task count: {task_count}")
        
        pool = multithread.AdaptiveThreadPool(**pool_config)
        
        latencies = [] if monitor_latency else None
        lock = threading.Lock() if monitor_latency else None
        
        def wrapped_task():
            submit_time = time.perf_counter()
            result = workload()
            if monitor_latency:
                completion_time = time.perf_counter()
                with lock:
                    latencies.append(completion_time - submit_time)
            return result
        
        # Execute benchmark
        start_time = time.perf_counter()
        
        for _ in range(task_count):
            pool.submit(wrapped_task if monitor_latency else workload)
        
        pool.shutdown(wait=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Collect metrics
        metrics = pool.get_metrics()
        
        # Calculate latency statistics if monitoring
        avg_latency = 0.0
        p50_latency = 0.0
        p99_latency = 0.0
        
        if monitor_latency and latencies:
            avg_latency = statistics.mean(latencies)
            sorted_latencies = sorted(latencies)
            p50_latency = sorted_latencies[len(sorted_latencies) // 2]
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        result = BenchmarkResult(
            name=name,
            duration=duration,
            tasks_completed=metrics['total_completed'],
            throughput=task_count / duration,
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p99_latency=p99_latency,
            thread_count=metrics['current_threads'],
            cpu_utilization=metrics['cpu_utilization_percent']
        )
        
        self.results.append(result)
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {result.throughput:.2f} tasks/sec")
        if monitor_latency:
            print(f"  Avg latency: {avg_latency*1000:.2f}ms")
            print(f"  P99 latency: {p99_latency*1000:.2f}ms")
        
        return result
    
    def print_summary(self):
        """Print comprehensive summary of all benchmark results"""
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Benchmark':<40s} {'Duration':>10s} {'Throughput':>12s} "
              f"{'Threads':>8s} {'CPU%':>6s}")
        print("-" * 80)
        
        for result in self.results:
            print(f"{result.name:<40s} {result.duration:10.2f}s "
                  f"{result.throughput:12.2f} t/s {result.thread_count:8d} "
                  f"{result.cpu_utilization:6.1f}%")
        
        # Find best performer for each workload type
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        
        # Group results by workload type
        workload_groups = {}
        for result in self.results:
            workload_type = result.name.split('(')[0].strip()
            if workload_type not in workload_groups:
                workload_groups[workload_type] = []
            workload_groups[workload_type].append(result)
        
        for workload_type, group in workload_groups.items():
            best = min(group, key=lambda x: x.duration)
            print(f"\n{workload_type}:")
            print(f"  Best configuration: {best.name}")
            print(f"  Duration: {best.duration:.2f}s")
            print(f"  Throughput: {best.throughput:.2f} tasks/sec")
            
            # Show improvement over worst
            worst = max(group, key=lambda x: x.duration)
            if best != worst:
                improvement = ((worst.duration - best.duration) / worst.duration) * 100
                print(f"  Improvement over worst: {improvement:.1f}%")


def benchmark_static_vs_adaptive():
    """Compare static thread pools with adaptive configurations"""
    
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Static vs Adaptive Thread Pools")
    print("=" * 80)
    print("\nThis benchmark compares fixed thread counts with adaptive scaling")
    print("using medium CPU-bound tasks.")
    
    runner = BenchmarkRunner()
    workload = WorkloadGenerator.cpu_bound_medium()
    task_count = 500
    
    # Static configurations
    for thread_count in [2, 4, 8, 16]:
        runner.run_benchmark(
            f"Static {thread_count} threads",
            {
                'min_threads': thread_count,
                'max_threads': thread_count,
                'soft_limit': thread_count,
                'hard_limit': thread_count,
                'policy': multithread.POLICY_BALANCED
            },
            workload,
            task_count
        )
    
    # Adaptive configurations
    for policy_name, policy in [
        ('Conservative', multithread.POLICY_CONSERVATIVE),
        ('Balanced', multithread.POLICY_BALANCED),
        ('Aggressive', multithread.POLICY_AGGRESSIVE)
    ]:
        runner.run_benchmark(
            f"Adaptive (2-16, {policy_name})",
            {
                'min_threads': 2,
                'max_threads': 16,
                'soft_limit': 8,
                'hard_limit': 16,
                'policy': policy
            },
            workload,
            task_count
        )
    
    runner.print_summary()
    return runner.results


def benchmark_workload_types():
    """Benchmark different workload characteristics"""
    
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Different Workload Types")
    print("=" * 80)
    print("\nThis benchmark tests the adaptive pool with various workload patterns")
    print("to demonstrate how it handles different task characteristics.")
    
    runner = BenchmarkRunner()
    
    pool_config = {
        'min_threads': 4,
        'max_threads': 12,
        'soft_limit': 8,
        'hard_limit': 12,
        'policy': multithread.POLICY_BALANCED
    }
    
    # Light workload
    runner.run_benchmark(
        "Light CPU-bound (10k iterations)",
        pool_config,
        WorkloadGenerator.cpu_bound_light(),
        1000
    )
    
    # Medium workload
    runner.run_benchmark(
        "Medium CPU-bound (50k iterations)",
        pool_config,
        WorkloadGenerator.cpu_bound_medium(),
        500
    )
    
    # Heavy workload
    runner.run_benchmark(
        "Heavy CPU-bound (200k iterations)",
        pool_config,
        WorkloadGenerator.cpu_bound_heavy(),
        200
    )
    
    # Mixed workload
    runner.run_benchmark(
        "Mixed CPU + I/O",
        pool_config,
        WorkloadGenerator.mixed_workload(),
        500
    )
    
    runner.print_summary()
    return runner.results


def benchmark_scaling_efficiency():
    """Measure scaling efficiency with increasing load"""
    
    print("\n" + "=" * 80)
    print("BENCHMARK 3: Scaling Efficiency")
    print("=" * 80)
    print("\nThis benchmark measures how well the adaptive pool scales with")
    print("increasing task counts.")
    
    runner = BenchmarkRunner()
    workload = WorkloadGenerator.cpu_bound_medium()
    
    pool_config = {
        'min_threads': 2,
        'max_threads': multiprocessing.cpu_count(),
        'soft_limit': multiprocessing.cpu_count() // 2,
        'hard_limit': multiprocessing.cpu_count(),
        'policy': multithread.POLICY_AGGRESSIVE
    }
    
    for task_count in [100, 500, 1000, 2000]:
        runner.run_benchmark(
            f"Scaling test ({task_count} tasks)",
            pool_config,
            workload,
            task_count
        )
    
    runner.print_summary()
    
    # Calculate scaling efficiency
    print("\nScaling Analysis:")
    baseline = runner.results[0]
    for result in runner.results[1:]:
        expected_duration = baseline.duration * (result.tasks_completed / baseline.tasks_completed)
        actual_duration = result.duration
        efficiency = (expected_duration / actual_duration) * 100
        print(f"  {result.name}: {efficiency:.1f}% scaling efficiency")
    
    return runner.results


def benchmark_latency_characteristics():
    """Measure task submission to completion latency"""
    
    print("\n" + "=" * 80)
    print("BENCHMARK 4: Latency Characteristics")
    print("=" * 80)
    print("\nThis benchmark measures end-to-end latency from task submission")
    print("to completion under different load conditions.")
    
    runner = BenchmarkRunner()
    workload = WorkloadGenerator.cpu_bound_light()
    
    # Low load scenario
    runner.run_benchmark(
        "Low load latency (4 threads, 100 tasks)",
        {
            'min_threads': 4,
            'max_threads': 4,
            'soft_limit': 4,
            'hard_limit': 4,
            'policy': multithread.POLICY_BALANCED
        },
        workload,
        100,
        monitor_latency=True
    )
    
    # High load scenario
    runner.run_benchmark(
        "High load latency (4 threads, 1000 tasks)",
        {
            'min_threads': 4,
            'max_threads': 4,
            'soft_limit': 4,
            'hard_limit': 4,
            'policy': multithread.POLICY_BALANCED
        },
        workload,
        1000,
        monitor_latency=True
    )
    
    # Adaptive scaling scenario
    runner.run_benchmark(
        "Adaptive latency (2-12 threads, 1000 tasks)",
        {
            'min_threads': 2,
            'max_threads': 12,
            'soft_limit': 6,
            'hard_limit': 12,
            'policy': multithread.POLICY_AGGRESSIVE
        },
        workload,
        1000,
        monitor_latency=True
    )
    
    # Print detailed latency statistics
    print("\nLatency Statistics:")
    print(f"{'Benchmark':<45s} {'Avg':>10s} {'P50':>10s} {'P99':>10s}")
    print("-" * 80)
    
    for result in runner.results:
        if result.avg_latency > 0:
            print(f"{result.name:<45s} "
                  f"{result.avg_latency*1000:10.2f}ms "
                  f"{result.p50_latency*1000:10.2f}ms "
                  f"{result.p99_latency*1000:10.2f}ms")
    
    return runner.results


def benchmark_overhead():
    """Measure overhead of adaptive thread pool management"""
    
    print("\n" + "=" * 80)
    print("BENCHMARK 5: Overhead Measurement")
    print("=" * 80)
    print("\nThis benchmark compares the overhead of adaptive management")
    print("versus static configurations.")
    
    runner = BenchmarkRunner()
    
    # Trivial workload to isolate overhead
    def trivial_task():
        return 42
    
    task_count = 10000
    
    # Minimal static pool
    runner.run_benchmark(
        "Overhead: Static (4 threads)",
        {
            'min_threads': 4,
            'max_threads': 4,
            'soft_limit': 4,
            'hard_limit': 4,
            'policy': multithread.POLICY_BALANCED
        },
        trivial_task,
        task_count
    )
    
    # Adaptive pool
    runner.run_benchmark(
        "Overhead: Adaptive (2-8 threads)",
        {
            'min_threads': 2,
            'max_threads': 8,
            'soft_limit': 4,
            'hard_limit': 8,
            'policy': multithread.POLICY_BALANCED,
            'monitor_interval_ms': 250
        },
        trivial_task,
        task_count
    )
    
    # Calculate overhead percentage
    static_result = runner.results[0]
    adaptive_result = runner.results[1]
    
    overhead_percent = ((adaptive_result.duration - static_result.duration) / 
                       static_result.duration) * 100
    
    print(f"\nOverhead Analysis:")
    print(f"  Static pool duration: {static_result.duration:.4f}s")
    print(f"  Adaptive pool duration: {adaptive_result.duration:.4f}s")
    print(f"  Overhead: {overhead_percent:.2f}%")
    
    if overhead_percent < 5:
        print(f"  ✓ Overhead is acceptably low (<5%)")
    elif overhead_percent < 10:
        print(f"  ⚠ Overhead is moderate (5-10%)")
    else:
        print(f"  ✗ Overhead is high (>10%)")
    
    return runner.results


def main():
    """Run comprehensive benchmark suite"""
    
    print("=" * 80)
    print("ADAPTIVE THREAD POOL - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print(f"\nSystem Information:")
    print(f"  Python version: {sys.version}")
    print(f"  Module version: {multithread.__version__}")
    print(f"  CPU cores: {multiprocessing.cpu_count()}")
    
    print("\nThis benchmark suite will execute multiple tests to measure:")
    print("  1. Static vs adaptive pool performance")
    print("  2. Behavior across different workload types")
    print("  3. Scaling efficiency with increasing load")
    print("  4. Latency characteristics")
    print("  5. Management overhead")
    
    input("\nPress Enter to begin benchmarks...")
    
    try:
        all_results = []
        
        # Run all benchmark suites
        all_results.extend(benchmark_static_vs_adaptive())
        all_results.extend(benchmark_workload_types())
        all_results.extend(benchmark_scaling_efficiency())
        all_results.extend(benchmark_latency_characteristics())
        all_results.extend(benchmark_overhead())
        
        # Final summary
        print("\n" + "=" * 80)
        print("BENCHMARK SUITE COMPLETE")
        print("=" * 80)
        
        print(f"\nTotal benchmarks executed: {len(all_results)}")
        print("\nKey Findings:")
        
        # Find best overall configuration
        best_overall = min(all_results, key=lambda x: x.duration / x.tasks_completed)
        print(f"  Best overall performance: {best_overall.name}")
        print(f"  Throughput: {best_overall.throughput:.2f} tasks/sec")
        
        print("\nBenchmark suite completed successfully.")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
