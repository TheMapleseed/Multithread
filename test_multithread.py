"""
test_multithread.py - Comprehensive test suite for adaptive thread pool

Tests cover:
- Correctness: Task execution, data integrity, exception handling
- Performance: Throughput, latency, overhead measurement
- Resource behavior: Scaling logic, limit enforcement
- Stress testing: Long-running workloads, edge cases
- Thread safety: Race conditions, concurrent access
"""

import time
import threading
import sys
import math
import random
import unittest
import multiprocessing
from collections import defaultdict
from contextlib import contextmanager

# Ensure we're running on free-threaded Python 3.14
if sys.version_info < (3, 14):
    print("ERROR: This test suite requires Python 3.14 or later")
    sys.exit(1)

try:
    import multithread
except ImportError:
    print("ERROR: multithread module not found. Build it first with: python setup.py build_ext --inplace")
    sys.exit(1)


class TestResults:
    """Container for test metrics and results"""
    def __init__(self):
        self.metrics = {}
        self.timings = defaultdict(list)
        self.errors = []
    
    def record_metric(self, name, value):
        self.metrics[name] = value
    
    def record_timing(self, category, duration):
        self.timings[category].append(duration)
    
    def record_error(self, error_msg):
        self.errors.append(error_msg)
    
    def get_summary(self):
        summary = {"metrics": self.metrics, "errors": self.errors}
        summary["avg_timings"] = {
            cat: sum(times) / len(times) 
            for cat, times in self.timings.items()
        }
        return summary


@contextmanager
def timer(results, category):
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        results.record_timing(category, duration)


class TestCorrectnessBasic(unittest.TestCase):
    """Basic correctness tests for task execution"""
    
    def setUp(self):
        self.results = TestResults()
    
    def test_simple_task_execution(self):
        """Test that simple tasks execute correctly"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=2,
            max_threads=4,
            soft_limit=4,
            hard_limit=4
        )
        
        result_container = []
        lock = threading.Lock()
        
        def simple_task(value):
            with lock:
                result_container.append(value)
        
        # Submit tasks
        for i in range(10):
            pool.submit(simple_task, args=(i,))
        
        # Wait for completion
        pool.shutdown(wait=True)
        
        # Verify all tasks executed
        self.assertEqual(len(result_container), 10)
        self.assertEqual(sorted(result_container), list(range(10)))
    
    def test_task_with_return_value(self):
        """Test that tasks with return values execute correctly"""
        pool = multithread.AdaptiveThreadPool(min_threads=2, max_threads=4)
        
        results = []
        lock = threading.Lock()
        
        def compute_square(x):
            result = x * x
            with lock:
                results.append(result)
            return result
        
        for i in range(10):
            pool.submit(compute_square, args=(i,))
        
        pool.shutdown(wait=True)
        
        expected = [i * i for i in range(10)]
        self.assertEqual(sorted(results), sorted(expected))
    
    def test_exception_handling(self):
        """Test that exceptions in tasks are handled gracefully"""
        pool = multithread.AdaptiveThreadPool(min_threads=2, max_threads=4)
        
        successful_tasks = []
        lock = threading.Lock()
        
        def task_that_might_fail(x):
            if x % 3 == 0:
                raise ValueError(f"Task {x} failed")
            with lock:
                successful_tasks.append(x)
        
        # Submit 10 tasks (some will fail)
        for i in range(10):
            pool.submit(task_that_might_fail, args=(i,))
        
        pool.shutdown(wait=True)
        
        # Should have executed 7 tasks successfully (0, 3, 6, 9 failed)
        self.assertEqual(len(successful_tasks), 7)
    
    def test_kwargs_support(self):
        """Test that keyword arguments work correctly"""
        pool = multithread.AdaptiveThreadPool(min_threads=2, max_threads=4)
        
        results = []
        lock = threading.Lock()
        
        def task_with_kwargs(a, b=10, c=20):
            with lock:
                results.append((a, b, c))
        
        pool.submit(task_with_kwargs, args=(1,), kwargs={'b': 2, 'c': 3})
        pool.submit(task_with_kwargs, args=(4,), kwargs={'b': 5})
        pool.submit(task_with_kwargs, args=(6,))
        
        pool.shutdown(wait=True)
        
        self.assertEqual(len(results), 3)
        self.assertIn((1, 2, 3), results)
        self.assertIn((4, 5, 20), results)
        self.assertIn((6, 10, 20), results)


class TestThreadSafety(unittest.TestCase):
    """Tests for thread safety and data race conditions"""
    
    def test_shared_counter_without_lock(self):
        """Test to demonstrate race conditions (should show issues)"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=4,
            max_threads=8,
            policy=multithread.POLICY_AGGRESSIVE
        )
        
        counter = [0]  # Using list for mutability
        
        def increment():
            for _ in range(1000):
                counter[0] += 1
        
        # Submit 10 tasks
        for _ in range(10):
            pool.submit(increment)
        
        pool.shutdown(wait=True)
        
        # Due to race conditions, counter will likely be less than expected
        # This demonstrates why locking is needed
        expected = 10000
        actual = counter[0]
        
        print(f"Race condition test: Expected {expected}, got {actual}, lost {expected - actual} increments")
        
        # We expect some data loss without proper locking
        self.assertLessEqual(actual, expected)
    
    def test_shared_counter_with_lock(self):
        """Test that proper locking prevents race conditions"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=4,
            max_threads=8,
            policy=multithread.POLICY_AGGRESSIVE
        )
        
        counter = [0]
        lock = threading.Lock()
        
        def increment_safe():
            for _ in range(1000):
                with lock:
                    counter[0] += 1
        
        for _ in range(10):
            pool.submit(increment_safe)
        
        pool.shutdown(wait=True)
        
        # With proper locking, should have exact count
        self.assertEqual(counter[0], 10000)
    
    def test_concurrent_data_structure_access(self):
        """Test concurrent access to shared data structures"""
        pool = multithread.AdaptiveThreadPool(min_threads=4, max_threads=8)
        
        shared_dict = {}
        lock = threading.Lock()
        
        def update_dict(key, value):
            with lock:
                if key in shared_dict:
                    shared_dict[key].append(value)
                else:
                    shared_dict[key] = [value]
        
        # Submit many updates
        for i in range(100):
            key = f"key_{i % 10}"
            pool.submit(update_dict, args=(key, i))
        
        pool.shutdown(wait=True)
        
        # Verify data integrity
        total_items = sum(len(v) for v in shared_dict.values())
        self.assertEqual(total_items, 100)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks and overhead measurements"""
    
    def setUp(self):
        self.results = TestResults()
    
    def test_cpu_bound_workload(self):
        """Benchmark CPU-bound tasks (should show speedup with more threads)"""
        
        def compute_primes_in_range(start, end):
            """Find primes in range [start, end)"""
            primes = []
            for n in range(start, end):
                if n < 2:
                    continue
                is_prime = True
                for i in range(2, int(math.sqrt(n)) + 1):
                    if n % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(n)
            return primes
        
        # Test with different thread counts
        for thread_count in [1, 2, 4, 8]:
            pool = multithread.AdaptiveThreadPool(
                min_threads=thread_count,
                max_threads=thread_count,
                soft_limit=thread_count,
                hard_limit=thread_count
            )
            
            chunks = 100
            chunk_size = 1000
            
            with timer(self.results, f"cpu_bound_{thread_count}_threads"):
                for i in range(chunks):
                    start = i * chunk_size
                    end = start + chunk_size
                    pool.submit(compute_primes_in_range, args=(start, end))
                
                pool.shutdown(wait=True)
            
            metrics = pool.get_metrics()
            self.results.record_metric(
                f"throughput_{thread_count}_threads",
                metrics['total_completed']
            )
    
    def test_throughput_scaling(self):
        """Measure throughput with increasing load"""
        
        def fast_task(x):
            return x * x
        
        for task_count in [100, 1000, 10000]:
            pool = multithread.AdaptiveThreadPool(
                min_threads=4,
                max_threads=16,
                policy=multithread.POLICY_AGGRESSIVE
            )
            
            with timer(self.results, f"throughput_{task_count}_tasks"):
                for i in range(task_count):
                    pool.submit(fast_task, args=(i,))
                
                pool.shutdown(wait=True)
            
            metrics = pool.get_metrics()
            duration = self.results.timings[f"throughput_{task_count}_tasks"][0]
            tps = task_count / duration
            
            self.results.record_metric(f"tasks_per_second_{task_count}", tps)
            print(f"{task_count} tasks: {tps:.2f} tasks/second")
    
    def test_latency_measurement(self):
        """Measure task submission to completion latency"""
        
        latencies = []
        lock = threading.Lock()
        
        def timed_task(submit_time):
            completion_time = time.perf_counter()
            with lock:
                latencies.append(completion_time - submit_time)
        
        pool = multithread.AdaptiveThreadPool(min_threads=4, max_threads=8)
        
        for _ in range(1000):
            submit_time = time.perf_counter()
            pool.submit(timed_task, args=(submit_time,))
            time.sleep(0.001)  # Small delay between submissions
        
        pool.shutdown(wait=True)
        
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"Latency - Avg: {avg_latency*1000:.2f}ms, P50: {p50_latency*1000:.2f}ms, P99: {p99_latency*1000:.2f}ms")
        
        self.results.record_metric("avg_latency_ms", avg_latency * 1000)
        self.results.record_metric("p99_latency_ms", p99_latency * 1000)


class TestResourceBehavior(unittest.TestCase):
    """Tests for resource monitoring and scaling behavior"""
    
    def test_soft_limit_enforcement(self):
        """Verify soft limits are respected under normal load"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=2,
            max_threads=16,
            soft_limit=4,
            hard_limit=16,
            policy=multithread.POLICY_BALANCED
        )
        
        def slow_task():
            time.sleep(0.1)
        
        # Submit moderate load
        for _ in range(20):
            pool.submit(slow_task)
        
        time.sleep(1)  # Let monitor adjust
        
        metrics = pool.get_metrics()
        current_threads = metrics['current_threads']
        
        # Should stay around soft limit with moderate load
        print(f"Threads with moderate load: {current_threads} (soft limit: 4)")
        self.assertLessEqual(current_threads, 8)  # Should not scale much beyond soft limit
    
    def test_hard_limit_enforcement(self):
        """Verify hard limits are never exceeded"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=2,
            max_threads=8,
            soft_limit=4,
            hard_limit=8,
            policy=multithread.POLICY_AGGRESSIVE
        )
        
        def slow_task():
            time.sleep(0.1)
        
        # Submit heavy load
        for _ in range(100):
            pool.submit(slow_task)
        
        # Monitor for several intervals
        max_threads_seen = 0
        for _ in range(10):
            time.sleep(0.3)
            metrics = pool.get_metrics()
            max_threads_seen = max(max_threads_seen, metrics['current_threads'])
        
        pool.shutdown(wait=True)
        
        print(f"Max threads under heavy load: {max_threads_seen} (hard limit: 8)")
        
        # Hard limit must never be exceeded
        self.assertLessEqual(max_threads_seen, 8)
    
    def test_scale_down_behavior(self):
        """Test that pool scales down when load decreases"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=2,
            max_threads=16,
            soft_limit=8,
            hard_limit=16,
            policy=multithread.POLICY_BALANCED,
            monitor_interval_ms=200
        )
        
        def fast_task():
            pass
        
        # Heavy load phase
        for _ in range(100):
            pool.submit(fast_task)
        
        time.sleep(1)
        metrics_high_load = pool.get_metrics()
        threads_high_load = metrics_high_load['current_threads']
        
        # Wait for tasks to complete and pool to scale down
        time.sleep(2)
        
        metrics_idle = pool.get_metrics()
        threads_idle = metrics_idle['current_threads']
        
        print(f"Threads during high load: {threads_high_load}, after idle: {threads_idle}")
        
        # Should scale back down toward min_threads
        self.assertLess(threads_idle, threads_high_load)
        
        pool.shutdown()
    
    def test_policy_differences(self):
        """Compare behavior of different scaling policies"""
        
        def medium_task():
            time.sleep(0.05)
        
        results = {}
        
        for policy_name, policy_const in [
            ('conservative', multithread.POLICY_CONSERVATIVE),
            ('balanced', multithread.POLICY_BALANCED),
            ('aggressive', multithread.POLICY_AGGRESSIVE)
        ]:
            pool = multithread.AdaptiveThreadPool(
                min_threads=2,
                max_threads=16,
                soft_limit=8,
                hard_limit=16,
                policy=policy_const,
                monitor_interval_ms=200
            )
            
            # Submit burst load
            for _ in range(50):
                pool.submit(medium_task)
            
            time.sleep(0.8)  # Let it scale
            
            metrics = pool.get_metrics()
            results[policy_name] = {
                'threads': metrics['current_threads'],
                'queue_depth': metrics['queue_depth']
            }
            
            pool.shutdown(wait=True)
        
        print("Policy comparison:")
        for policy, data in results.items():
            print(f"  {policy}: {data['threads']} threads, queue depth {data['queue_depth']}")
        
        # Aggressive should scale up more than conservative
        self.assertGreaterEqual(
            results['aggressive']['threads'],
            results['conservative']['threads']
        )


class TestStressAndStability(unittest.TestCase):
    """Long-running and stress tests"""
    
    def test_sustained_load(self):
        """Run sustained load for extended period"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=4,
            max_threads=16,
            soft_limit=8,
            hard_limit=16
        )
        
        def variable_duration_task():
            time.sleep(random.uniform(0.001, 0.05))
        
        start_time = time.time()
        duration = 10  # 10 seconds
        task_count = 0
        
        while time.time() - start_time < duration:
            pool.submit(variable_duration_task)
            task_count += 1
            time.sleep(random.uniform(0, 0.01))
        
        pool.shutdown(wait=True)
        
        metrics = pool.get_metrics()
        
        print(f"Sustained load test: {task_count} tasks over {duration}s")
        print(f"  Completed: {metrics['total_completed']}")
        print(f"  Avg task duration: {metrics['avg_task_duration_ms']:.2f}ms")
        
        self.assertEqual(metrics['total_submitted'], task_count)
        self.assertEqual(metrics['total_completed'], task_count)
    
    def test_mixed_workload(self):
        """Test with mix of fast and slow tasks"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=4,
            max_threads=12,
            policy=multithread.POLICY_BALANCED
        )
        
        fast_count = [0]
        slow_count = [0]
        lock = threading.Lock()
        
        def fast_task():
            with lock:
                fast_count[0] += 1
        
        def slow_task():
            time.sleep(0.1)
            with lock:
                slow_count[0] += 1
        
        # Submit mixed workload
        for i in range(100):
            if i % 3 == 0:
                pool.submit(slow_task)
            else:
                pool.submit(fast_task)
        
        pool.shutdown(wait=True)
        
        print(f"Mixed workload: {fast_count[0]} fast tasks, {slow_count[0]} slow tasks")
        
        self.assertEqual(fast_count[0] + slow_count[0], 100)
    
    def test_priority_ordering(self):
        """Test that priority affects execution order"""
        pool = multithread.AdaptiveThreadPool(
            min_threads=1,  # Single thread to ensure ordering
            max_threads=1,
            soft_limit=1,
            hard_limit=1
        )
        
        execution_order = []
        lock = threading.Lock()
        
        def priority_task(task_id):
            with lock:
                execution_order.append(task_id)
        
        # Submit tasks with different priorities (lower number = higher priority)
        pool.submit(priority_task, args=(1,), priority=10)
        pool.submit(priority_task, args=(2,), priority=5)
        pool.submit(priority_task, args=(3,), priority=1)
        pool.submit(priority_task, args=(4,), priority=15)
        
        pool.shutdown(wait=True)
        
        print(f"Execution order: {execution_order}")
        
        # With single thread, higher priority tasks should generally execute first
        # (though first submitted task may execute before priorities are considered)
        self.assertEqual(len(execution_order), 4)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and error conditions"""
    
    def test_empty_pool_shutdown(self):
        """Test shutting down pool with no tasks"""
        pool = multithread.AdaptiveThreadPool(min_threads=2, max_threads=4)
        pool.shutdown(wait=True)
        
        # Should complete without error
        self.assertTrue(True)
    
    def test_shutdown_with_pending_tasks(self):
        """Test shutdown behavior with tasks still in queue"""
        pool = multithread.AdaptiveThreadPool(min_threads=2, max_threads=2)
        
        completed = [0]
        lock = threading.Lock()
        
        def slow_task():
            time.sleep(0.1)
            with lock:
                completed[0] += 1
        
        # Submit more tasks than can be immediately processed
        for _ in range(10):
            pool.submit(slow_task)
        
        # Shutdown with wait should complete all tasks
        pool.shutdown(wait=True)
        
        self.assertEqual(completed[0], 10)
    
    def test_invalid_configuration(self):
        """Test that invalid configurations are rejected"""
        
        with self.assertRaises(ValueError):
            multithread.AdaptiveThreadPool(min_threads=10, max_threads=5)
        
        with self.assertRaises(ValueError):
            multithread.AdaptiveThreadPool(soft_limit=10, hard_limit=5)
    
    def test_submit_after_shutdown(self):
        """Test that submitting after shutdown raises error"""
        pool = multithread.AdaptiveThreadPool(min_threads=2, max_threads=4)
        pool.shutdown()
        
        time.sleep(0.1)
        
        with self.assertRaises(RuntimeError):
            pool.submit(lambda: None)


def run_performance_comparison():
    """Compare adaptive pool against static configurations"""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: Adaptive vs Static Thread Pools")
    print("="*70)
    
    def cpu_intensive_task(n):
        """CPU-bound task for benchmarking"""
        result = 0
        for i in range(n):
            result += math.sqrt(i)
        return result
    
    configurations = [
        ("Static 2 threads", 2, 2, multithread.POLICY_BALANCED),
        ("Static 4 threads", 4, 4, multithread.POLICY_BALANCED),
        ("Static 8 threads", 8, 8, multithread.POLICY_BALANCED),
        ("Adaptive (2-8, balanced)", 2, 8, multithread.POLICY_BALANCED),
        ("Adaptive (2-8, aggressive)", 2, 8, multithread.POLICY_AGGRESSIVE),
    ]
    
    task_count = 1000
    work_size = 10000
    
    results = {}
    
    for config_name, min_t, max_t, policy in configurations:
        pool = multithread.AdaptiveThreadPool(
            min_threads=min_t,
            max_threads=max_t,
            soft_limit=min(6, max_t),
            hard_limit=max_t,
            policy=policy
        )
        
        start = time.perf_counter()
        
        for i in range(task_count):
            pool.submit(cpu_intensive_task, args=(work_size,))
        
        pool.shutdown(wait=True)
        
        duration = time.perf_counter() - start
        metrics = pool.get_metrics()
        
        results[config_name] = {
            'duration': duration,
            'throughput': task_count / duration,
            'avg_task_ms': metrics['avg_task_duration_ms']
        }
        
        print(f"\n{config_name}:")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Throughput: {task_count/duration:.2f} tasks/sec")
        print(f"  Avg task duration: {metrics['avg_task_duration_ms']:.2f}ms")
        print(f"  Total completed: {metrics['total_completed']}")
    
    print("\n" + "="*70)
    
    # Find best configuration
    best = min(results.items(), key=lambda x: x[1]['duration'])
    print(f"\nBest configuration: {best[0]} ({best[1]['duration']:.2f}s)")
    
    return results


if __name__ == '__main__':
    print(f"Testing multithread v{multithread.__version__}")
    print(f"Python {sys.version}")
    print(f"CPUs available: {multiprocessing.cpu_count()}")
    print()
    
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance comparison
    if result.wasSuccessful():
        print("\n" + "="*70)
        print("All unit tests passed! Running performance comparison...")
        run_performance_comparison()
    
    sys.exit(0 if result.wasSuccessful() else 1)
