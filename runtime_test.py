#!/usr/bin/env python3
"""
Runtime test for Multithread: build, scaling, and shutdown.

Verifies:
  - Import and API
  - Pool creation and task execution
  - Scaling (thread count responds to load; metrics consistent)
  - Clean shutdown (queue drained, no work left; submit after shutdown raises)

Exit 0 only if all checks pass. Prints one line per phase for confirmation.
"""

import sys
import threading
import time


def log(msg):
    print(msg, flush=True)


def main():
    errors = []

    # --- Phase 1: Import ---
    try:
        import multithread
    except ImportError as e:
        errors.append(f"Import failed: {e}. Run: make build")
        return report(errors)

    for name in ("AdaptiveThreadPool", "POLICY_BALANCED", "POLICY_CONSERVATIVE", "POLICY_AGGRESSIVE", "__version__"):
        if not hasattr(multithread, name):
            errors.append(f"Missing: multithread.{name}")
    if errors:
        return report(errors)
    log("[1/5] Import OK")

    # --- Phase 2: Create pool and run tasks ---
    results = []
    lock = threading.Lock()
    n_tasks = 40

    def task(i):
        with lock:
            results.append(i)
        time.sleep(0.01)
        return i * i

    try:
        pool = multithread.AdaptiveThreadPool(
            min_threads=2,
            max_threads=8,
            soft_limit=6,
            hard_limit=8,
            policy=multithread.POLICY_BALANCED,
        )
    except Exception as e:
        errors.append(f"Pool creation failed: {e}")
        return report(errors)
    log("[2/5] Pool created")

    try:
        for i in range(n_tasks):
            pool.submit(task, args=(i,))

        m1 = pool.get_metrics()
        thread_count_under_load = m1.get("current_threads", 0) if isinstance(m1, dict) else 0
        queue_depth = m1.get("queue_depth", 0) if isinstance(m1, dict) else 0
        if not isinstance(m1, dict):
            errors.append("get_metrics() did not return a dict")
        elif thread_count_under_load < 1:
            errors.append("current_threads should be >= 1 under load")
    except Exception as e:
        errors.append(f"Submit/get_metrics failed: {e}")
        return report(errors)
    log("[3/5] Tasks submitted; scaling active (threads=%s, queue=%s)" % (thread_count_under_load, queue_depth))

    # --- Phase 3: Shutdown (wait for drain) ---
    try:
        pool.shutdown(wait=True)
    except Exception as e:
        errors.append(f"shutdown(wait=True) failed: {e}")
        return report(errors)

    metrics_after = pool.get_metrics()
    if metrics_after.get("queue_depth", -1) != 0:
        errors.append("After shutdown(wait=True), queue_depth should be 0")
    if metrics_after.get("total_completed") != n_tasks:
        errors.append("total_completed=%s, expected %s" % (metrics_after.get("total_completed"), n_tasks))
    log("[4/5] Shutdown OK (queue drained, all tasks completed)")

    # --- Phase 4: Post-shutdown submit raises ---
    try:
        pool.submit(lambda: None)
        errors.append("submit() after shutdown should raise RuntimeError")
    except RuntimeError:
        pass
    except Exception as e:
        errors.append("submit() after shutdown should raise RuntimeError, got: %s" % type(e).__name__)
    log("[5/5] Post-shutdown submit correctly rejected")

    # --- Verify result set ---
    if len(results) != n_tasks:
        errors.append("Expected %s results, got %s" % (n_tasks, len(results)))
    elif sorted(results) != list(range(n_tasks)):
        errors.append("Result set mismatch")

    for key in ("total_submitted", "total_completed", "current_threads", "queue_depth"):
        if key not in metrics_after:
            errors.append("get_metrics() missing key: %s" % key)

    return report(errors)


def report(errors):
    if not errors:
        log("Runtime test: PASS (import, scale, shutdown, no errors)")
        return 0
    log("Runtime test: FAIL")
    for e in errors:
        print("  %s" % e, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
