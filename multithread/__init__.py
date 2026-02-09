"""
Multithread: adaptive thread pool for Python 3.14 free-threading (no-GIL).

This package provides a C extension that implements a dynamic, self-adjusting
thread pool. Use AdaptiveThreadPool for CPU-bound parallelism with automatic
scaling based on load and system resources.

Public API:
    AdaptiveThreadPool  - Main pool class (submit, shutdown, get_metrics, get_config)
    POLICY_CONSERVATIVE - Scale up slowly, scale down quickly
    POLICY_BALANCED     - Moderate scaling (default)
    POLICY_AGGRESSIVE   - Scale up quickly, scale down slowly
    __version__         - Package version string
"""

from ._multithread import (
    AdaptiveThreadPool,
    POLICY_CONSERVATIVE,
    POLICY_BALANCED,
    POLICY_AGGRESSIVE,
    __version__,
)

__all__ = [
    "AdaptiveThreadPool",
    "POLICY_CONSERVATIVE",
    "POLICY_BALANCED",
    "POLICY_AGGRESSIVE",
    "__version__",
]
