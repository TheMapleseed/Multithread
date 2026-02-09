#!/usr/bin/env python3
"""
Validation: build, runtime test (scaling + shutdown), and optional full suite.

  python3.14t validate.py           # build + runtime test + full suite
  python3.14t validate.py --quick   # build + runtime test only
  python3.14t validate.py --no-build  # skip build

Exit 0 only if build succeeds and all run steps pass.
Runtime test confirms: import, pool, scaling, shutdown, post-shutdown reject.
"""

import os
import subprocess
import sys
import argparse


def run(cmd, env=None, timeout=300):
    """Run command; return (returncode, combined stdout+stderr)."""
    env = env or os.environ.copy()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)


def main():
    parser = argparse.ArgumentParser(description="Validate Multithread: build and test")
    parser.add_argument("--quick", action="store_true", help="Build + runtime test only")
    parser.add_argument("--no-build", action="store_true", help="Skip build")
    args = parser.parse_args()

    python = sys.executable
    root = os.path.dirname(os.path.abspath(__file__)) or "."
    os.chdir(root)

    # Environment
    code, _ = run([python, "-c", "import sys; sys.exit(0 if sys.version_info >= (3,14) else 1)"])
    if code != 0:
        print("FAIL: Python 3.14 or later required", file=sys.stderr)
        return 1

    # Build
    if not args.no_build:
        env = os.environ.copy()
        env.setdefault("CC", "clang")
        code, out = run([python, "setup.py", "build_ext", "--inplace"], env=env)
        if code != 0:
            print("FAIL: Build failed", file=sys.stderr)
            for line in (out or "").splitlines()[-40:]:
                print(line, file=sys.stderr)
            return 1
        print("PASS: Build")
    else:
        print("SKIP: Build")

    # Runtime test (scaling + shutdown)
    code, out = run([python, "runtime_test.py"])
    print(out or "")
    if code != 0:
        print("FAIL: Runtime test", file=sys.stderr)
        return 1
    print("PASS: Runtime test")

    # Full suite
    if not args.quick:
        code, out = run([python, "test_multithread.py"], timeout=120)
        if code != 0:
            print("FAIL: Full test suite", file=sys.stderr)
            if out:
                print(out[-3500:] if len(out) > 3500 else out, file=sys.stderr)
            return 1
        print("PASS: Full test suite")
    else:
        print("SKIP: Full test suite")

    print("Validation: all steps passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
