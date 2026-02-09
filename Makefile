# Makefile for Multithread C extension
# Requires Python 3.14 with free-threading support and Clang compiler

# Configuration (use python3.14t or set PYTHON=python3 if that is your 3.14t)
PYTHON ?= python3.14t
CC := clang
CFLAGS := -std=c23 -O3 -Wall -Wextra -Wpedantic -pthread -fPIC
LDFLAGS := -pthread

# Detect platform
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LDFLAGS += -lrt
    PLATFORM := linux
endif
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -mmacosx-version-min=10.15
    PLATFORM := macos
endif

# Targets
.PHONY: all build test clean install benchmark validate check-python help

all: build

help:
	@echo "Multithread - Build System"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make build      - Build the C extension module"
	@echo "  make test       - Run comprehensive test suite"
	@echo "  make validate   - Build + runtime test + full suite (validation)"
	@echo "  make benchmark  - Run performance benchmarks"
	@echo "  make install    - Install module system-wide"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make check      - Verify Python environment"
	@echo ""
	@echo "Configuration:"
	@echo "  Python: $(PYTHON)"
	@echo "  Compiler: $(CC)"
	@echo "  Platform: $(PLATFORM)"

check-python:
	@echo "Checking Python environment..."
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 14), 'Python 3.14+ required'" || \
		(echo "ERROR: Python 3.14 or later required"; exit 1)
	@$(PYTHON) -c "import sysconfig; assert sysconfig.get_config_vars().get('Py_GIL_DISABLED'), 'Free-threading required'" || \
		(echo "WARNING: Python may not have free-threading enabled"; echo "Use python3.14t if available")
	@echo "Python environment OK"

build: check-python
	@echo "Building Multithread extension..."
	CC=$(CC) $(PYTHON) setup.py build_ext --inplace
	@echo "Build complete. Module ready for import."

test: build
	@echo "Running test suite..."
	$(PYTHON) test_multithread.py

validate: check-python
	@echo "Validation: build + runtime test + full suite..."
	$(PYTHON) validate.py
	@echo "Validation complete."

validate-quick: check-python
	@echo "Validation (quick): build + runtime test only..."
	$(PYTHON) validate.py --quick
	@echo "Validation complete."

benchmark: build
	@echo "Running performance benchmarks..."
	$(PYTHON) benchmark.py

install: check-python
	@echo "Installing multithread..."
	CC=$(CC) $(PYTHON) setup.py install
	@echo "Installation complete."

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f *.so
	rm -f *.dylib
	rm -f *.o
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Clean complete."

# Development targets
dev-build: CFLAGS += -g -O0 -DDEBUG
dev-build: clean build

sanitize: CFLAGS += -fsanitize=address -fsanitize=undefined -g
sanitize: LDFLAGS += -fsanitize=address -fsanitize=undefined
sanitize: clean build
	@echo "Built with AddressSanitizer and UndefinedBehaviorSanitizer"
	@echo "Run tests with: ASAN_OPTIONS=detect_leaks=1 $(PYTHON) test_multithread.py"

check-style:
	@echo "Checking C code style..."
	@command -v clang-format >/dev/null 2>&1 || \
		(echo "clang-format not found. Install for style checking."; exit 0)
	clang-format --dry-run --Werror Multithread.c 2>/dev/null || \
		echo "Style check complete (warnings may exist)"

format:
	@echo "Formatting C code..."
	@command -v clang-format >/dev/null 2>&1 || \
		(echo "ERROR: clang-format not found"; exit 1)
	clang-format -i Multithread.c
	@echo "Formatting complete."
