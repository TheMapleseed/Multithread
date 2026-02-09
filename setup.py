"""
setup.py for Multithread C extension module
Requires Python 3.14 with free-threading support
"""

from setuptools import setup, Extension
import sys
import os
import sysconfig

# Verify Python version
if sys.version_info < (3, 14):
    raise RuntimeError("This module requires Python 3.14 or later")

# Check for free-threading support
config_vars = sysconfig.get_config_vars()
if not config_vars.get('Py_GIL_DISABLED'):
    print("WARNING: Python appears to be built without free-threading support.")
    print("This module is designed for free-threaded Python 3.14t")

# Clang and C23 only (all platforms)
os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'

extra_compile_args = [
    '-std=c23',
    '-O3',
    '-Wall', '-Wextra', '-Wpedantic',
    '-Wno-unused-parameter',
    '-pthread',
    '-fPIC',
]

extra_link_args = ['-pthread']

if sys.platform == 'darwin':
    extra_compile_args.append('-mmacosx-version-min=10.15')
elif sys.platform.startswith('linux'):
    extra_link_args.append('-lrt')

# C extension: installed as multithread._multithread (inside the package)
multithread_ext = Extension(
    'multithread._multithread',
    sources=['Multithread.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c',
)

setup(
    ext_modules=[multithread_ext],
    packages=['multithread'],
)
