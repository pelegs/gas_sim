#!/usr/bin/env python3

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
        ext_modules = cythonize('vecfuncs.pyx'),
        include_dirs=[np.get_include()]
)

setup(
        ext_modules = cythonize('gaslibc.pyx'),
        include_dirs=[np.get_include()]
)
