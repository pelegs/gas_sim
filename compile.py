from distutils.core import setup
from Cython.Build import cythonize

setup(
        ext_modules = cythonize('vecfuncs.pyx')
)

setup(
        ext_modules = cythonize('gaslibc.pyx')
)
