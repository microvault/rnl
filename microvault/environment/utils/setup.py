import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(ext_modules=cythonize("map2d.pyx"), include_dirs=[numpy.get_include()])
