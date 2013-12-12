#!/usr/bin/env python
"""
General instructions:
  python setup.py build
  python setup.py install

To include parts that depend on R's survival module, do:
  python setup.py build --with-R

Info: This package depends on numpy, boost-random and optionally R, RInside
"""

from distutils.core import setup, Extension
import subprocess
import numpy
import sys


sources = ['src/PythonModule.cpp',
           'src/ErrorFunctions.cpp',
           'src/ErrorFunctionsGeneral.cpp',
           'src/ErrorFunctionsSurvival.cpp',
           'src/RPropNetworkWrapper.cpp',
           'src/RPropNetwork.cpp',
           'src/drand.cpp',
           'src/activationfunctions.cpp',
           'src/c_index.cpp', 'src/CIndexWrapper.cpp',
           'src/MatrixNetwork.cpp',
           'src/MatrixNetworkWrapper.cpp',
           'src/GeneticNetwork.cpp',
           'src/GeneticFitness.cpp',
           'src/GeneticSelection.cpp',
           'src/GeneticMutation.cpp',
           'src/GeneticCrossover.cpp',
           'src/GeneticNetworkWrapper.cpp',
           'src/Random.cpp',
           'src/global.cpp']


# Numpy stuff
numpy_include = numpy.get_include()
#linkargs = [] #['-Wl,--no-undefined', '-lboost_random-mt']
compileargs = []
libs=[]
libdirs=[]
linkargs, libs = [], []

if ("--help" in sys.argv or
    "-h" in sys.argv or
    len(sys.argv) == 1):
    sys.exit(__doc__)

# Python setup
_ann = Extension('ann._ann',
                 sources = sources,
                 include_dirs = [numpy_include],
                 extra_compile_args = ['-std=c++11',
                                       '-Wall', '-O2',
                                       '-pthread'] + compileargs,
                 extra_link_args = linkargs,
                 libraries=libs, library_dirs=libdirs)

setup(name = 'pysurvival-ann',
      version = '0.71',
      description = 'A C++ neural network package for survival data',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'None at this time',
      packages = ['ann'],
      package_dir = {'ann': 'ann'},
      ext_modules = [_ann],
      setup_requires = ['numpy'],
      install_requires = ['numpy>=1.7']
     )
