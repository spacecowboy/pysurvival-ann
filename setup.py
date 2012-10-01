#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include = numpy.get_include()

ann = Extension('ann',
          sources = ['src/PythonModule.cpp', 'src/FFNetworkWrapper.cpp', 'src/FFNeuron.cpp', 'src/FFNetwork.cpp', 'src/drand.cpp', 'src/activationfunctions.cpp'],
          include_dirs = [numpy_include],
          extra_compile_args = ['-std=c++0x'])

setup(name = 'aNeuralN',
      version = '0.1',
      description = 'A c++ Neural network package',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'None at this time',
      packages = ['ann'],
      package_dir = {'ann': 'src'},
      #ext_package = 'ann',
      ext_modules = [ann],
      requires = ['numpy'],
     )
