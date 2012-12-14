#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

numpy_include = numpy.get_include()
linkargs = [] #['-Wl,--no-undefined', '-lboost_random-mt']

_ann = Extension('ann._ann',
                 sources = ['src/PythonModule.cpp',
                            'src/RPropNetworkWrapper.cpp',
                            'src/RPropNetwork.cpp',
                            'src/FFNetworkWrapper.cpp',
                            'src/FFNeuron.cpp',
                            'src/FFNetwork.cpp',
                            'src/drand.cpp',
                            'src/activationfunctions.cpp',
                            'src/GeneticSurvivalNetwork.cpp',
                            'src/GeneticSurvivalNetworkWrapper.cpp',
                            'src/c_index.cpp', 'src/CIndexWrapper.cpp'],
                 include_dirs = [numpy_include],
                 extra_compile_args = ['-std=c++0x'],
                 extra_link_args = linkargs)

setup(name = 'AnnPlusPlus',
      version = '0.2',
      description = 'A c++ Neural network package',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'None at this time',
      packages = ['ann'],
      package_dir = {'ann': 'ann'},
      ext_modules = [_ann],
      requires = ['numpy'],
     )
