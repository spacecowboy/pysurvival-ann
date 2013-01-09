#!/usr/bin/env python

from distutils.core import setup, Extension
import subprocess
import numpy

# Numpy stuff
numpy_include = numpy.get_include()
linkargs = [] #['-Wl,--no-undefined', '-lboost_random-mt']
compileargs = []

# R stuff
#rhome = subprocess.Popen(["R", "RHOME"],
#                         stdout=subprocess.PIPE).stdout.readline().strip()
rldflags = subprocess.Popen(["R", "CMD", "config", "--ldflags"],
                            stdout=subprocess.PIPE).stdout.readline().strip()
print(rldflags)
rcppflags = subprocess.Popen(["R", "CMD", "config", "--cppflags"],
                             stdout=subprocess.PIPE).stdout.readline().strip()
print(rcppflags)
rblas = subprocess.Popen(["R", "CMD", "config", "BLAS_LIBS"],
                         stdout=subprocess.PIPE).stdout.readline().strip()
print(rblas)
rlapack = subprocess.Popen(["R", "CMD", "config", "LAPACK_LIBS"],
                           stdout=subprocess.PIPE).stdout.readline().strip()
print(rlapack)
# Rcpp interface classes
p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)
rcppincl = p.communicate("Rcpp:::CxxFlags()")[0]
print(rcppincl)
p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)
rcpplibs = p.communicate("Rcpp:::LdFlags()")[0]
print(rcpplibs)
# Rinsid embedding classes
p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)
rinsideincl = p.communicate("RInside:::CxxFlags()")[0]
print(rinsideincl)
p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                     stdin=subprocess.PIPE)
rinsidelibs = p.communicate("RInside:::LdFlags()")[0]
print(rinsidelibs)
# compiler etc settings used in default make rules
#rcxx = subprocess.Popen(["R", "CMD", "config", "CXX"],
#                        stdout=subprocess.PIPE).stdout.readline().strip()
#rcppflags = subprocess.Popen(["R", "CMD", "config", "CPPFLAGS"],
#                        stdout=subprocess.PIPE).stdout.readline().strip()
#rcxxflags = subprocess.Popen(["R", "CMD", "config", "CXXFLAGS"],
#                        stdout=subprocess.PIPE).stdout.readline().strip()
#linkargs.extend()
compileargs.extend([rcppflags, rinsideincl, rcppincl])

# Python setup
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
                            'src/c_index.cpp', 'src/CIndexWrapper.cpp',
                            'src/CascadeNetwork.cpp',
                            'src/CascadeNetworkWrapper.cpp',
                            'src/rutil.cpp',
                            'src/CoxCascadeNetwork.cpp'],
                 include_dirs = [numpy_include],
                 extra_compile_args = ['-std=c++0x'] + compileargs,
                 extra_link_args = linkargs)

setup(name = 'AnnPlusPlus',
      version = '0.3',
      description = 'A c++ Neural network package',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'None at this time',
      packages = ['ann'],
      package_dir = {'ann': 'ann'},
      ext_modules = [_ann],
      requires = ['numpy'],
     )
