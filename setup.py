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
           'src/RPropNetworkWrapper.cpp',
           'src/RPropNetwork.cpp',
           'src/FFNetworkWrapper.cpp',
           'src/FFNeuron.cpp',
           'src/FFNetwork.cpp',
           'src/drand.cpp',
           'src/activationfunctions.cpp',
           'src/c_index.cpp', 'src/CIndexWrapper.cpp',
           'src/CascadeNetwork.cpp',
           'src/CascadeNetworkWrapper.cpp',
           'src/MatrixNetwork.cpp',
           'src/MatrixNetworkWrapper.cpp',
#           'src/GeneticNetwork.cpp',
#          'src/GeneticFitness.cpp',
#          'src/GeneticNetworkWrapper.cpp',
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

if "--with-R" in sys.argv:
    sys.argv.remove("--with-R")

    sources += ['src/rutil.cpp',
                'src/CoxCascadeNetwork.cpp',
                'src/CoxCascadeNetworkWrapper.cpp']

    # R stuff
    #rhome = subprocess.Popen(["R", "RHOME"],
    #                         stdout=subprocess.PIPE).stdout.readline().strip()
    rldflags = subprocess.Popen(["R", "CMD", "config", "--ldflags"],
                                stdout=subprocess.PIPE).stdout.readline().strip()
    rlib, rmain = rldflags.split(" ")
    rlib = rlib[2:]
    rmain = rmain[2:]
    print(rldflags)
    print(rlib)
    print(rmain)
    rcppflags = subprocess.Popen(["R", "CMD", "config", "--cppflags"],
                                 stdout=subprocess.PIPE).stdout.readline().strip()
    rcppflags = rcppflags
    print(rcppflags)
    rblas = subprocess.Popen(["R", "CMD", "config", "BLAS_LIBS"],
                             stdout=subprocess.PIPE).stdout.readline().strip()
    rblas = rblas[2:]
    print(rblas)
    rlapack = subprocess.Popen(["R", "CMD", "config", "LAPACK_LIBS"],
                               stdout=subprocess.PIPE).stdout.readline().strip()
    rlapack = rlapack[2:]
    print(rlapack)
    # Rcpp interface classes
    p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    rcppincl = p.communicate("Rcpp:::CxxFlags()")[0]
    print(rcppincl)
    p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    rcpplibs = p.communicate("Rcpp:::LdFlags()")[0]
    #-L/home/jonas/R/x86_64-pc-linux-gnu-library/2.15/Rcpp/lib -lRcpp -Wl,-rpath
    #,/home/jonas/R/x86_64-pc-linux-gnu-library/2.15/Rcpp/lib

    rcpplib, rcpp, rcpplinkargs = rcpplibs.split(" ")

    rcpplib = rcpplib[2:]
    rcpp = rcpp[2:]
    print(rcpplibs)
    print(rcpplib)
    print(rcpp)
    print(rcpplinkargs)
    # Rinside embedding classes
    p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    rinsideincl = p.communicate("RInside:::CxxFlags()")[0]
    print(rinsideincl)
    p = subprocess.Popen(["R", "--vanilla", "--slave"], stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    rinsidelibs = p.communicate("RInside:::LdFlags()")[0]
    rinsidelib, rinside, rinsidelinkargs = rinsidelibs.split(" ")
    rinsidelib = rinsidelib[2:]
    rinside = rinside[2:]
    print(rinsidelibs)
    print(rinsidelib)
    print(rinside)
    print(rinsidelinkargs)
    # compiler etc settings used in default make rules
    #rcxx = subprocess.Popen(["R", "CMD", "config", "CXX"],
    #                        stdout=subprocess.PIPE).stdout.readline().strip()
    #rcppflags = subprocess.Popen(["R", "CMD", "config", "CPPFLAGS"],
    #                        stdout=subprocess.PIPE).stdout.readline().strip()
    #rcxxflags = subprocess.Popen(["R", "CMD", "config", "CXXFLAGS"],
    #                        stdout=subprocess.PIPE).stdout.readline().strip()
    #linkargs.extend([rldflags, rblas, rlapack, rcpplibs, rinsidelibs])
    libs=[rmain, rblas, rlapack, rcpp, rinside]
    libdirs=[rlib, rcpplib, rinsidelib]
    compileargs.extend([rcppflags, rinsideincl, rcppincl])
    linkargs = [rcpplinkargs, rinsidelinkargs]


# Python setup
_ann = Extension('ann._ann',
                 sources = sources,
                 include_dirs = [numpy_include],
                 extra_compile_args = ['-std=c++11',
                                       '-Wall', '-O2',
                                       '-pthread'] + compileargs,
                 extra_link_args = linkargs,
                 libraries=libs, library_dirs=libdirs)

setup(name = 'AnnPlusPlus',
      version = '4',
      description = 'A c++ Neural network package',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'None at this time',
      packages = ['ann'],
      package_dir = {'ann': 'ann'},
      ext_modules = [_ann],
      requires = ['numpy'],
     )
