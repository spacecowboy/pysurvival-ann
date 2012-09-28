#!/bin/bash
#First compile relevant files
g++ -I /usr/include/python2.7/ -std=c++0x -c -fPIC -Wall FFNeuron.cpp FFNetwork.cpp RPropNetwork.cpp GeneticSurvivalNetwork.cpp drand.cpp activationfunctions.cpp RPropWrapper.cpp
#g++ -std=c++0x -fPIC -Wall -lboost_python -lpython2.7 -I /usr/include/python2.7/ -o ann.so -shared FFNeuron.cpp FFNetwork.cpp RPropNetwork.cpp GeneticSurvivalNetwork.cpp drand.cpp activationfunctions.cpp RPropWrapper.cpp
#Next make a share library
g++ -shared -Wl,-soname,ann.so -o ann.so FFNeuron.o FFNetwork.o RPropNetwork.o GeneticSurvivalNetwork.o drand.o activationfunctions.o RPropWrapper.o -lpython2.7 -lboost_python
#g++ -shared -o ann.so FFNeuron.o FFNetwork.o RPropNetwork.o GeneticSurvivalNetwork.o drand.o activationfunctions.o RPropWrapper.o
