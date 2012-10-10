#!/bin/bash
g++ -std=c++0x -Wall -pedantic -I /usr/include/python2.7/ simple_test.cpp FFNeuron.cpp FFNetwork.cpp RPropNetwork.cpp GeneticSurvivalNetwork.cpp drand.cpp activationfunctions.cpp 
c_index.cpp -o ann.o

#valgrind --track-origins=yes --show-reachable=yes --leak-check=full ./ann.o

