#!/bin/bash
g++ -std=c++0x simple_test.cpp FFNeuron.cpp FFNetwork.cpp RPropNetwork.cpp GeneticSurvivalNetwork.cpp drand.cpp activationfunctions.cpp -o ann.o;valgrind --track-origins=yes --leak-check=full ./ann.o
