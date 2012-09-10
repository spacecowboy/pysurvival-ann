#!/bin/bash
g++ -std=c++0x simple_test.cpp FFNeuron.cpp FFNetwork.cpp activationfunctions.cpp -o ann.o;valgrind --track-origins=yes ./ann.o
