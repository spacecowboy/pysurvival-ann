#!/bin/bash
g++ -std=c++0x FFNeuron.cpp -o ann.o;valgrind ./ann.o
