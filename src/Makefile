# Makefile

CPP = g++
CPPFLAGS = -g -Wall -std=c++0x -O2 -pthread
LDFLAGS = -lm

matrix: MatrixNetwork.o test.o activationfunctions.o Random.o GeneticSelection.o
	$(CPP) $(CPPFLAGS) -o matrix MatrixNetwork.o test.o Random.o \
GeneticSelection.o activationfunctions.o global.o

test: matrix
	./matrix

test.o: test.cpp
	$(CPP) $(CPPFLAGS) -c test.cpp

MatrixNetwork.o: MatrixNetwork.hpp MatrixNetwork.cpp
	$(CPP) $(CPPFLAGS) -c MatrixNetwork.cpp

activationfunctions.o: activationfunctions.hpp activationfunctions.cpp
	$(CPP) $(CPPFLAGS) -c activationfunctions.cpp

Random.o: Random.hpp Random.cpp
	$(CPP) $(CPPFLAGS) -c Random.cpp

GeneticSelection.o: GeneticSelection.hpp GeneticSelection.cpp
	$(CPP) $(CPPFLAGS) -c GeneticSelection.cpp

global.o: global.hpp global.cpp
	$(CPP) $(CPPFLAGS) -c global.cpp

clean:
	rm -f MatrixNetwork.o matrix test.o activationfunctions.o Random.o \
GeneticSelection.o global.o
