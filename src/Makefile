# Makefile

CPP = g++
CPPFLAGS = -g -Wall -std=c++0x -O2 -pthread
LDFLAGS = -lm

### Test program ###
test: matrix
	./matrix

### Main test program is linked here###
matrix: MatrixNetwork.o test.o activationfunctions.o Random.o \
GeneticSelection.o global.o GeneticNetwork.o GeneticFitness.o \
GeneticMutation.o GeneticCrossover.o c_index.o ErrorFunctions.o
#### Build command ####
	$(CPP) $(CPPFLAGS) -o matrix MatrixNetwork.o test.o Random.o \
GeneticSelection.o activationfunctions.o global.o GeneticNetwork.o \
GeneticFitness.o GeneticMutation.o GeneticCrossover.o \
c_index.o ErrorFunctions.o


### Dependencies ###
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

GeneticFitness.o: GeneticFitness.hpp GeneticFitness.cpp
	$(CPP) $(CPPFLAGS) -c GeneticFitness.cpp

GeneticMutation.o: GeneticMutation.hpp GeneticMutation.cpp
	$(CPP) $(CPPFLAGS) -c GeneticMutation.cpp

GeneticCrossover.o: GeneticCrossover.hpp GeneticCrossover.cpp
	$(CPP) $(CPPFLAGS) -c GeneticCrossover.cpp

global.o: global.hpp global.cpp
	$(CPP) $(CPPFLAGS) -c global.cpp

GeneticNetwork.o: GeneticNetwork.cpp GeneticNetwork.hpp
	$(CPP) $(CPPFLAGS) -c GeneticNetwork.cpp

c_index.o: c_index.h c_index.cpp
	$(CPP) $(CPPFLAGS) -c c_index.cpp

ErrorFunctions.o: ErrorFunctions.h ErrorFunctions.cpp
	$(CPP) $(CPPFLAGS) -c ErrorFunctions.cpp

clean:
	rm -f MatrixNetwork.o matrix test.o activationfunctions.o Random.o \
GeneticSelection.o global.o GeneticNetwork.o GeneticFitness.o \
GeneticMutation.o GeneticCrossover.o c_index.o ErrorFunctions.o
