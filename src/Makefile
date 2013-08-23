# Makefile

### Compilar and flags ###
CPP = g++
CPPFLAGS = -g -Wall -Werror -std=c++0x -O2 -pthread
LDFLAGS = -lm

### List source files ###
SRC = MatrixNetwork.cpp test.cpp activationfunctions.cpp Random.cpp \
GeneticSelection.cpp global.cpp GeneticNetwork.cpp GeneticFitness.cpp \
GeneticMutation.cpp GeneticCrossover.cpp c_index.cpp ErrorFunctions.cpp

### And corresponding object files ###
OBJ = $(addsuffix .o, $(basename $(SRC)))

### SUFFIXX rule to compile files ###
.SUFFIXES: .cpp

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $<

### Main task ###
valgrind: matrix
	valgrind --tool=memcheck --leak-check=yes --track-origins=yes ./matrix

helgrind: matrix
	valgrind --tool=helgrind --read-var-info=yes --log-file=helgrind.txt ./matrix

test: matrix
	./matrix

### Clean task ###
clean:
	rm -f $(OBJ) matrix

### Main test program is linked here###
matrix: $(OBJ)
	$(CPP) $(CPPFLAGS) -o $@ $(OBJ)

### Dependencies ###
test.o: test.cpp

MatrixNetwork.o: MatrixNetwork.hpp MatrixNetwork.cpp

activationfunctions.o: activationfunctions.hpp activationfunctions.cpp

Random.o: Random.hpp Random.cpp

GeneticSelection.o: GeneticSelection.hpp GeneticSelection.cpp

GeneticFitness.o: GeneticFitness.hpp GeneticFitness.cpp

GeneticMutation.o: GeneticMutation.hpp GeneticMutation.cpp

GeneticCrossover.o: GeneticCrossover.hpp GeneticCrossover.cpp

global.o: global.hpp global.cpp

GeneticNetwork.o: GeneticNetwork.cpp GeneticNetwork.hpp

c_index.o: c_index.h c_index.cpp

ErrorFunctions.o: ErrorFunctions.cpp ErrorFunctions.h
