# Makefile

### Compiler and flags ###
CPPFLAGS = -g -Wall -Werror -std=c++11 -pthread -O0 # -fsanitize=thread
LDFLAGS = -lm

CLANG = clang++ -std=c++11 -Werror -Wall -pthread# -fsyntax-only

#CPP = g++
#CC = $(CPP) $(CPPFLAGS)
CPP = $(CLANG)
CC = $(CLANG)

#Used by Python build
PYCPP = gcc
PYFLAGS = -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -fPIC \
-I/usr/lib/python2.7/dist-packages/numpy/core/include \
-I/usr/include/python2.7 \
-std=c++0x -Wall -Werror -pthread

PYCC = $(PYCPP) $(PYFLAGS)

### List source files ###
SRC = MatrixNetwork.cpp test.cpp activationfunctions.cpp Random.cpp \
GeneticSelection.cpp global.cpp GeneticNetwork.cpp GeneticFitness.cpp \
GeneticMutation.cpp GeneticCrossover.cpp c_index.cpp ErrorFunctions.cpp \
ErrorFunctionsGeneral.cpp ErrorFunctionsSurvival.cpp \
RPropNetwork.cpp

### And corresponding object files ###
OBJ = $(addsuffix .o, $(basename $(SRC)))

### SUFFIXX rule to compile files if not specified differently ###
.SUFFIXES: .cpp

.cpp.o:
	$(CC) -c $<

### Main task ###
valgrind: matrix
	valgrind --tool=memcheck --leak-check=yes --track-origins=yes ./matrix

helgrind: matrix
	valgrind --tool=helgrind --read-var-info=yes --log-file=helgrind.txt ./matrix

test: matrix
	./matrix

prep:
	rm -f ./matrix

### Clean task ###
clean:
	rm -f $(OBJ) matrix *.o

### Main test program is linked here###
matrix: $(OBJ) prep
	$(CPP) $(CPPFLAGS) -o $@ $(OBJ)

### Dependencies ###
#test.o: test.cpp

#MatrixNetwork.o: MatrixNetwork.hpp MatrixNetwork.cpp

#activationfunctions.o: activationfunctions.hpp activationfunctions.cpp

#Random.o: Random.hpp Random.cpp

#GeneticSelection.o: GeneticSelection.hpp GeneticSelection.cpp

#GeneticFitness.o: GeneticFitness.hpp GeneticFitness.cpp

#GeneticMutation.o: GeneticMutation.hpp GeneticMutation.cpp

#GeneticCrossover.o: GeneticCrossover.hpp GeneticCrossover.cpp

#global.o: global.hpp global.cpp

#GeneticNetwork.o: GeneticNetwork.cpp GeneticNetwork.hpp

#c_index.o: c_index.h c_index.cpp

#ErrorFunctions.o: ErrorFunctions.cpp ErrorFunctions.hpp

### Wrappers ###
MatrixNetworkWrapper.o: MatrixNetworkWrapper.hpp MatrixNetworkWrapper.cpp
	$(PYCC) -c $< -o $@

GeneticNetworkWrapper.o: GeneticNetworkWrapper.hpp GeneticNetworkWrapper.cpp
	$(PYCC) -c $< -o $@
