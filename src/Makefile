# Makefile

### Compiler and flags ###
CPPFLAGS = -Wall -std=c++11 -pthread -O0 -g -fopenmp
LDFLAGS = -lm

CLANG = clang++ #-std=c++11 -Werror -Wall -pthread -fopenmp

CPP = g++
CC = $(CPP) $(CPPFLAGS)
#CPP = $(CLANG)
#CC = $(CLANG) $(CPPFLAGS)

#Used by Python build
PYCPP = gcc

PYFLAGS2 = -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -fPIC \
-I/usr/lib/python2.7/dist-packages/numpy/core/include \
-I/usr/include/python2.7 \
-std=c++0x -Wall -Werror -pthread

PYFLAGS3 = -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -fPIC \
-I/usr/lib/python3.3/site-packages/numpy/core/include \
-I/usr/include/python3.3m \
-std=c++0x -Wall -Werror -pthread

# Change this depending on your system
PYFLAGS = $(PYFLAGS3)

PYCC = $(PYCPP) $(PYFLAGS)

### List source files ###
SRC = MatrixNetwork.cpp test.cpp activationfunctions.cpp Random.cpp \
GeneticSelection.cpp global.cpp GeneticNetwork.cpp GeneticFitness.cpp \
GeneticMutation.cpp GeneticCrossover.cpp c_index.cpp ErrorFunctions.cpp \
ErrorFunctionsGeneral.cpp ErrorFunctionsSurvival.cpp \
RPropNetwork.cpp

### And corresponding object files ###
OBJ = $(addsuffix .o, $(basename $(SRC)))

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

# Pattern rules
# Object files depend on cpp and hpp files
%.o: %.cpp %.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

# Wrapper files compile with python
%Wrapper.o: %Wrapper.cpp %Wrapper.hpp
	$(PYCC) -c -o $@ $<
