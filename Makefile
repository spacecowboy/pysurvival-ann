#CPP = g++
#CPPFLAGS = -g -Wall -Werror -std=c++11 -pthread -O0 # -fsanitize=thread
#LDFLAGS = -lm
#CC = $(CPP) $(CPPFLAGS)

#SRC = src/MatrixNetwork.cpp src/test.cpp src/activationfunctions.cpp \
#src/Random.cpp \
#GeneticSelection.cpp global.cpp GeneticNetwork.cpp GeneticFitness.cpp \
#GeneticMutation.cpp GeneticCrossover.cpp c_index.cpp ErrorFunctions.cpp

#WRP = CascadeNetworkWrapper.cpp CascadeNetworkWrapper.h CIndexWrapper.cpp \
#CIndexWrapper.h CoxCascadeNetworkWrapper.cpp CoxCascadeNetworkWrapper.h \
#FFNetworkWrapper.cpp FFNetworkWrapper.h GeneticCascadeNetworkWrapper.cpp \
#GeneticCascadeNetworkWrapper.h GeneticNetworkWrapper.cpp \
#GeneticNetworkWrapper.hpp MatrixNetworkWrapper.cpp MatrixNetworkWrapper.hpp \
#RPropNetworkWrapper.cpp RPropNetworkWrapper.h

#DEPS = setup.py $(SRC) $(WRP)

DEPS = setup.py $(wildcard src/*.cpp) $(wildcard src/*.h*) $(wildcard ann/*.py)

perf: test.py ann/_ann.so $(DEPS)
	python -m cProfile -s cumulative test.py

test: test.py ann/_ann.so $(DEPS)
	nosetests -v -x -s test.py
	#python test.py

#.SUFFIXES: .cpp
#.cpp.o:
#	$(CC) -c $<

ann/__init__.py:
	CC=clang++ python setup.py build_ext --inplace

ann/_ann.so: $(DEPS)
	CC=clang++ python setup.py build_ext --inplace

install: $(DEPS)
	#CC=clang++ python setup.py install
	CC=clang++ pip install -r requirements.txt
	CC=clang++ pip install -e .

build: $(DEPS)
	CC=clang++ python setup.py build

clean:
	rm -f ann/_ann.so
	rm -rf build/
