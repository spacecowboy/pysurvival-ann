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

test: test.py ann/_ann.so
	nosetests -v test.py
	#python test.py

ann/_ann.so: $(DEPS)
	python setup.py build_ext --inplace

install: $(DEPS)
	python setup.py install

build: $(DEPS)
	python setup.py build

clean:
	rm -f ann/_ann.so
