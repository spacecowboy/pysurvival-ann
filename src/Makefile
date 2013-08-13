# Makefile

CPP = g++
CPPFLAGS = -g -Wall -std=c++0x -O2 -pthread
LDFLAGS = -lm

matrix: MatrixNetwork.o test.o activationfunctions.o Random.o
	$(CPP) $(CPPFLAGS) -o matrix MatrixNetwork.o test.o Random.o activationfunctions.o

test: matrix
	./matrix

test.o: test.cpp
	$(CPP) $(CPPFLAGS) -c test.cpp

MatrixNetwork.o: MatrixNetwork.h MatrixNetwork.cpp
	$(CPP) $(CPPFLAGS) -c MatrixNetwork.cpp

activationfunctions.o: activationfunctions.h activationfunctions.cpp
	$(CPP) $(CPPFLAGS) -c activationfunctions.cpp

Random.o: Random.h Random.cpp
	$(CPP) $(CPPFLAGS) -c Random.cpp

clean:
	rm -f MatrixNetwork.o matrix test.o activationfunctions.o
