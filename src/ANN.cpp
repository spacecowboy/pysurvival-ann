//============================================================================
// Name        : ANN.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdio.h>

class ANN_rprop {
public:
	static void sayHi();
};

void ANN_rprop::sayHi() {
	printf("Heya!\n");
}

int main(int argc, char* argv[]) {
	printf("Hello, world\n");
	ANN_rprop::sayHi();
}
