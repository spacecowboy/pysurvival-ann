/*
 * drand.cpp
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#include "drand.h"
#include <time.h>
#include <stdlib.h>

void setSeed() {
	srand(time(NULL));
}

double dRand() {
	double fMin = -1;
	double fMax = 1;
	double f = (double) rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}
