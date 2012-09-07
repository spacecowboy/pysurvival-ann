/*
 * activationfunctions.h
 *
 *  Created on: 7 sep 2012
 *      Author: jonas
 */

#ifndef ACTIVATIONFUNCTIONS_H_
#define ACTIVATIONFUNCTIONS_H_

#include <math.h>

/*
 * Linear, y = x, derivative = 1
 */
double linear(double x) {
	return x;
}
double linearDeriv(double y) {
	return 1;
}

/*
 * Sigmoid, y = 1 / (1 + exp(-x)), deriv = y * (1 - y)
 */
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double sigmoidDeriv(double y) {
	return y * (1 - y);
}

/*
 * Hyperbole, y = tanh(x), deriv = (1 - y) * (1 + y)
 */
double hyperbole(double x) {
	return tanh(x);
}

double hyperboleDeriv(double y) {
	return (1 - y) * (1 + y);
}

#endif /* ACTIVATIONFUNCTIONS_H_ */
