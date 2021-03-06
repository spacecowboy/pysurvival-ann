/*
 * activationfunctions.cpp
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#include <math.h>
#include "activationfunctions.hpp"

double evaluateActFunction(ActivationFuncEnum func, double x) {
  double retval;
  switch (func) {
  case LOGSIG:
    retval = sigmoid(x);
    break;
  case TANH:
    retval = hyperbole(x);
    break;
  case SOFTMAX:
    // Handle in MatrixNetwork.output
    retval = x;
    break;
  case LINEAR:
  default:
    retval = linear(x);
    break;
  }
  return retval;
}

double evaluateActFuncDerivative(ActivationFuncEnum func, double y) {
  double retval;
  switch (func) {
  case LOGSIG:
    retval = sigmoidDeriv(y);
    break;
  case TANH:
    retval = hyperboleDeriv(y);
    break;
  case SOFTMAX:
    // Not implemented
    retval = 0;
    break;
  case LINEAR:
  default:
    retval = linearDeriv(y);
    break;
  }
  return retval;
}

ActivationFuncEnum getFuncFromNumber(int num) {
  switch (num) {
  case LOGSIG:
    return LOGSIG;
  case TANH:
    return TANH;
  case SOFTMAX:
    return SOFTMAX;
  case LINEAR:
  default:
    return LINEAR;
  }
}

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
