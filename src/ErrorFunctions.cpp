#include "ErrorFunctions.h"
#include <cmath>

/*
 * Train using sumSquare
 */
double SSEDeriv(double target, double output) {
  return target - output;
}

double *SSEDerivs(double *target, double *output, int length) {
  double *derivs = new double[length];
  for (int i = 0; i < length; i++) {
    derivs[i] = target[i] - output[i];
  }
  return derivs;
}

double SSE(double target, double output) {
  return std::pow(target - output, 2.0) / 2.0;
}

double *SSEs(double *target, double *output, int length) {
  double *errors = new double[length];
  for (int i = 0; i < length; i++) {
    errors[i] = std::pow(target[i] - output[i], 2.0) / 2.0;
  }
  return errors;
}
