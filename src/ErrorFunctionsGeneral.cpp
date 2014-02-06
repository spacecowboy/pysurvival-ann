#include "ErrorFunctionsGeneral.hpp"
#include <cmath>


double SSEDeriv(double target, double output) {
  return output - target;
}

double SSE(double target, double output) {
  return std::pow(target - output, 2.0) / 2.0;
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 */
void errorMSE(const double * const Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const double * const outputs,
              double * const errors)
{
  unsigned int i, n;
  // Set all to zero first
  for (n = 0; n < numOfOutput; n++) {
    errors[n] = 0;
  }
  // Evaluate each input set
  // Average over all inputs
  for (i = 0; i < length; i++) {
    for (n = 0; n < numOfOutput; n++) {
      errors[n] += SSE(Y[numOfOutput * i + n],
                       outputs[numOfOutput * i + n])
        / ((double) length);
    }
  }
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 * result.size = numOfOutput
 */
void derivativeMSE(const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   double * const result)
{
  for (unsigned int i = 0; i < numOfOutput; i++) {
    result[i] = SSEDeriv(Y[index + i], outputs[index + i]);
  }
}
