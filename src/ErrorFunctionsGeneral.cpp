#include "ErrorFunctionsGeneral.hpp"
#include <cmath>


double SSEDeriv(double target, double output) {
  return target - output;
}

double SSE(double target, double output) {
  return std::pow(target - output, 2.0) / 2.0;
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 */
double errorMSE(const double * const Y,
                const unsigned int length,
                const unsigned int numOfOutput,
                const double * const outputs)
{
  unsigned int i, n;
  double error = 0;
  // Evaluate each input set
  // Average over all inputs and number of outputs
  for (i = 0; i < length; i++) {
    for (n = 0; n < numOfOutput; n++) {
      error += SSE(Y[numOfOutput * i + n],
                   outputs[numOfOutput * i + n]);
    }
  }

  return error / ((double) length * numOfOutput);
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
    result[i] = Y[index + i] - outputs[index + i];
  }
}
