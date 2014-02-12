#include "ErrorFunctionsGeneral.hpp"
#include <cmath>


double SSEDeriv(double target, double output)
{
  return output - target;
}

double SSE(double target, double output)
{
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
              const unsigned int index,
              double * const errors)
{
  unsigned int i;
  // Evaluate each output neuron
  for (i = 0; i < numOfOutput; i++)
  {
    errors[index + i] = SSE(Y[index + i], outputs[index + i]);
  }
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 * result.size = numOfOutput
 * This is because RProp algorithm only needs derivative for current
 * output pattern.
 */
void derivativeMSE(const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   double * const result)
{
  for (unsigned int i = 0; i < numOfOutput; i++)
  {
    result[i] = SSEDeriv(Y[index + i], outputs[index + i]);
  }
}
