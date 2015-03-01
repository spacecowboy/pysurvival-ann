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
void errorMSE(const std::vector<double> &Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const std::vector<double> &outputs,
              const unsigned int index,
              std::vector<double> &errors)
{
  unsigned int i;
  // Evaluate each output neuron
  for (i = 0; i < numOfOutput; i++)
  {
    errors.at(index + i) = SSE(Y.at(index + i), outputs.at(index + i));
  }
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 * result.size = numOfOutput
 * This is because RProp algorithm only needs derivative for current
 * output pattern.
 */
void derivativeMSE(const std::vector<double> &Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const std::vector<double> &outputs,
                   const unsigned int index,
                   std::vector<double>::iterator result)
{
  for (unsigned int i = 0; i < numOfOutput; i++)
  {
    *(result + i) = SSEDeriv(Y.at(index + i), outputs.at(index + i));
  }
}
