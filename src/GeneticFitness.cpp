#include "GeneticFitness.hpp"
#include "ErrorFunctions.hpp"
#include "c_index.h"
#include <math.h>

/**
 * Convert an error into a fitness. Just minus the sum of the errors.
 */
double errorToFitness(ErrorFunction errorfunc,
                      const double * const Y,
                      const unsigned int length,
                      const unsigned int numOfOutput,
                      const double * const outputs)
{
  double fitness = 0;
  double errors[numOfOutput];

  getError(errorfunc, Y, length, numOfOutput, outputs, errors);

  // Negative sum of errors
  for (int n = 0; n < numOfOutput; n++) {
    fitness -= errors[n];
  }

  return fitness;
}

double getFitness(FitnessFunction func,
                  const double * const Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const double * const outputs)
{
  double retval;
  switch(func) {
  case FitnessFunction::FITNESS_SURV_LIKELIHOOD:
    retval = errorToFitness(ErrorFunction::ERROR_SURV_LIKELIHOOD,
                            Y, length,
                            numOfOutput,
                            outputs);
    break;
  case FitnessFunction::FITNESS_MSE_CENS:
    retval = errorToFitness(ErrorFunction::ERROR_SURV_MSE,
                            Y, length,
                            numOfOutput,
                            outputs);
    break;
  case FitnessFunction::FITNESS_CINDEX:
    retval = fitnessCIndex(Y, length,
                           numOfOutput,
                           outputs);
    break;
  case FitnessFunction::FITNESS_MSE:
  default:
    retval = errorToFitness(ErrorFunction::ERROR_MSE,
                            Y, length,
                            numOfOutput,
                            outputs);
    break;
  }
  return retval;
}


// Returns the C-index of the network output
double fitnessCIndex(const double * const Y,
                     const unsigned int length,
                     const unsigned int numOfOutput,
                     const double * const outputs)
{
  return get_C_index(outputs, Y, length);
}
