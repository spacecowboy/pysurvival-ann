#include "GeneticFitness.hpp"
#include "ErrorFunctions.hpp"
#include "c_index.h"
#include "Statistics.hpp"
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
  double errors[numOfOutput * length];
  double avgErrors[numOfOutput];

  getAllErrors(errorfunc, Y, length, numOfOutput, outputs, errors);
  averagePatternError(errors, length, numOfOutput, avgErrors);

  // Negative sum of errors
  for (unsigned int n = 0; n < numOfOutput; n++) {
    fitness -= avgErrors[n];
  }

  return fitness;
}

int getExpectedTargetCount(const FitnessFunction func) {
  switch(func) {
  case FitnessFunction::FITNESS_SURV_LIKELIHOOD:
  case FitnessFunction::FITNESS_MSE_CENS:
  case FitnessFunction::FITNESS_CINDEX:
  case FitnessFunction::FITNESS_SURV_KAPLAN_MAX:
  case FitnessFunction::FITNESS_SURV_KAPLAN_MIN:
  case FitnessFunction::FITNESS_TARONEWARE_MEAN:
  case FitnessFunction::FITNESS_TARONEWARE_HIGHLOW:
    // Expecting time and event
    return 2;
  default:
    // By default, expecting same as number of outputs
    return -1;
  }
}


double getFitness(const FitnessFunction func,
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
