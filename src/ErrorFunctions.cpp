#include "ErrorFunctions.hpp"
#include "ErrorFunctionsGeneral.hpp"
#include "ErrorFunctionsSurvival.hpp"
#include <cmath>

// Evaluate the specified function
double getError(ErrorFunction func,
                const double * const Y,
                const unsigned int length,
                const unsigned int numOfOutput,
                const double * const outputs)
{
  double retval;
  switch (func) {
  case ErrorFunction::ERROR_SURV_MSE:
    retval = errorSurvMSE(Y, length, numOfOutput, outputs);
    break;
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    retval = errorSurvLikelihood(Y, length, numOfOutput, outputs);
    break;
  case ErrorFunction::ERROR_MSE:
  default:
    retval = errorMSE(Y, length, numOfOutput, outputs);
    break;
  }
  return retval;
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 * result.size = length * numOfOutput
 */
void getDerivative(ErrorFunction func,
                   const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   double * const result)
{
  switch (func) {
  case ErrorFunction::ERROR_SURV_MSE:
    derivativeSurvMSE(Y, length, numOfOutput, outputs, index, result);
    break;
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    derivativeSurvLikelihood(Y, length, numOfOutput, outputs, index, result);
    break;
  case ErrorFunction::ERROR_MSE:
  default:
    derivativeMSE(Y, length, numOfOutput, outputs, index, result);
    break;
  }
}
