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

void getDerivative(ErrorFunction func,
                   const double * const target,
                   const double * const output,
                   const unsigned int length,
                   double * const result)
{
  switch (func) {
  case ErrorFunction::ERROR_SURV_MSE:
    derivativeSurvMSE(target, output, length, result);
    break;
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    derivativeSurvLikelihood(target, output, length, result);
    break;
  case ErrorFunction::ERROR_MSE:
  default:
    derivativeMSE(target, output, length, result);
    break;
  }
}


// /*
//  * Train using sumSquare
//  */
// double SSEDeriv(double target, double output) {
//   return target - output;
// }

// double *SSEDerivs(double *target, double *output, int length) {
//   double *derivs = new double[length];
//   for (int i = 0; i < length; i++) {
//     derivs[i] = target[i] - output[i];
//   }
//   return derivs;
// }

// double SSE(double target, double output) {
//   return std::pow(target - output, 2.0) / 2.0;
// }

// double *SSEs(double *target, double *output, int length) {
//   double *errors = new double[length];
//   for (int i = 0; i < length; i++) {
//     errors[i] = std::pow(target[i] - output[i], 2.0) / 2.0;
//   }
//   return errors;
// }
