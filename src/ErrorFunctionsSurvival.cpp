#include "ErrorFunctionsSurvival.hpp"
#include <cmath>

double errorSurvMSE(const double * const Y,
                    const unsigned int length,
                    const unsigned int numOfOutput,
                    const double * const outputs)
{
  unsigned int i, n;
  double error = 0, time, event, output;
  // Evaluate each input set
  // Average over all inputs and number of outputs
  for (i = 0; i < length; i++) {
    for (n = 0; n < numOfOutput; n++) {
      // Plus two because there is an event column as well
      time = Y[2*n];
      event = Y[2*n + 1];
      // No event column in predictions
      output = outputs[n];
      if ((event == 0 && output < time) || event != 0) {
        // Censored event which we are underestimating
        // Or real event
        error += std::pow(output - time, 2.0) / 2.0;
      }
    }
  }
  return error / ((double) length * numOfOutput);
}

void derivativeSurvMSE(const double * const target,
                       const double * const output,
                       const unsigned int length,
                       double * const result)
{
  double time, event;
  for (unsigned int i = 0; i < length; i++) {
    event = target[2 * i + 1];
    time = target[2 * i];
    // Only for events or underestimated censored
    if ((event == 0 && output[i] < time) || event != 0) {
      result[i] = time - output[i];
    }
  }
}


double errorSurvLikelihood(const double * const Y,
                           const unsigned int length,
                           const unsigned int numOfOutput,
                           const double * const outputs)
{
  return 0;
}

void derivativeSurvLikelihood(const double * const target,
                              const double * const output,
                              const unsigned int length,
                              double * const result)
{
  // TODO
}
