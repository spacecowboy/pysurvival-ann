#include "GeneticFitness.hpp"
#include "GeneticNetwork.hpp"
#include "ErrorFunctions.h"
#include "c_index.h"
#include <math.h>


fitness_func_ptr getFitnessFunctionPtr(fitness_function_t val)
{
  fitness_func_ptr retval;
  switch(val) {
  case FITNESS_MSE_CENS:
    retval = &fitness_mse_cens;
    break;
  case FITNESS_CINDEX:
    retval = &fitness_cindex;
    break;
  case FITNESS_MSE:
  default:
    retval = &fitness_mse;
    break;
  }
  return retval;
}

double fitness_mse(GeneticNetwork &net,
                   const double * const X,
                   const double * const Y,
                   const unsigned int length,
                   double * const outputs) {
  unsigned int i, n;
  double error = 0;

  // Evaluate each input set
  // Average over all inputs and number of outputs
  for (i = 0; i < length; i++) {
    // Place output in correct position here
    net.output(X + i*net.getNumOfInputs(),
               outputs + net.getNumOfOutputs() * i);
    for (n = 0; n < net.getNumOfOutputs(); n++) {
      error += sqrt(SSE(Y[i * net.getNumOfOutputs() + n],
                        outputs[net.getNumOfOutputs() * i + n]))
        / ((double) length * net.getNumOfOutputs());
    }
  }

  return -error;
}


// Returns the C-index of the network output
double fitness_cindex(GeneticNetwork &net,
                      const double * const X,
                      const double * const Y,
                      const unsigned int length,
                      double * const outputs)
{
  // Evaluate each input set
  for (unsigned int i = 0; i < length; i++) {
    // Place output in correct position here
    net.output(X + i*net.getNumOfInputs(), outputs + i);
  }
  // Now calculate c-index
  return get_C_index(outputs, Y, length);
}

// Returns the MSE of the network output, giving credit
// for censored points that are over-estimated.
double fitness_mse_cens(GeneticNetwork &net,
                        const double * const X,
                        const double * const Y,
                        const unsigned int length,
                        double * const outputs)
{
  double sum = 0, q, time, event, output;
  unsigned int n;

  for (n = 0; n < length; n++) {
    // First evaluate the network
    net.output(X + n*net.getNumOfInputs(), outputs + n);

    // Relevant data for this evaluation
    time = Y[2 * n];
    event = Y[2 * n + 1];
    output = outputs[n];
    //   calc q, which penalizes under-estimation
    q = event;
    // if no event, check output
    if (q == 0 && output < time) {
      q = 1;
    }
    //   times (output - target)^2
    sum += q * pow(output - time, 2.0);
  }
  // divide by length
  // Return the negative of this to get the fitness
  return -(sum / (double) length);
}
