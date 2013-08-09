#include "GeneticFitness.h"
#include "GeneticNetwork.h"
#include "c_index.h"

fitness_func getFitnessFunction(fitness_function_t)
{
  fitness_func retval;
  switch(val) {
  case FITNESS_MSE_CENS:
    retval = &fitness_mse_cens;
    break;
  case FITNESS_CINDEX:
  default:
    retval = &fitness_cindex;
    break;
  }
  return retval;
}
/*
fitness_func_t longToFitnessType(long val)
{
  fitness_func_t retval;
  switch(val) {
  case MSE_CENS:
    retval = MSE_CENS;
  }
  }*/

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
    time = targets[2 * n];
    event = targets[2 * n + 1];
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
