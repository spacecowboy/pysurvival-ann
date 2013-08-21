#include "GeneticFitness.hpp"
#include "ErrorFunctions.h"
#include "c_index.h"
#include <math.h>


FitnessFunctionPtr getFitnessFunctionPtr(FitnessFunction val)
{
  FitnessFunctionPtr retval;
  switch(val) {
  case FitnessFunction::FITNESS_MSE_CENS:
    retval = &fitnessMSECens;
    break;
  case FitnessFunction::FITNESS_CINDEX:
    retval = &fitnessCIndex;
    break;
  case FitnessFunction::FITNESS_MSE:
  default:
    retval = &fitnessMSE;
    break;
  }
  return retval;
}

double getFitness(FitnessFunction func,
                  const double * const X,
                  const double * const Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const double * const outputs)
{
  double retval;
  switch(func) {
  case FitnessFunction::FITNESS_MSE_CENS:
    retval = fitnessMSECens(X, Y, length,
                            numOfOutput,
                            outputs);
    break;
  case FitnessFunction::FITNESS_CINDEX:
    retval = fitnessCIndex(X, Y, length,
                           numOfOutput,
                           outputs);
    break;
  case FitnessFunction::FITNESS_MSE:
  default:
    retval = fitnessMSE(X, Y, length,
                        numOfOutput,
                        outputs);
    break;
  }
  return retval;
}

double fitnessMSE(const double * const X,
                  const double * const Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const double * const outputs) {
  unsigned int i, n;
  double error = 0;

  // Evaluate each input set
  // Average over all inputs and number of outputs
  for (i = 0; i < length; i++) {
    // Place output in correct position here
    //net.output(X + i*net.getNumOfInputs(),
    //           outputs + net.getNumOfOutputs() * i);
    for (n = 0; n < numOfOutput; n++) {
      error += sqrt(SSE(Y[i * numOfOutput + n],
                        outputs[numOfOutput * i + n]))
        / ((double) length * numOfOutput);
    }
  }

  return -error;
}


// Returns the C-index of the network output
double fitnessCIndex(const double * const X,
                     const double * const Y,
                     const unsigned int length,
                     const unsigned int numOfOutput,
                     const double * const outputs)
{
  // Evaluate each input set
  //for (unsigned int i = 0; i < length; i++) {
    // Place output in correct position here
    //net.output(X + i*net.getNumOfInputs(), outputs + i);
    //}
  // Now calculate c-index, only one output supported
  return get_C_index(outputs, Y, length);
}

// Returns the MSE of the network output, giving credit
// for censored points that are over-estimated.
double fitnessMSECens(const double * const X,
                      const double * const Y,
                      const unsigned int length,
                      const unsigned int numOfOutput,
                      const double * const outputs)
{
  double sum = 0, q, time, event, output;
  unsigned int n;

  for (n = 0; n < length; n++) {
    // First evaluate the network
    //net.output(X + n*net.getNumOfInputs(), outputs + n);

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
