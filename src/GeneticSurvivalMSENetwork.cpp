#include "GeneticSurvivalMSENetwork.h"
#include "GeneticNetwork.h"
#include <stdio.h>
#include "boost/random.hpp"
#include <cmath>

using namespace std;

GeneticNetwork*
GeneticSurvivalMSENetwork::getGeneticNetwork(GeneticNetwork &cloner,
                                             boost::variate_generator<boost::mt19937&,
                                             boost::normal_distribution<double> >
                                             &gaussian,
                                             boost::variate_generator<boost::mt19937&,
                                             boost::uniform_real<> >
                                             &uniform)
{
  GeneticSurvivalMSENetwork *net =
    new GeneticSurvivalMSENetwork(cloner.getNumOfInputs(),
                                  cloner.getNumOfHidden());
  net->initNodes();

  // First clone all the weights as initial values
  net->cloneNetworkSlow(cloner);

  // Then mutate the weights. Set halfpoint to irrelevant values
  net->mutateWeights(gaussian, uniform, cloner.weightMutationChance,
                     cloner.weightMutationFactor, 1, 0, !cloner.getResume());

  return net;
}

GeneticSurvivalMSENetwork::GeneticSurvivalMSENetwork(const unsigned int numOfInputs,
                                                     const unsigned int numOfHidden) :
  GeneticNetwork(numOfInputs, numOfHidden, 1) {
}

// Calculate censored MSE
double censMSE(const double * const outputs,
               const double * const targets,
               const unsigned int length)
{
  double sum = 0, q, time, event, output;
  unsigned int n;

  for (n = 0; n < length; n++) {
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
  return sum / (double) length;
}

double GeneticSurvivalMSENetwork::
evaluateNetwork(GeneticNetwork &net,
                const double * const X,
                const double * const Y,
				const unsigned int length, double * const outputs) {
  // Evaluate each input set
  for (unsigned int i = 0; i < length; i++) {
    // Place output in correct position here
      net.output(X + i*net.getNumOfInputs(), outputs + i);
  }
  // Now calculate MSE
  double error = censMSE(outputs, Y, length);

  // Weight decay terms
  // Up to the user to (not) mix these
  // L2 weight decay
  if (decayL2 != 0) {
    error += decayL2 * weightSquaredSum(net);
  }

  // L1 weight decay
  if (decayL1 != 0) {
    error += decayL1 * weightAbsoluteSum(net);
  }

  // Weight elimination
  if (weightElimination != 0 &&
      weightEliminationLambda != 0) {
    error += weightElimination *
      weightEliminationSum(net, weightEliminationLambda);
  }

  return error;
}
