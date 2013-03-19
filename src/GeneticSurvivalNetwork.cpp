/*
 * GeneticNetwork.cpp
 *
 *  Created on: 11 sep 2012
 *      Author: jonas
 */

#include "GeneticSurvivalNetwork.h"
#include "GeneticNetwork.h"
//#include "activationfunctions.h"
#include "c_index.h"
//#include <vector>
#include <stdio.h>
#include "boost/random.hpp"
//#include <time.h>
//#include <math.h>

using namespace std;

GeneticNetwork*
GeneticSurvivalNetwork::getGeneticNetwork(GeneticNetwork *cloner,
                                          boost::variate_generator<boost::mt19937&,
                                          boost::normal_distribution<double> >* gaussian,
                                          boost::variate_generator<boost::mt19937&,
                                          boost::uniform_int<> > *uniform)
{
  GeneticSurvivalNetwork *net =
    new GeneticSurvivalNetwork(cloner->getNumOfInputs(),
                               cloner->getNumOfHidden());
  net->initNodes();

  // First clone all the weights as initial values
  net->cloneNetworkSlow(cloner);

  double dummy = 0;
  // Then mutate the weights. Set halfpoint to irrelevant values
  net->mutateWeights(gaussian, uniform, cloner->weightMutationChance,
                     cloner->weightMutationFactor, 1, 0, !cloner->getResume(),
                     &dummy, &dummy);

  return net;
}

GeneticSurvivalNetwork::GeneticSurvivalNetwork(unsigned int numOfInputs,
                                               unsigned int numOfHidden) :
  GeneticNetwork(numOfInputs, numOfHidden, 1) {
}

// Calculate the sum of all weights squared (L2 norm)
double weightSquaredSum(FFNetwork *net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net->getNumOfHidden(); n++) {
    // neuron weights
    sum += net->getHiddenNeuron(n)->getWeightsSquaredSum();
    numOfCons += net->getHiddenNeuron(n)->getNumOfConnections();
  }
  /*
    for (n = 0; n < net->getNumOfOutputs(); n++) {
    // Input weights
    sum += net->getOutputNeuron(n)->getWeightsSquaredSum();
    numOfCons += net->getOutputNeuron(n)->getNumOfConnections();
    }
  */
  //printf("Weight squared sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of absolute values of weights (L1 norm)
double weightAbsoluteSum(FFNetwork *net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net->getNumOfHidden(); n++) {
    // neuron weights
    sum += net->getHiddenNeuron(n)->getWeightsAbsoluteSum();
    numOfCons += net->getHiddenNeuron(n)->getNumOfConnections();
  }
  /*
    for (n = 0; n < net->getNumOfOutputs(); n++) {
    // Input weights
    sum += net->getOutputNeuron(n)->getWeightsAbsoluteSum();
    numOfCons += net->getOutputNeuron(n)->getNumOfConnections();
    }*/
  //printf("Weight absolute sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of soft weight elimination terms
double weightEliminationSum(FFNetwork *net, double lambda) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net->getNumOfHidden(); n++) {
    // neuron weights
    sum += net->getHiddenNeuron(n)->
      getWeightEliminationSum(lambda);
    numOfCons += net->getHiddenNeuron(n)->getNumOfConnections();
  }
  /*
    for (n = 0; n < net->getNumOfOutputs(); n++) {
    // Input weights
    sum += net->getOutputNeuron(n)->
    getWeightEliminationSum(lambda);
    numOfCons += net->getOutputNeuron(n)->getNumOfConnections();
    }*/
  //printf("Weight elimination sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

double GeneticSurvivalNetwork::
evaluateNetwork(GeneticNetwork *net, double *X, double *Y,
				unsigned int length, double *outputs) {
  // Evaluate each input set
  for (unsigned int i = 0; i < length; i++) {
    // Place output in correct position here
    net->output(X + i*net->getNumOfInputs(), outputs + i);
  }
  // Now calculate c-index
  double ci = get_C_index(outputs, Y, length);

  // Return the inverse since this returns the error of the network
  // If less than 0.001, return 1000 instead to avoid dividing by zero
  if (ci < 0.0000001)
    ci = 10000000;
  else
    ci = 1.0 / ci;

  // Weight decay terms
  // Up to the user to (not) mix these
  // L2 weight decay
  if (decayL2 != 0) {
    ci += decayL2 * weightSquaredSum(net);
  }

  // L1 weight decay
  if (decayL1 != 0) {
    ci += decayL1 * weightAbsoluteSum(net);
  }

  // Weight elimination
  if (weightElimination != 0 &&
      weightEliminationLambda != 0) {
    ci += weightElimination *
      weightEliminationSum(net, weightEliminationLambda);
  }

  return ci;
}
