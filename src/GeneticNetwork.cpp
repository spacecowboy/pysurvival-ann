/*
 * GeneticNetwork.cpp
 *
 *  Created on: 11 sep 2012
 *      Author: jonas
 */

#include "GeneticNetwork.h"
#include "FFNeuron.h"
#include "FFNetwork.h"
#include "activationfunctions.h"
#include <vector>
#include <stdio.h>
#include "boost/random.hpp"
#include <time.h>
#include <math.h>
#include "ErrorFunctions.h"

using namespace std;

GeneticNetwork*
GeneticNetwork::getGeneticNetwork(GeneticNetwork *cloner,
                                  boost::variate_generator<boost::mt19937&,
                                  boost::normal_distribution<double> >* gaussian,
                                  boost::variate_generator<boost::mt19937&,
                                  boost::uniform_int<> > *uniform)
{
  GeneticNetwork *net = new GeneticNetwork(cloner->getNumOfInputs(),
                                           cloner->getNumOfHidden(),
                                           cloner->getNumOfOutputs());
  net->initNodes();

  // First clone all the weights as initial values
  net->cloneNetworkSlow(cloner);

  double dummy = 0;
  // Then mutate the weights. Set halfpoint to irrelevant values
  // Reverse of resume is independent
  net->mutateWeights(gaussian, uniform, cloner->weightMutationChance,
                     cloner->weightMutationFactor, 1, 0, !cloner->getResume(),
                     &dummy, &dummy);

  return net;
}

GeneticNetwork::GeneticNetwork(unsigned int numOfInputs,
                               unsigned int numOfHidden,
                               unsigned int numOfOutputs) :
  FFNetwork(numOfInputs, numOfHidden, numOfOutputs) {
  populationSize = 50;
  generations = 100;
  weightMutationChance = 0.15;
  weightMutationFactor = 1.0;
  weightMutationHalfPoint = 0;
  decayL2 = 0;
  decayL1 = 0;
  weightElimination = 0;
  weightEliminationLambda = 0;
  resume = false;
  crossoverChance = 1.0;
}

void GeneticNetwork::initNodes() {
  this->hiddenNeurons = new Neuron*[this->numOfHidden];
  unsigned int i;
  for (i = 0; i < this->numOfHidden; i++) {
    this->hiddenNeurons[i] = new GeneticNeuron(i, &sigmoid,
                                               &sigmoidDeriv);
  }
  this->outputNeurons = new Neuron*[1];
  for (i = 0; i < this->numOfOutput; i++) {
    this->outputNeurons[i] = new GeneticNeuron(i, &sigmoid,
                                               &sigmoidDeriv);
  }
  this->bias = new GeneticBias;
}

// Safe to remove
void insertSorted(vector<GeneticNetwork*> * const sortedPopulation,
                  vector<double> * const sortedErrors, const double error,
                  GeneticNetwork * const net) {
  vector<GeneticNetwork*>::iterator netIt;
  vector<double>::iterator errorIt;
  bool inserted = false;
  unsigned int j;

  netIt = sortedPopulation->begin();
  errorIt = sortedErrors->begin();
  // Insert in sorted position
  for (j = 0; j < sortedPopulation->size(); j++) {
    if (error < errorIt[j]) {
      //printf("Inserting at %d, error = %f\n", j, error);
      sortedPopulation->insert(netIt + j, net);
      sortedErrors->insert(errorIt + j, error);
      inserted = true;
      break;
    }
  }
  // If empty, or should be placed last in list
  if (!inserted) {
    //printf("Inserting last, error = %f\n", error);
    sortedPopulation->push_back(net);
    sortedErrors->push_back(error);
    inserted = true;
  }
}

// Safe to remove
void selectParents(boost::variate_generator<boost::mt19937&,
                   boost::geometric_distribution<int, double> > *geometric,
                   unsigned int maximum, unsigned int *mother,
                   unsigned int *father) {

  *mother = (*geometric)() - 1;
  while (*mother >= maximum) {
    *mother = (*geometric)();
  }
  // Make sure they are not the same
  *father = *mother;
  while (*father == *mother || *father >= maximum) {
    *father = (*geometric)() - 1;
  }
}


// Safe to remove
void GeneticNetwork::crossover(boost::variate_generator<boost::mt19937&,
                               boost::uniform_int<> > *uniform,
                               GeneticNetwork* mother,
                               GeneticNetwork* father) {
  // Each individual node is replaced with some probability
  unsigned int n;
  for (n = 0; n < numOfHidden; n++) {
    if ((*uniform)() < 0.5)
      ((GeneticNeuron *) hiddenNeurons[n])->
        cloneNeuron(mother->hiddenNeurons[n]);
    else
      ((GeneticNeuron *) hiddenNeurons[n])->
        cloneNeuron(father->hiddenNeurons[n]);
  }
  // Then output node
  if ((*uniform)() < 0.5)
    ((GeneticNeuron *) outputNeurons[0])->
      cloneNeuron(mother->outputNeurons[0]);
  else
    ((GeneticNeuron *) outputNeurons[0])->
      cloneNeuron(father->outputNeurons[0]);
}

// Safe to remove
void GeneticNetwork::mutateWeights(boost::variate_generator<boost::mt19937&,
                                   boost::normal_distribution<double> >* gaussian,
                                   boost::variate_generator<boost::mt19937&,
                                   boost::uniform_int<> > *uniform,
                                   double mutationChance, double factor,
                                   int deviationHalfPoint, int epoch,
                                   bool independent, double *mutSmallest,
                                   double *mutLargest) {

  double currentFactor = factor;
  if (deviationHalfPoint > 0 && epoch > 0) {
    currentFactor = factor * (1.0 - 0.5 * ((double) epoch /
                                           (double) deviationHalfPoint));
  }
  // Neuron should calculate its own variance
  unsigned int n;
  for (n = 0; n < numOfHidden; n++) {
    ((GeneticNeuron*) hiddenNeurons[n])->mutateWeights(gaussian,
                                                       uniform,
                                                       mutationChance,
                                                       currentFactor,
                                                       independent,
                                                       false,
                                                       mutSmallest,
                                                       mutLargest);
  }
  for (n = 0; n < numOfOutput; n++) {
    ((GeneticNeuron*) outputNeurons[n])->mutateWeights(gaussian,
                                                       uniform,
                                                       mutationChance,
                                                       currentFactor,
                                                       independent,
                                                       false,
                                                       mutSmallest,
                                                       mutLargest);
    }
}

unsigned int GeneticNetwork::getGenerations() const {
  return generations;
}

void GeneticNetwork::setGenerations(unsigned int generations) {
  this->generations = generations;
}

unsigned int GeneticNetwork::getPopulationSize() const {
  return populationSize;
}

void GeneticNetwork::setPopulationSize(unsigned int populationSize) {
  this->populationSize = populationSize;
}

double GeneticNetwork::getWeightMutationChance() const {
  return weightMutationChance;
}

void GeneticNetwork::setWeightMutationChance(double weightMutationChance) {
  this->weightMutationChance = weightMutationChance;
}

unsigned int GeneticNetwork::getWeightMutationHalfPoint() const {
  return weightMutationHalfPoint;
}

void GeneticNetwork::setWeightMutationHalfPoint(unsigned int weightMutationHalfPoint) {
  this->weightMutationHalfPoint = weightMutationHalfPoint;
}

double GeneticNetwork::getWeightMutationFactor() const {
  return weightMutationFactor;
}

void GeneticNetwork::setWeightMutationFactor(double weightMutationFactor) {
  this->weightMutationFactor = weightMutationFactor;
}

double GeneticNetwork::getDecayL1() const {
  return decayL1;
}
void GeneticNetwork::setDecayL1(double val) {
  this->decayL1 = val;
}

double GeneticNetwork::getDecayL2() const {
  return decayL2;
}
void GeneticNetwork::setDecayL2(double val) {
  this->decayL2 = val;
}

double GeneticNetwork::getWeightElimination() const {
  return weightElimination;
}
void GeneticNetwork::setWeightElimination(double val) {
  this->weightElimination = val;
}
double GeneticNetwork::getWeightEliminationLambda() const {
  return weightEliminationLambda;
}
void GeneticNetwork::setWeightEliminationLambda(double val) {
  this->weightEliminationLambda = val;
}

bool GeneticNetwork::getResume()const {
  return resume;
}
void GeneticNetwork::setResume(bool val) {
  this->resume = val;
}

// Safe to remove
void GeneticNetwork::cloneNetwork(GeneticNetwork* original) {
  unsigned int n;
  for (n = 0; n < numOfHidden; n++) {
    ((GeneticNeuron *) hiddenNeurons[n])->
      cloneNeuron(original->hiddenNeurons[n]);
  }
  // Then output neuron
  for (n = 0; n < numOfOutput; n++) {
    ((GeneticNeuron *) outputNeurons[n])->
      cloneNeuron(original->outputNeurons[0]);
  }
}

// Safe to remove
void GeneticNetwork::cloneNetworkSlow(GeneticNetwork* cloner) {
  resetNodes();
  GeneticNetwork *net = this;
  unsigned int i, j;
  int neuronId, targetId;
  double weight;
  // Connect hidden to input, bias and hidden
  // Connect output to hidden and input
  for (i = 0; i < cloner->getNumOfHidden(); i++) {
    neuronId = cloner->getHiddenNeuron(i)->getId();
    // Bias
    if (cloner->getHiddenNeuron(i)->getNeuronWeight(-1, &weight)) {
      net->connectHToB((unsigned int) neuronId, weight);
    }

    // Output to hidden
    for (j = 0; j < cloner->getNumOfOutputs(); j++) {
      if (cloner->getOutputNeuron(j)->getNeuronWeight(neuronId, &weight)) {
        net->connectOToH(j, (unsigned int) neuronId, weight);
      }
    }

    // Inputs
    for (j = 0; j < cloner->getNumOfInputs(); j++) {
      if (cloner->getHiddenNeuron(i)->getInputWeight(j, &weight)) {
        net->connectHToI(neuronId, j, weight);
      }
    }

    // Hidden
    for (j = 0; j < cloner->getNumOfHidden(); j++) {
      targetId = cloner->getHiddenNeuron(j)->getId();
      if (cloner->getHiddenNeuron(i)->getNeuronWeight(targetId, &weight)) {
        net->connectHToH((unsigned int) neuronId, (unsigned int) targetId,
                         weight);
      }
    }
  }

  for (i = 0; i < cloner->getNumOfOutputs(); i++) {

    // Connect output to bias
    if (cloner->getOutputNeuron(i)->getNeuronWeight(-1, &weight)) {
      net->connectOToB(i, weight);
    }

    // Output to input
    for (j = 0; j < cloner->getNumOfInputs(); j++) {
      if(cloner->getOutputNeuron(i)->getInputWeight(j, &weight)) {
        net->connectOToI(i, j, weight);
      }
    }

  }

  // Set functions
  net->setHiddenActivationFunction(cloner->getHiddenActivationFunction());
  net->setOutputActivationFunction(cloner->getOutputActivationFunction());
}

// Calculate the sum of all weights squared (L2 norm)
double weightSquaredSum2(FFNetwork *net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net->getNumOfHidden(); n++) {
    // neuron weights
    sum += net->getHiddenNeuron(n)->getWeightsSquaredSum();
    numOfCons += net->getHiddenNeuron(n)->getNumOfConnections();
  }

  for (n = 0; n < net->getNumOfOutputs(); n++) {
    // Input weights
    sum += net->getOutputNeuron(n)->getWeightsSquaredSum();
    numOfCons += net->getOutputNeuron(n)->getNumOfConnections();
  }

  //printf("Weight squared sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of absolute values of weights (L1 norm)
double weightAbsoluteSum2(FFNetwork *net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net->getNumOfHidden(); n++) {
    // neuron weights
    sum += net->getHiddenNeuron(n)->getWeightsAbsoluteSum();
    numOfCons += net->getHiddenNeuron(n)->getNumOfConnections();
  }

  for (n = 0; n < net->getNumOfOutputs(); n++) {
    // Input weights
    sum += net->getOutputNeuron(n)->getWeightsAbsoluteSum();
    numOfCons += net->getOutputNeuron(n)->getNumOfConnections();
  }
  //printf("Weight absolute sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of soft weight elimination terms
double weightEliminationSum2(FFNetwork *net, double lambda) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net->getNumOfHidden(); n++) {
    // neuron weights
    sum += net->getHiddenNeuron(n)->
      getWeightEliminationSum(lambda);
    numOfCons += net->getHiddenNeuron(n)->getNumOfConnections();
  }

  for (n = 0; n < net->getNumOfOutputs(); n++) {
    // Input weights
    sum += net->getOutputNeuron(n)->
      getWeightEliminationSum(lambda);
    numOfCons += net->getOutputNeuron(n)->getNumOfConnections();
  }
  //printf("Weight elimination sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

double GeneticNetwork::evaluateNetwork(GeneticNetwork *net, double *X,
                                       double *Y,unsigned int length,
                                       double *outputs) {
  unsigned int i, n;
  double error = 0;

  // Evaluate each input set
  // Average over all inputs and number of outputs
  for (i = 0; i < length; i++) {
    // Place output in correct position here
    net->output(X + i*net->getNumOfInputs(),
                outputs + net->getNumOfOutputs() * i);
    for (n = 0; n < net->getNumOfOutputs(); n++) {
      error += SSE(Y[i * net->getNumOfOutputs() + n],
                   outputs[net->getNumOfOutputs() * i + n])
        / ((double) length * net->getNumOfOutputs());
    }
  }

  // Weight decay terms
  // Up to the user to (not) mix these
  // L2 weight decay
  if (decayL2 != 0) {
    error += decayL2 * weightSquaredSum2(net);
  }

  // L1 weight decay
  if (decayL1 != 0) {
    error += decayL1 * weightAbsoluteSum2(net);
  }

  // Weight elimination
  if (weightElimination != 0 &&
      weightEliminationLambda != 0) {
    error += weightElimination *
      weightEliminationSum2(net, weightEliminationLambda);
  }

  return error;
}


// Safe to remove
/*
 * This version does not replace the entire population each generation.
 * Two parents are selected at random to create a child.
 * This child is inserted into the list sorted on error. The worst network is
 * destroyed if population exceeds limit.
 * One generation is considered to be the same number of matings as
 * population size.
 * Networks to be mated are selected with the geometric distribution,
 * probability of the top network to be chosen = 0.05
 * Mutation chance dictate the probability of every single weight being mutated.
 */
void GeneticNetwork::learn(double *X, double *Y,
                           unsigned int length) {
  // Init random number stuff
  boost::mt19937 eng; // a core engine class
  eng.seed(time(NULL));
  // Geometric distribution for selecting parents
  boost::geometric_distribution<int, double> geo_dist(0.95);
  boost::variate_generator<boost::mt19937&,
                           boost::geometric_distribution<int, double> > geometric(eng,
                                                                                  geo_dist);
  // Normal distribution for weight mutation, 0 mean and 1 stddev
  // We can then get any normal distribution with y = mean + stddev * x
  boost::normal_distribution<double> gauss_dist(0, 1);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > gaussian(
                                                                                          eng, gauss_dist);
  // Uniform distribution 0 to 1 (inclusive)
  boost::uniform_int<> uni_dist(0, 1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform(
                                                                           eng, uni_dist);

  // Create a population of networks
  vector<GeneticNetwork*> sortedPopulation;
  vector<double> sortedErrors;

  // Have one throw away network used to create next child
  sortedPopulation.reserve(populationSize + 1);
  sortedErrors.reserve(populationSize + 1);

  // Rank and insert them in a sorted order
  double error;
  double *outputs = new double[numOfOutput*length];
  unsigned int i;
  vector<GeneticNetwork*>::iterator netIt;
  vector<double>::iterator errorIt;

  printf("Data size: %d\n", length);

  for (i = 0; i < populationSize + 1; i++) {
       GeneticNetwork *net = getGeneticNetwork(this, &gaussian,
                                               &uniform);
    // evaluate error here
    error = evaluateNetwork(net, X, Y, length, outputs);
    insertSorted(&sortedPopulation, &sortedErrors, error, net);
  }

  // Save the best network in the population
  GeneticNetwork *best = sortedPopulation.front();
  GeneticNetwork *child;

  // For debug, keep track of every generation's least and biggest mutations
  double mutSmallest, mutLargest;

  // For each generation
  unsigned int curGen, genChild, mother, father;
  for (curGen = 0; curGen < generations; curGen++) {
    mutSmallest = 1000;
    mutLargest = 0;

    for (genChild = 0; genChild < populationSize; genChild++) {
      // Chance that we we don't do crossover
      // debug
      // The key is that crappy children must die. Now they dont
      // there is no selection process. e.g. just a random process
      crossoverChance = 2;
      if (uniform() < crossoverChance) {
        //printf("Doing crossover\n");
        // We recycle the worst network
        child = sortedPopulation.back();
        sortedPopulation.pop_back();
        sortedErrors.pop_back();
        // Select two networks
        selectParents(&geometric, populationSize, &mother, &father);
        // Create new child through crossover
        child->crossover(&uniform, sortedPopulation[mother],
                         sortedPopulation[father]);
      }
      else {
        // Mutate an existing network!
        selectParents(&geometric, populationSize, &mother, &father);
        //printf("m: %d, f: %d\n", mother, father);
        child = sortedPopulation.at(mother);
        sortedPopulation.erase(sortedPopulation.begin() + mother);
        sortedErrors.erase(sortedErrors.begin() + mother);
      }
      // Mutate child
      child->mutateWeights(&gaussian, &uniform, weightMutationChance,
                           weightMutationFactor, weightMutationHalfPoint,
                           curGen, false, &mutSmallest, &mutLargest);
      // evaluate error child
      error = evaluateNetwork(child, X, Y, length, outputs);
      // Insert child into the sorted list
      insertSorted(&sortedPopulation, &sortedErrors, error, child);
      // Save best network
      best = sortedPopulation.front();
    }
    // Add printEpoch check here
    printf("gen: %d, best: %f, weightSmallest: %f, weightLargest: %f\n", curGen,
           sortedErrors.front(), mutSmallest, mutLargest);

    if (decayL2 != 0) {
      printf("L2term = %f * %f\n", decayL2, weightSquaredSum2(best));
    }
    // L1 weight decay
    if (decayL1 != 0) {
      printf("L1term = %f * %f\n", decayL1, weightAbsoluteSum2(best));
    }
    // Weight elimination
    if (weightElimination != 0 &&
        weightEliminationLambda != 0) {
      printf("Decayterm(%f) = %f * %f\n",weightEliminationLambda,
             weightElimination,
             weightEliminationSum2(best, weightEliminationLambda));
    }
  }

  // When done, make this network into the best network
  printf("best eval error: %f\n", (evaluateNetwork(best, X, Y,
                                                   length, outputs)));
  this->cloneNetworkSlow(best);
  printf("this eval error: %f\n", (evaluateNetwork(this, X, Y,
                                                   length, outputs)));

  // And destroy population
  // do this last of all!
  best = NULL;
  for (netIt = sortedPopulation.begin(); netIt < sortedPopulation.end();
       netIt++) {
    delete *netIt;
  }
  delete[] outputs;
}

/*
 * ------------------------
 * Neuron definition
 * ------------------------
 */
GeneticNeuron::GeneticNeuron(int id) :
  Neuron(id) {

}

GeneticNeuron::GeneticNeuron(int id,
                             double (*activationFunction)(double),
                             double (*activationDerivative)(double)) :
  Neuron(id, activationFunction, activationDerivative) {

}

GeneticNeuron::~GeneticNeuron() {

}

void GeneticNeuron::cloneNeuron(Neuron* original) {
  unsigned int i;
  // First neuron connections
  for (i = 0; i < neuronConnections->size(); i++) {
    neuronConnections->at(i).second =
      original->neuronConnections->at(i).second;
  }

  // Then input connections
  for (i = 0; i < inputConnections->size(); i++) {
    inputConnections->at(i).second =
      original->inputConnections->at(i).second;
  }
}

void GeneticNeuron::cloneNeuronSlow(Neuron* original) {
  unsigned int i, j;
  int originalId, cloneId;
  unsigned int originalIndex, cloneIndex;
  // First neuron connections
  for (i = 0; i < neuronConnections->size(); i++) {
    cloneId = neuronConnections->at(i).first->getId();
    for (j = 0; j < original->neuronConnections->size(); j++) {
      originalId = original->neuronConnections->at(j).first->getId();
      if (cloneId == originalId) {
        neuronConnections->at(i).second =
          original->neuronConnections->at(j).second;
        break;
      }
    }
  }

  // Then input connections
  for (i = 0; i < inputConnections->size(); i++) {
    cloneIndex = inputConnections->at(i).first;
    for (j = 0; j < original->inputConnections->size(); j++) {
      originalIndex = original->inputConnections->at(j).first;
      if (cloneIndex == originalIndex) {
        inputConnections->at(i).second =
          original->inputConnections->at(j).second;
        break;
      }
    }
  }
}

/**
 * Setting independent
 */
void GeneticNeuron::mutateWeights(boost::variate_generator<boost::mt19937&,
                                  boost::normal_distribution<double> >* gaussian,
                                  boost::variate_generator<boost::mt19937&,
                                  boost::uniform_int<> > *uniform,
                                  double mutationChance, double factor,
                                  bool independent, bool l2scale,
                                  double *mutSmallest,
                                  double *mutLargest) {
  unsigned int n;
  // l2 = sqrt( sum( x^2 ) )
  double l2 = 0, mutation = 0;
  // Base deviation is the magnitude of the weight. Minus-signs are irrelevant
  // Since it's random anyway
  for (n = 0; n < neuronConnections->size(); n++) {
    if (independent) {
      neuronConnections->at(n).second = (*gaussian)();
    }
    else if ((*uniform)() <= mutationChance) {
      //mutation = (*gaussian)() * neuronConnections->at(n).second * factor;
      mutation = (*gaussian)() * factor;
      if (neuronConnections->at(n).second < 1.0)
        mutation *= neuronConnections->at(n).second;
      neuronConnections->at(n).second += mutation;
      /*
      if (fabs(mutation) > *mutLargest)
        *mutLargest = fabs(mutation);
      else if (fabs(mutation) < *mutSmallest)
        *mutSmallest = fabs(mutation);
        */
    }

    l2 += neuronConnections->at(n).second * neuronConnections->at(n).second;
  }
  for (n = 0; n < inputConnections->size(); n++) {
    if (independent) {
      inputConnections->at(n).second = (*gaussian)();
    }
    else if ((*uniform)() <= mutationChance) {
      //mutation = (*gaussian)()* inputConnections->at(n).second * factor;
      mutation = (*gaussian)() * factor;
      if (inputConnections->at(n).second < 1.0)
        mutation *= inputConnections->at(n).second;
      inputConnections->at(n).second += mutation;

      /*
      if (fabs(mutation) > *mutLargest)
        *mutLargest = fabs(mutation);
      else if (fabs(mutation) < *mutSmallest)
        *mutSmallest = fabs(mutation);
        */
    }

    l2 += inputConnections->at(n).second * inputConnections->at(n).second;
  }

  // Scale by L2 norm
  if (independent || l2scale) {
    l2 = sqrt(l2);
    for (n = 0; n < neuronConnections->size(); n++) {
      neuronConnections->at(n).second /= l2;
    }
    for (n = 0; n < inputConnections->size(); n++) {
      inputConnections->at(n).second /= l2;
    }
  }
  for (n = 0; n < neuronConnections->size(); n++) {

    if (fabs(neuronConnections->at(n).second) > *mutLargest)
      *mutLargest = fabs(neuronConnections->at(n).second);
    else if (fabs(neuronConnections->at(n).second) < *mutSmallest)
      *mutSmallest = fabs(neuronConnections->at(n).second);
  }

  for (n = 0; n < inputConnections->size(); n++) {
    if (fabs(inputConnections->at(n).second) > *mutLargest)
      *mutLargest = fabs(inputConnections->at(n).second);
    else if (fabs(inputConnections->at(n).second) < *mutSmallest)
      *mutSmallest = fabs(inputConnections->at(n).second);
  }
}
