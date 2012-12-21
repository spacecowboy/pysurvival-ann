/*
 * GeneticNetwork.cpp
 *
 *  Created on: 11 sep 2012
 *      Author: jonas
 */

#include "GeneticSurvivalNetwork.h"
#include "FFNeuron.h"
#include "FFNetwork.h"
#include "activationfunctions.h"
#include "c_index.h"
#include <vector>
#include <stdio.h>
#include "boost/random.hpp"
#include <time.h>
#include <math.h>

using namespace std;

GeneticSurvivalNetwork* getGeneticSurvivalNetwork(GeneticSurvivalNetwork *cloner,
                                                  boost::variate_generator<boost::mt19937&,
                                                  boost::uniform_int<> >* uniform) {

  GeneticSurvivalNetwork *net = new GeneticSurvivalNetwork(cloner->getNumOfInputs(),
                                                           cloner->getNumOfHidden());
  net->initNodes();

  unsigned int i, j;
  int neuronId, targetId;
  double weight;
  // Connect hidden to input, bias and hidden
  // Connect output to hidden and input
  for (i = 0; i < cloner->getNumOfHidden(); i++) {
    neuronId = cloner->getHiddenNeuron(i)->getId();
    // Bias
    if (cloner->getHiddenNeuron(i)->getNeuronWeight(-1, &weight)) {
      net->connectHToB((unsigned int) neuronId,
                       ((*uniform)() - 0.5) * cloner->getWeightMutationStdDev());
    }

    // Output to hidden
    if (cloner->getOutputNeuron(0)->getNeuronWeight(neuronId, &weight)) {
      net->connectOToH(0, (unsigned int) neuronId,
                       ((*uniform)() - 0.5) * cloner->getWeightMutationStdDev());
    }

    // Inputs
    for (j = 0; j < cloner->getNumOfInputs(); j++) {
      if (cloner->getHiddenNeuron(i)->getInputWeight(j, &weight)) {
        net->connectHToI(neuronId, j,
                         ((*uniform)() - 0.5) * cloner->getWeightMutationStdDev());
      }
    }

    // Hidden
    for (j = 0; j < cloner->getNumOfHidden(); j++) {
      targetId = cloner->getHiddenNeuron(j)->getId();
      if (cloner->getHiddenNeuron(i)->getNeuronWeight(targetId, &weight)) {
        net->connectHToH((unsigned int) neuronId, (unsigned int) targetId,
                         ((*uniform)() - 0.5) * cloner->getWeightMutationStdDev());
      }
    }
  }
  // Connect output to bias
  if (cloner->getOutputNeuron(0)->getNeuronWeight(-1, &weight)) {
    net->connectOToB(0, ((*uniform)() - 0.5) * cloner->getWeightMutationStdDev());
  }

  // Output to input
  // Inputs
  for (j = 0; j < cloner->getNumOfInputs(); j++) {
    if(cloner->getOutputNeuron(0)->getInputWeight(j, &weight)) {
      net->connectOToI(0, j,
                       ((*uniform)() - 0.5) * cloner->getWeightMutationStdDev());
    }
  }

  // Set functions
  net->setHiddenActivationFunction(cloner->getHiddenActivationFunction());
  net->setOutputActivationFunction(cloner->getOutputActivationFunction());

  return net;
}

GeneticSurvivalNetwork::GeneticSurvivalNetwork(unsigned int numOfInputs,
                                               unsigned int numOfHidden) :
  FFNetwork(numOfInputs, numOfHidden, 1) {
  populationSize = 50;
  generations = 100;
  weightMutationChance = 0.15;
  weightMutationStdDev = 0.1;
  weightMutationHalfPoint = 0;
}

void GeneticSurvivalNetwork::initNodes() {
  this->hiddenNeurons = new Neuron*[this->numOfHidden];
  unsigned int i;
  for (i = 0; i < this->numOfHidden; i++) {
    this->hiddenNeurons[i] = new GeneticSurvivalNeuron(i, &hyperbole,
                                                       &hyperboleDeriv);
  }
  this->outputNeurons = new Neuron*[1];
  this->outputNeurons[0] = new GeneticSurvivalNeuron(0, &linear, &linearDeriv);
  this->bias = new GeneticSurvivalBias;
}

void insertSorted(vector<GeneticSurvivalNetwork*> * const sortedPopulation,
                  vector<double> * const sortedErrors, const double error,
                  GeneticSurvivalNetwork * const net) {
  vector<GeneticSurvivalNetwork*>::iterator netIt;
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

void selectParents(
                   boost::variate_generator<boost::mt19937&,
                                            boost::geometric_distribution<int, double> > *geometric,
                   unsigned int maximum, unsigned int *mother, unsigned int *father) {

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

void GeneticSurvivalNetwork::crossover(
                                       boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
                                       GeneticSurvivalNetwork* mother, GeneticSurvivalNetwork* father) {
  // Each individual node is replaced with some probability
  unsigned int n;
  for (n = 0; n < numOfHidden; n++) {
    if ((*uniform)() < 0.5)
      ((GeneticSurvivalNeuron *) hiddenNeurons[n])->cloneNeuron(
                                                                mother->hiddenNeurons[n]);
    else
      ((GeneticSurvivalNeuron *) hiddenNeurons[n])->cloneNeuron(
                                                                father->hiddenNeurons[n]);
  }
  // Then output node
  if ((*uniform)() < 0.5)
    ((GeneticSurvivalNeuron *) outputNeurons[0])->cloneNeuron(
                                                              mother->outputNeurons[0]);
  else
    ((GeneticSurvivalNeuron *) outputNeurons[0])->cloneNeuron(
                                                              father->outputNeurons[0]);

}

void GeneticSurvivalNetwork::mutateWeights(
                                           boost::variate_generator<boost::mt19937&,
                                                                    boost::normal_distribution<double> >* gaussian,
                                           boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
                                           double mutationChance, double stdDev, int deviationHalfPoint,
                                           int epoch) {

  double currentStdDev = stdDev;
  if (deviationHalfPoint > 0 && epoch > 0) {
    currentStdDev = stdDev * (1.0 - 0.5 * ((double) epoch / (double) deviationHalfPoint));
  }

  unsigned int n;
  for (n = 0; n < numOfHidden; n++) {
    ((GeneticSurvivalNeuron*) hiddenNeurons[n])->mutateWeights(gaussian,
                                                               uniform, mutationChance, currentStdDev);
  }
  ((GeneticSurvivalNeuron*) outputNeurons[0])->mutateWeights(gaussian, uniform,
                                                             mutationChance, currentStdDev);
}

unsigned int GeneticSurvivalNetwork::getGenerations() const {
  return generations;
}

void GeneticSurvivalNetwork::setGenerations(unsigned int generations) {
  this->generations = generations;
}

unsigned int GeneticSurvivalNetwork::getPopulationSize() const {
  return populationSize;
}

void GeneticSurvivalNetwork::setPopulationSize(unsigned int populationSize) {
  this->populationSize = populationSize;
}

double GeneticSurvivalNetwork::getWeightMutationChance() const {
  return weightMutationChance;
}

void GeneticSurvivalNetwork::setWeightMutationChance(
                                                     double weightMutationChance) {
  this->weightMutationChance = weightMutationChance;
}

unsigned int GeneticSurvivalNetwork::getWeightMutationHalfPoint() const {
  return weightMutationHalfPoint;
}

void GeneticSurvivalNetwork::setWeightMutationHalfPoint(
                                                        unsigned int weightMutationHalfPoint) {
  this->weightMutationHalfPoint = weightMutationHalfPoint;
}

double GeneticSurvivalNetwork::getWeightMutationStdDev() const {
  return weightMutationStdDev;
}

void GeneticSurvivalNetwork::setWeightMutationStdDev(
                                                     double weightMutationStdDev) {
  this->weightMutationStdDev = weightMutationStdDev;
}

void GeneticSurvivalNetwork::cloneNetwork(GeneticSurvivalNetwork* original) {
  unsigned int n;
  for (n = 0; n < numOfHidden; n++) {
    ((GeneticSurvivalNeuron *) hiddenNeurons[n])->
      cloneNeuron(
                  original->hiddenNeurons[n]);
  }
  // Then output neuron
  ((GeneticSurvivalNeuron *) outputNeurons[0])->
    cloneNeuron(
                original->outputNeurons[0]);
}

void GeneticSurvivalNetwork::cloneNetworkSlow(GeneticSurvivalNetwork* cloner) {
  resetNodes();
  GeneticSurvivalNetwork *net = this;
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
    if (cloner->getOutputNeuron(0)->getNeuronWeight(neuronId, &weight)) {
      net->connectOToH(0, (unsigned int) neuronId, weight);
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
        net->connectHToH((unsigned int) neuronId, (unsigned int) targetId, weight);
      }
    }
  }
  // Connect output to bias
  if (cloner->getOutputNeuron(0)->getNeuronWeight(-1, &weight)) {
    net->connectOToB(0, weight);
  }

  // Output to input
  // Inputs
  for (j = 0; j < cloner->getNumOfInputs(); j++) {
    if(cloner->getOutputNeuron(0)->getInputWeight(j, &weight)) {
      net->connectOToI(0, j, weight);
    }
  }

  // Set functions
  net->setHiddenActivationFunction(cloner->getHiddenActivationFunction());
  net->setOutputActivationFunction(cloner->getOutputActivationFunction());

  /*











unsigned int n, m;
  int nId, mId;
  for (n = 0; n < numOfHidden; n++) {
    nId = hiddenNeurons[n]->getId();
    for (m = 0; m < original->getNumOfHidden(); m++) {
      mId = original->hiddenNeurons[m]->getId();

      if (nId == mId) {
        //printf("ACSH: %d == %d\n", nId, mId);
        ((GeneticSurvivalNeuron *) hiddenNeurons[n])->
          cloneNeuronSlow(
                          original->hiddenNeurons[m]);
        break;
      }
    }
  }
  // Then output neuron
  for (n = 0; n < numOfOutput; n++) {
    nId = outputNeurons[n]->getId();
    for (m = 0; m < original->getNumOfOutputs(); m++) {
      mId = original->outputNeurons[m]->getId();

      if (nId == mId) {
        //printf("ACSO: %d == %d\n", nId, mId);
        ((GeneticSurvivalNeuron *) outputNeurons[n])->
          cloneNeuronSlow(
                          original->outputNeurons[m]);
        break;
      }
    }
    } */
}

double evaluateNetwork(GeneticSurvivalNetwork *net, double *X, double *Y,
                       unsigned int length, double *outputs) {
  // Evaluate each input set
  for (unsigned int i = 0; i < length; i++) {
    // Place output in correct position here
    net->output(X + i*net->getNumOfInputs(), outputs + i);
  }
  // Now calculate c-index
  double ci = get_C_index(outputs, Y, length);

  //printf("ci = %f\n", ci);

  // Return the inverse since this returns the error of the network
  // If less than 0.001, return 1000 instead to avoid dividing by zero
  if (ci < 0.0000001)
    return 10000000;
  else
    return 1.0 / ci;
}


/*
 * This version does not replace the entire population each generation. Two parents are selected at random to create a child.
 * This child is inserted into the list sorted on error. The worst network is destroyed if population exceeds limit.
 * One generation is considered to be the same number of matings as population size.
 * Networks to be mated are selected with the geometric distribution, probability of the top network to be chosen = 0.05
 * Mutation chance dictate the probability of every single weight being mutated.
 */
void GeneticSurvivalNetwork::learn(double *X, double *Y,
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
  vector<GeneticSurvivalNetwork*> sortedPopulation;
  vector<double> sortedErrors;

  // Have one throw away network used to create next child
  sortedPopulation.reserve(populationSize + 1);
  sortedErrors.reserve(populationSize + 1);

  // Rank and insert them in a sorted order
  double error;
  double *outputs = new double[numOfOutput*length];
  unsigned int i;
  vector<GeneticSurvivalNetwork*>::iterator netIt;
  vector<double>::iterator errorIt;

  printf("Length: %d\n", length);

  for (i = 0; i < populationSize + 1; i++) {
    GeneticSurvivalNetwork *net = getGeneticSurvivalNetwork(this, &uniform);
    // evaluate error here
    error = evaluateNetwork(net, X, Y, length, outputs);

    insertSorted(&sortedPopulation, &sortedErrors, error, net);
  }

  // Save the best network in the population
  GeneticSurvivalNetwork *best = sortedPopulation.front();
  GeneticSurvivalNetwork *child;

  // For each generation
  unsigned int curGen, genChild, mother, father;
  for (curGen = 0; curGen < generations; curGen++) {
    for (genChild = 0; genChild < populationSize; genChild++) {
      //printf("gen: %d, genchild: %d\n", curGen, genChild);
      // We recycle the worst network
      child = sortedPopulation.back();
      //printf("error at back: %f\n", sortedErrors.back());
      // Remove it from the list
      sortedPopulation.pop_back();
      sortedErrors.pop_back();
      // Select two networks
      selectParents(&geometric, populationSize, &mother, &father);
      //printf("Mother: %d, Father: %d\n", mother, father);

      // Create new child through crossover
      child->crossover(&uniform, sortedPopulation[mother],
                       sortedPopulation[father]);
      // Mutate child
      child->mutateWeights(&gaussian, &uniform, weightMutationChance,
                           weightMutationStdDev, weightMutationHalfPoint, curGen);

      // evaluate error child
      //printf("evaluating...\n");
      error = evaluateNetwork(child, X, Y, length, outputs);
      //printf("evaluated: %d\n", error);

      //printf("new child error: %f\n", error);
      // Insert child into the sorted list
      insertSorted(&sortedPopulation, &sortedErrors, error, child);
      // Save best network
      best = sortedPopulation.front();
    }
    // Add printEpoch check here
    printf("gen: %d, best: %f\n", curGen, 1.0/sortedErrors.front());
  }

  // When done, make this network into the best network
  printf("best eval result: %f\n", 1.0/(evaluateNetwork(best, X, Y, length, outputs)));
  this->cloneNetworkSlow(best);
  printf("this eval result: %f\n", 1.0/(evaluateNetwork(this, X, Y, length, outputs)));

  // And destroy population
  // do this last of all!
  best = NULL;
  for (netIt = sortedPopulation.begin(); netIt < sortedPopulation.end();
       netIt++) {
    //printf("deleting population\n");
    delete *netIt;
  }

  delete[] outputs;
}

/*
 * ------------------------
 * Neuron definition
 * ------------------------
 */
GeneticSurvivalNeuron::GeneticSurvivalNeuron(int id) :
  Neuron(id) {

}

GeneticSurvivalNeuron::GeneticSurvivalNeuron(int id,
                                             double (*activationFunction)(double),
                                             double (*activationDerivative)(double)) :
  Neuron(id, activationFunction, activationDerivative) {

}

GeneticSurvivalNeuron::~GeneticSurvivalNeuron() {

}

void GeneticSurvivalNeuron::cloneNeuron(Neuron* original) {
  unsigned int i;
  // First hidden connections
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

void GeneticSurvivalNeuron::cloneNeuronSlow(Neuron* original) {
  //printf("NCS start: %d == %d\n", getId(), original->getId());
  unsigned int i, j;
  int originalId, cloneId;
  unsigned int originalIndex, cloneIndex;
  // First hidden connections
  for (i = 0; i < neuronConnections->size(); i++) {
    cloneId = neuronConnections->at(i).first->getId();
    for (j = 0; j < original->neuronConnections->size(); j++) {
      originalId = original->neuronConnections->at(j).first->getId();
      if (cloneId == originalId) {
        //printf("NCSH: %d == %d\n", cloneId, originalId);
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
        //printf("NCSI: %d == %d\n", cloneIndex, originalIndex);
        inputConnections->at(i).second =
          original->inputConnections->at(j).second;
        break;
      }
    }
  }
}

void GeneticSurvivalNeuron::mutateWeights(
                                          boost::variate_generator<boost::mt19937&,
                                                                   boost::normal_distribution<double> >* gaussian,
                                          boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
                                          double mutationChance, double stdDev) {
  unsigned int n;
  for (n = 0; n < neuronConnections->size(); n++) {
    if ((*uniform)() <= mutationChance)
      neuronConnections->at(n).second += (*gaussian)() * stdDev;
  }
  for (n = 0; n < inputConnections->size(); n++) {
    if ((*uniform)() <= mutationChance)
      inputConnections->at(n).second += (*gaussian)() * stdDev;
  }
}
