/*
 * GeneticNetwork.cpp
 *
 *  Created on: 11 sep 2012
 *      Author: jonas
 */

#include "GeneticNetwork.hpp"
#include "MatrixNetwork.hpp"
#include "GeneticFitness.hpp"
#include "GeneticSelection.hpp"
#include "GeneticCrossover.hpp"
#include "GeneticMutation.hpp"
#include "global.hpp"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <thread>
#include <ctime>

using namespace std;


GeneticNetwork::GeneticNetwork(const unsigned int numOfInputs,
                               const unsigned int numOfHidden,
                               const unsigned int numOfOutputs) :
  MatrixNetwork(numOfInputs, numOfHidden, numOfOutputs),
  populationSize(50),
  generations(100),
  weightMutationChance(0.15),
  weightMutationFactor(1.0),
  connsMutationChance(0.1),
  actFuncMutationChance(0.1),
  crossoverChance(1.0),
  selectionMethod(SelectionMethod::SELECTION_GEOMETRIC),
  crossoverMethod(CrossoverMethod::CROSSOVER_ONEPOINT),
//insertMethod(INSERT_ALL),
  fitnessFunction(FitnessFunction::FITNESS_MSE),
  weightMutationHalfPoint(0),
  decayL2(0),
  decayL1(0),
  weightElimination(0),
  weightEliminationLambda(0)
{
}

void GeneticNetwork::insertSorted(vector<GeneticNetwork*>  &sortedPopulation,
                                  vector<double> &sortedErrors,
                                  const double error,
                                  GeneticNetwork * const net)
{
  vector<GeneticNetwork*>::iterator netIt;
  vector<double>::iterator errorIt;
  bool inserted = false;
  unsigned int j;

  netIt = sortedPopulation.begin();
  errorIt = sortedErrors.begin();
  // Insert in sorted position
  for (j = 0; j < sortedPopulation.size(); j++) {
    if (error < errorIt[j]) {
      //printf("Inserting at %d, error = %f\n", j, error);
      sortedPopulation.insert(netIt + j, net);
      sortedErrors.insert(errorIt + j, error);
      inserted = true;
      break;
    }
  }
  // If empty, or should be placed last in list
  if (!inserted) {
    //printf("Inserting last, error = %f\n", error);
    sortedPopulation.push_back(net);
    sortedErrors.push_back(error);
    inserted = true;
  }
}

void GeneticNetwork::insertLast(vector<GeneticNetwork*>  &sortedPopulation,
                                vector<double> &sortedErrors,
                                GeneticNetwork * const net)
{
  //printf("Inserting last, error = %f\n", error);
  sortedPopulation.push_back(net);
  sortedErrors.push_back(9999999999.0);
}

void GeneticNetwork::cloneNetwork(GeneticNetwork &original) {
  std::copy(original.weights,
            original.weights + original.LENGTH * original.LENGTH,
            this->weights);

  std::copy(original.conns,
            original.conns + original.LENGTH * original.LENGTH,
            this->conns);

  std::copy(original.actFuncs,
            original.actFuncs + original.LENGTH * original.LENGTH,
            this->actFuncs);
}


/**
 * Does the actual work in an epoch. Designed to be launched in parallell
 * in many threads.
 */
void breedNetworks(GeneticNetwork *self,
                   vector<GeneticNetwork*> *sortedPopulation,
                   vector<double> *sortedErrors,
                   const unsigned int childCount,
                   const unsigned int curGen,
                   const double * const X,
                   const double * const Y,
                   const unsigned int length)
{
  //std::cout << "Launched by thread " << std::this_thread::get_id() << std::endl;

  unsigned int motherIndex, fatherIndex;

  GeneticNetwork *pBrother, *pSister, *pMother, *pFather;

  double bFitness, sFitness;
  double *outputs = new double[self->OUTPUT_COUNT * length];

  unsigned int threadChild, i;
  for (threadChild = 0; threadChild < childCount; threadChild++) {
    // We recycle the worst networks
    //pChild = self->popLastNetwork(*sortedPopulation, *sortedErrors);

    // Select two networks
    getSelection(self->selectionMethod,
                 *sortedErrors,
                 self->populationSize,
                 motherIndex, fatherIndex);

    // get mother and father in a thread safe way
    JGN_lockPopulation();

    pMother = self->popNetwork(motherIndex, *sortedPopulation, *sortedErrors);
    pFather = self->popNetwork(fatherIndex, *sortedPopulation, *sortedErrors);

    pBrother = self->popLastNetwork(*sortedPopulation, *sortedErrors);
    pSister = self->popLastNetwork(*sortedPopulation, *sortedErrors);

    JGN_unlockPopulation();

    // Crossover
    if (JGN_rand.uniform() < self->crossoverChance) {
      evaluateCrossoverFunction(self->crossoverMethod,
                                *pMother, *pFather,
                                *pBrother, *pSister);
    }
    // No crossover, work with clones
    else {
      pSister->cloneNetwork(*pMother);
      pBrother->cloneNetwork(*pFather);
    }


    // Mutation brother
    mutateWeights(*pBrother,
                  self->weightMutationChance,
                  self->weightMutationFactor);
    mutateConns(*pBrother,
                self->connsMutationChance);
    mutateActFuncs(*pBrother,
                   self->actFuncMutationChance);
    // Mutation sister
    mutateWeights(*pSister,
                  self->weightMutationChance,
                  self->weightMutationFactor);
    mutateConns(*pSister,
                self->connsMutationChance);
    mutateActFuncs(*pSister,
                   self->actFuncMutationChance);

    // evaluate error of brother
    // TODO multiple outputs
    for (i = 0; i < length; i++) {
      pBrother->output(X + i * pBrother->INPUT_COUNT,
                       outputs + i * pBrother->INPUT_COUNT);
    }
    bFitness = getFitness(self->fitnessFunction,
                          Y, length,
                          self->OUTPUT_COUNT,
                          outputs);

    // evaluate error of sister
    // TODO multiple outputs
    for (i = 0; i < length; i++) {
      pSister->output(X + i * pSister->INPUT_COUNT,
                      outputs + i * pSister->INPUT_COUNT);
    }
    sFitness = getFitness(self->fitnessFunction,
                          Y, length,
                          self->OUTPUT_COUNT,
                          outputs);


    //    error = -(*(self->pFitnessFunction))(*pChild, X, Y, length, outputs);

    // Weight decay terms
    // Up to the user to (not) mix these
    // L2 weight decay
    /*
    if (self->decayL2 != 0) {
      error += self->decayL2 * weightSquaredSum2(*pChild);
    }

    // L1 weight decay
    if (self->decayL1 != 0) {
      error += self->decayL1 * weightAbsoluteSum2(*pChild);
    }

    // Weight elimination
    if (self->weightElimination != 0 &&
        self->weightEliminationLambda != 0) {
      error += self->weightElimination *
        weightEliminationSum2(*pChild, self->weightEliminationLambda);
    }
    */
    // Insert into the sorted list
    JGN_lockPopulation();

    // Place parents back as dummies
    self->insertLast(*sortedPopulation, *sortedErrors, pMother);
    self->insertLast(*sortedPopulation, *sortedErrors, pFather);

    // Place children at correct positions
    self->insertSorted(*sortedPopulation, *sortedErrors, bFitness, pBrother);
    self->insertSorted(*sortedPopulation, *sortedErrors, sFitness, pSister);

    JGN_unlockPopulation();
  }
  delete[] outputs;
}


void GeneticNetwork::learn(const double * const X,
                           const double * const Y,
                           const unsigned int length) {
  // Reset LOG
  if (this->aLogPerf != NULL) {
    delete[] this->aLogPerf;
    this->aLogPerf = NULL;
  }
  // Allocate new LOG
  this->logPerfLength = generations * this->OUTPUT_COUNT;
  // //Parenthesis initializes to zero
  this->aLogPerf = new double[this->logPerfLength]();
  //memset(& this->aLogPerf, 0, sizeof(double) * this->logPerfLength);

  // Create a population of networks
  vector<GeneticNetwork*> sortedPopulation;
  vector<double> sortedErrors;

  // Number of threads to use, TODO, dont hardcode this
  unsigned int num_threads = 8;
  unsigned int extras = num_threads * 4;
  std::thread threads[num_threads];
  // Each thread will breed these many networks each generation
  unsigned int breedCount = round((double) populationSize / num_threads);
  if (breedCount < 1) {
    breedCount = 1;
  }

  printf("Breed count: %d\n", breedCount);

  // Pre-allocate space
  sortedPopulation.reserve(populationSize + extras);
  sortedErrors.reserve(populationSize + extras);

  // Rank and insert them in a sorted order
  unsigned int i, j;
  vector<GeneticNetwork*>::iterator netIt;
  vector<double>::iterator errorIt;

  printf("Data size: %d\n", length);

  double fitness;
  double *outputs = new double[OUTPUT_COUNT * length];
  for (i = 0; i < populationSize + extras; i++) {
    // use references
    GeneticNetwork *pNet = new GeneticNetwork(INPUT_COUNT,
                                              HIDDEN_COUNT,
                                              OUTPUT_COUNT);

    randomizeNetwork(*pNet, weightMutationFactor);

    // evaluate error here
    for (j = 0; j < length; j++) {
      pNet->output(X + j * pNet->INPUT_COUNT,
                   outputs + j * pNet->INPUT_COUNT);
    }
    fitness = getFitness(fitnessFunction,
                         Y, length,
                         pNet->OUTPUT_COUNT,
                         outputs);

    //    error = -(*pFitnessFunction)(*pNet, X, Y, length, outputs);
    //    error = evaluateNetwork(*pNet, X, Y, length, outputs);

    insertSorted(sortedPopulation, sortedErrors, fitness, pNet);
  }

  // Save the best network in the population
  GeneticNetwork *best = sortedPopulation.front();

  // For each generation
  unsigned int curGen;
  for (curGen = 0; curGen < generations; curGen++) {
    time_t start, end;
    time(&start);
    for (i = 0; i < num_threads; ++i) {
      threads[i] = std::thread(breedNetworks, this,
                               &sortedPopulation,
                               &sortedErrors, breedCount, curGen,
                               X, Y, length);
    }

    // Wait for the threads to finish their work
    for (i = 0; i < num_threads; ++i) {
      threads[i].join();
    }

    time(&end);
    std::cout << "gen time: " << difftime(end, start) << "s" << std::endl;

    // Print some stats about the current best
    best = sortedPopulation.front();

    // Add printEpoch check here
    printf("gen: %d, best: %f\n", curGen,
           sortedErrors.front());

    // Save in log
    this->aLogPerf[curGen] = sortedErrors.front();

    /*
    if (decayL2 != 0) {
      printf("L2term = %f * %f\n", decayL2, weightSquaredSum2(*best));
    }
    // L1 weight decay
    if (decayL1 != 0) {
      printf("L1term = %f * %f\n", decayL1, weightAbsoluteSum2(*best));
    }
    // Weight elimination
    if (weightElimination != 0 &&
        weightEliminationLambda != 0) {
      printf("Decayterm(%f) = %f * %f\n",weightEliminationLambda,
             weightElimination,
             weightEliminationSum2(*best, weightEliminationLambda));
    }
    */
  }

  // When done, make this network into the best network
  for (j = 0; j < length; j++) {
    best->output(X + j * best->INPUT_COUNT,
                 outputs + j * best->INPUT_COUNT);
  }

  printf("best eval fitness: %f\n", getFitness(fitnessFunction,
                                               Y, length,
                                               best->OUTPUT_COUNT,
                                               outputs));
  this->cloneNetwork(*best);
  for (j = 0; j < length; j++) {
    this->output(X + j * this->INPUT_COUNT,
                 outputs + j * this->INPUT_COUNT);
  }
  printf("this eval fitness: %f\n", getFitness(fitnessFunction,
                                               Y, length,
                                               this->OUTPUT_COUNT,
                                               outputs));

  // And destroy population
  // do this last of all!
  best = NULL;
  for (netIt = sortedPopulation.begin(); netIt < sortedPopulation.end();
       netIt++) {
    delete *netIt;
  }
  delete[] outputs;
}

GeneticNetwork *GeneticNetwork::popLastNetwork(
    vector<GeneticNetwork*> &sortedPopulation,
    vector<double> &sortedErrors)
{
  GeneticNetwork *pChild = sortedPopulation.back();
  sortedPopulation.pop_back();
  sortedErrors.pop_back();

  return pChild;
}

GeneticNetwork *GeneticNetwork::popNetwork(unsigned int i,
                                           vector<GeneticNetwork*> &sortedPopulation,
                                           vector<double> &sortedErrors)
{
  // Just in case
  if (i >= sortedPopulation.size()) {
    i = sortedPopulation.size() - 1;
  }
  GeneticNetwork *pChild = sortedPopulation.at(i);
  sortedPopulation.erase(sortedPopulation.begin() + i);
  sortedErrors.erase(sortedErrors.begin() + i);

  return pChild;
}


/*
void
GeneticNetwork::mutateWeights(boost::variate_generator<boost::mt19937&,
                                   boost::normal_distribution<double> >
                                   &gaussian,
                                   boost::variate_generator<boost::mt19937&,
                                   boost::uniform_real<double> >
                                   &uniform,
                                   const double mutationChance,
                                   const double factor,
                                   const int deviationHalfPoint,
                                   const int epoch,
                                   const bool independent) {

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
                                                       false);
  }
  for (n = 0; n < numOfOutput; n++) {
    ((GeneticNeuron*) outputNeurons[n])->mutateWeights(gaussian,
                                                       uniform,
                                                       mutationChance,
                                                       currentFactor,
                                                       independent,
                                                       false);
    }
    }*/



// Getters Setters

SelectionMethod GeneticNetwork::getSelectionMethod() const {
  return selectionMethod;
}
void GeneticNetwork::setSelectionMethod(SelectionMethod val) {
  selectionMethod = val;
}

CrossoverMethod GeneticNetwork::getCrossoverMethod() const {
  return crossoverMethod;
}
void GeneticNetwork::setCrossoverMethod(CrossoverMethod val) {
  crossoverMethod = val;
}

FitnessFunction GeneticNetwork::getFitnessFunction() const {
  return fitnessFunction;
}
void GeneticNetwork::setFitnessFunction(FitnessFunction val) {
  fitnessFunction = val;
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
void GeneticNetwork::setWeightMutationHalfPoint(unsigned int
                                                weightMutationHalfPoint) {
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

double GeneticNetwork::getCrossoverChance() const {
  return crossoverChance;
}
void GeneticNetwork::setCrossoverChance(double val) {
  this->crossoverChance = val;
}

/*
double weightSquaredSum2(FFNetwork &net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net.getumOfHidden(); n++) {
    // neuron weights
    sum += net.getHiddenNeuron(n)->getWeightsSquaredSum();
    numOfCons += net.getHiddenNeuron(n)->getNumOfConnections();
  }

  for (n = 0; n < net.getNumOfOutputs(); n++) {
    // Input weights
    sum += net.getOutputNeuron(n)->getWeightsSquaredSum();
    numOfCons += net.getOutputNeuron(n)->getNumOfConnections();
  }

  //printf("Weight squared sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of absolute values of weights (L1 norm)
double weightAbsoluteSum2(FFNetwork &net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net.getNumOfHidden(); n++) {
    // neuron weights
    sum += net.getHiddenNeuron(n)->getWeightsAbsoluteSum();
    numOfCons += net.getHiddenNeuron(n)->getNumOfConnections();
  }

  for (n = 0; n < net.getNumOfOutputs(); n++) {
    // Input weights
    sum += net.getOutputNeuron(n)->getWeightsAbsoluteSum();
    numOfCons += net.getOutputNeuron(n)->getNumOfConnections();
  }
  //printf("Weight absolute sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of soft weight elimination terms
double weightEliminationSum2(FFNetwork &net, double lambda) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net.getNumOfHidden(); n++) {
    // neuron weights
    sum += net.getHiddenNeuron(n)->
      getWeightEliminationSum(lambda);
    numOfCons += net.getHiddenNeuron(n)->getNumOfConnections();
  }

  for (n = 0; n < net.getNumOfOutputs(); n++) {
    // Input weights
    sum += net.getOutputNeuron(n)->
      getWeightEliminationSum(lambda);
    numOfCons += net.getOutputNeuron(n)->getNumOfConnections();
  }
  //printf("Weight elimination sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}
*/

/*
// Utility methods
// Calculate the sum of all weights squared (L2 norm)
double weightSquaredSum(FFNetwork &net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net.getNumOfHidden(); n++) {
    // neuron weights
    sum += net.getHiddenNeuron(n)->getWeightsSquaredSum();
    numOfCons += net.getHiddenNeuron(n)->getNumOfConnections();
  }
  //printf("Weight squared sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of absolute values of weights (L1 norm)
double weightAbsoluteSum(FFNetwork &net) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net.getNumOfHidden(); n++) {
    // neuron weights
    sum += net.getHiddenNeuron(n)->getWeightsAbsoluteSum();
    numOfCons += net.getHiddenNeuron(n)->getNumOfConnections();
  }
  //printf("Weight absolute sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}

// Calculate the sum of soft weight elimination terms
double weightEliminationSum(FFNetwork &net, double lambda) {
  double sum = 0;
  unsigned int n, numOfCons = 0;
  for (n = 0; n < net.getNumOfHidden(); n++) {
    // neuron weights
    sum += net.getHiddenNeuron(n)->
      getWeightEliminationSum(lambda);
    numOfCons += net.getHiddenNeuron(n)->getNumOfConnections();
  }
  //printf("Weight elimination sum: %f\n", sum / (double) numOfCons);
  return sum / (double) numOfCons;
}
*/
