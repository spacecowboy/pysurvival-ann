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
  weightMutationHalfPoint(0),
  decayL2(0),
  decayL1(0),
  weightElimination(0),
  weightEliminationLambda(0),
  resume(false),
  crossoverChance(1.0),
  selectionMethod(SelectionMethod::SELECTION_GEOMETRIC),
  crossoverMethod(CrossoverMethod::CROSSOVER_ONEPOINT),
//insertMethod(INSERT_ALL),
  fitnessFunction(FitnessFunction::FITNESS_MSE)
{
}

// Thread safe
void GeneticNetwork::insertSorted(vector<GeneticNetwork*>  &sortedPopulation,
                                  vector<double> &sortedErrors,
                                  const double error,
                                  GeneticNetwork * const net)
{
  JGN_lockPopulation();

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

  JGN_unlockPopulation();
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

    GeneticNetwork motherNet(self->getNumOfInputs(),
                             self->getNumOfHidden(),
                             self->getNumOfOutputs());
    GeneticNetwork fatherNet(self->getNumOfInputs(),
                             self->getNumOfHidden(),
                             self->getNumOfOutputs());
    motherNet.initNodes();
    fatherNet.initNodes();

    unsigned int motherIndex, fatherIndex;

    GeneticNetwork *pChild;

    double error;
    double *outputs = new double[self->numOfOutput*length];

    unsigned int threadChild;
    for (threadChild = 0; threadChild < childCount; threadChild++) {
        // We recycle the worst network
        pChild = self->popLastNetwork(*sortedPopulation, *sortedErrors);

        // Select two networks
        switch (self->selectionMethod) {
        case SELECTION_ROULETTE:
            selectParentsRoulette(*self, *uniform, *sortedErrors,
                                  motherIndex, fatherIndex,
                                  self->populationSize);
            break;
        case SELECTION_TOURNAMENT:
            selectParentsTournament(*uniform,
                                    motherIndex, fatherIndex,
                                    self->populationSize);
            break;
        case SELECTION_GEOMETRIC:
        default:
            selectParents(*geometric, self->populationSize,
                          motherIndex, fatherIndex);
            break;
        }
        // get mother and father in a thread safe way
        self->cloneParents(motherNet, fatherNet, *sortedPopulation,
                           motherIndex, fatherIndex);

        // Create new child through crossover
        pChild->crossover(*uniform, motherNet, fatherNet);

        // Mutate child
        pChild->mutateWeights(*gaussian, *uniform, self->weightMutationChance,
                              self->weightMutationFactor,
                              self->weightMutationHalfPoint,
                              curGen, false);
        // evaluate error child
        error = -(*(self->pFitnessFunction))(*pChild, X, Y, length, outputs);

        // Weight decay terms
        // Up to the user to (not) mix these
        // L2 weight decay
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

        // Insert child into the sorted list
        self->insertSorted(*sortedPopulation, *sortedErrors, error, pChild);
    }
    delete[] outputs;
}


void GeneticNetwork::learn(const double * const X,
                           const double * const Y,
                           const unsigned int length) {
  // Setup algorithm stuff
  this->pFitnessFunction =
    getFitnessFunctionPtr(fitnessFunctionType);

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
  unsigned int extras = num_threads;
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
  unsigned int i;
  vector<GeneticNetwork*>::iterator netIt;
  vector<double>::iterator errorIt;

  printf("Data size: %d\n", length);

  double error;
  double *outputs = new double[numOfOutput*length];
  for (i = 0; i < populationSize + extras; i++) {
    // use references
    GeneticNetwork *pNet = new GeneticNetwork(INPUT_COUNT,
                                              HIDDEN_COUNT,
                                              OUTPUT_COUNT);

    randomizeNetwork(*pNet, weightMutationFactor);

    // evaluate error here
    error = -(*pFitnessFunction)(*pNet, X, Y, length, outputs);
    //    error = evaluateNetwork(*pNet, X, Y, length, outputs);
    // TODO use references on lists
    insertSorted(sortedPopulation, sortedErrors, error, pNet);
  }

  // Save the best network in the population
  GeneticNetwork *best = sortedPopulation.front();

  // For each generation
  unsigned int curGen;
  for (curGen = 0; curGen < generations; curGen++) {
    time_t start, end;
    time(&start);
    for (i = 0; i < num_threads; ++i) {
      threads[i] = std::thread(breedNetworks, this, &gaussian, &geometric,
                               &uniform, &sortedPopulation,
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
    this->aLogPerf[curGen] = -sortedErrors.front();

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
  }

  // When done, make this network into the best network
  printf("best eval fitness: %f\n", ((*pFitnessFunction)(*best, X, Y,
                                                     length, outputs)));
  this->cloneNetworkSlow(*best);
  printf("this eval fitness: %f\n", ((*pFitnessFunction)(*this, X, Y,
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

GeneticNetwork *GeneticNetwork::popLastNetwork(
    vector<GeneticNetwork*> &sortedPopulation,
    vector<double> &sortedErrors)
{
  JGN_lockPopulation();
  GeneticNetwork *pChild = sortedPopulation.back();
  sortedPopulation.pop_back();
  sortedErrors.pop_back();
  JGN_unlockPopulation();
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

bool GeneticNetwork::getResume()const {
  return resume;
}
void GeneticNetwork::setResume(bool val) {
  this->resume = val;
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
  for (n = 0; n < net.getNumOfHidden(); n++) {
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
