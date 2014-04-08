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
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <ctime>

#include <omp.h>

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
  fitnessFunction(FitnessFunction::FITNESS_MSE),
  weightMutationHalfPoint(0),
  decayL2(0),
  decayL1(0),
  weightElimination(0),
  weightEliminationLambda(0)
{
  //printf("\n Gen Net Constructor\n");
}

double evaluateNetwork(const FitnessFunction fitnessFunction,
                       GeneticNetwork * const pNet,
                       const double * const X,
                       const double * const Y,
                       const unsigned int length,
                       double * const outputs)
{
  unsigned int i;
  for (i = 0; i < length; i++) {
    pNet->output(X + i * pNet->INPUT_COUNT,
                     outputs + i * pNet->OUTPUT_COUNT);
  }
  return getFitness(fitnessFunction,
                    Y, length,
                    pNet->OUTPUT_COUNT,
                    outputs);
}

void insertSorted(vector<GeneticNetwork*>  &sortedPopulation,
                  vector<double> &sortedFitness,
                  const double fitness,
                  GeneticNetwork * const net)
{
  vector<GeneticNetwork*>::iterator netIt;
  vector<double>::iterator fitnessIt;
  bool inserted = false;
  unsigned int j;

  netIt = sortedPopulation.begin();
  fitnessIt = sortedFitness.begin();
  // Insert in sorted position
  for (j = 0; j < sortedPopulation.size(); j++) {
    if (fitness >= fitnessIt[j]) {
      //printf("Inserting at %d, fitness = %f\n", j, fitness);
      sortedPopulation.insert(netIt + j, net);
      sortedFitness.insert(fitnessIt + j, fitness);
      inserted = true;
      break;
    }
  }
  // If empty, or should be placed last in list
  if (!inserted) {
    //printf("Inserting last, fitness = %f\n", fitness);
    sortedPopulation.push_back(net);
    sortedFitness.push_back(fitness);
    inserted = true;
  }
}

void insertLast(vector<GeneticNetwork*>  &sortedPopulation,
                vector<double> &sortedFitness,
                GeneticNetwork * const net)
{
  //printf("Inserting last, error = %f\n", error);
  sortedPopulation.push_back(net);
  sortedFitness.push_back(-99999999999.0);
}

GeneticNetwork *popLastNetwork(vector<GeneticNetwork*> &sortedPopulation,
                               vector<double> &sortedFitness)
{
  GeneticNetwork *pChild = sortedPopulation.back();
  sortedPopulation.pop_back();
  sortedFitness.pop_back();

  return pChild;
}

GeneticNetwork *popNetwork(unsigned int i,
                           vector<GeneticNetwork*> &sortedPopulation,
                           vector<double> &sortedFitness)
{
  // Just in case
  if (i >= sortedPopulation.size()) {
    i = sortedPopulation.size() - 1;
  }
  GeneticNetwork *pChild = sortedPopulation.at(i);
  sortedPopulation.erase(sortedPopulation.begin() + i);
  sortedFitness.erase(sortedFitness.begin() + i);

  return pChild;
}

/**
 * Does the actual work in an epoch. Designed to be launched in parallell
 * in many threads.
 * Because each thread creates its own random generator, it needs
 * a unique thread each time, for which time() is not suitable as it
 * will be the same for all threads many times.
 */
void breedNetworks(GeneticNetwork &self,
                   vector<GeneticNetwork*> &sortedPopulation,
                   vector<double> &sortedFitness,
                   const unsigned int childCount,
                   const unsigned int curGen,
                   const double * const X,
                   const double * const Y,
                   const unsigned int length,
                   const unsigned int randomSeed)
{
#pragma omp parallel default(none) shared(self, sortedPopulation, sortedFitness)
  {
    unsigned int motherIndex, fatherIndex;

    GeneticNetwork *pBrother, *pSister, *pMother, *pFather;

    double bFitness, sFitness, mFitness, fFitness;
    double *outputs = new double[self.OUTPUT_COUNT * length];

    Random rand(randomSeed);

    GeneticSelector selector(rand);
    GeneticMutator mutator(rand);
    GeneticCrosser crosser(rand);

    unsigned int threadChild;
#pragma omp for
    for (threadChild = 0; threadChild < childCount; threadChild++) {
      // We recycle the worst networks

      //JGN_lockPopulation();
#pragma omp critical
      {
        // Select two networks
        selector.getSelection(self.selectionMethod,
                              sortedFitness,
                              self.populationSize,
                              &motherIndex, &fatherIndex);

        pMother = popNetwork(motherIndex, sortedPopulation, sortedFitness);
        pFather = popNetwork(fatherIndex, sortedPopulation, sortedFitness);

        pBrother = popLastNetwork(sortedPopulation, sortedFitness);
        pSister = popLastNetwork(sortedPopulation, sortedFitness);
      }
      // End critical
      //JGN_unlockPopulation();

      // Crossover
      if (rand.uniform() < self.crossoverChance) {
        crosser.evaluateCrossoverFunction(self.crossoverMethod,
                                          *pMother, *pFather,
                                          *pBrother, *pSister);
      }
      // No crossover, mutate parents
      else {
        pSister->cloneNetwork(*pMother);
        pBrother->cloneNetwork(*pFather);
      }


      // Mutation brother
      mutator.mutateWeights(*pBrother,
                            self.weightMutationChance,
                            self.weightMutationFactor);
      mutator.mutateConns(*pBrother,
                          self.connsMutationChance);
      mutator.mutateActFuncs(*pBrother,
                             self.actFuncMutationChance);
      // Mutation sister
      mutator.mutateWeights(*pSister,
                            self.weightMutationChance,
                            self.weightMutationFactor);
      mutator.mutateConns(*pSister,
                          self.connsMutationChance);
      mutator.mutateActFuncs(*pSister,
                             self.actFuncMutationChance);


      // evaluate fitness and insert back
      // TODO multiple outputs
      bFitness = evaluateNetwork(self.fitnessFunction,
                                 pBrother, X, Y, length,
                                 outputs);
      sFitness = evaluateNetwork(self.fitnessFunction,
                                 pSister, X, Y, length,
                                 outputs);
      mFitness = evaluateNetwork(self.fitnessFunction,
                                 pMother, X, Y, length,
                                 outputs);
      fFitness = evaluateNetwork(self.fitnessFunction,
                                 pFather, X, Y, length,
                                 outputs);

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
      //JGN_lockPopulation();
#pragma omp critical
      {
        // Place parents back
        insertSorted(sortedPopulation, sortedFitness, mFitness, pMother);
        insertSorted(sortedPopulation, sortedFitness, fFitness, pFather);

        // Place children back
        insertSorted(sortedPopulation, sortedFitness, bFitness, pBrother);
        insertSorted(sortedPopulation, sortedFitness, sFitness, pSister);
      }
      // End critical
      //JGN_unlockPopulation();
    }
    // End for
    delete[] outputs;
  }
  // End parallel
}

void GeneticNetwork::learn(const double * const X,
                           const double * const Y,
                           const unsigned int length) {
  // Reset LOG
  initLog(generations * this->OUTPUT_COUNT);

  // Create a population of networks
  vector<GeneticNetwork*> sortedPopulation;
  vector<double> sortedFitness;

  // Number of threads to use, TODO, dont hardcode this
  //unsigned int num_threads = 8;
  //unsigned int extras = num_threads * 4;
  //std::thread threads[num_threads];
  //  vector<thread> threads;
  // Each thread will breed these many networks each generation
  //unsigned int breedCount = round((double) populationSize / num_threads);
  //if (breedCount < 1) {
  //  breedCount = 1;
  //}

  unsigned int extras;
#pragma omp parallel
  {
#pragma omp master
    {
      extras = 4 * omp_get_num_threads();
    }
    // End master
  }
  // End parallel

  // Pre-allocate space
  sortedPopulation.reserve(populationSize + extras);
  sortedFitness.reserve(populationSize + extras);

  // Rank and insert them in a sorted order
  unsigned int i;
  vector<GeneticNetwork*>::iterator netIt;
  vector<double>::iterator errorIt;

  double fitness = 0;
  // predictions
  double *preds = new double[OUTPUT_COUNT * length]();

  Random rand;
  GeneticMutator mutator(rand);

  for (i = 0; i < populationSize + extras; i++) {
    // use references
    GeneticNetwork *pNet = new GeneticNetwork(INPUT_COUNT,
                                              HIDDEN_COUNT,
                                              OUTPUT_COUNT);

    // Base it on the all-mother
    pNet->cloneNetwork(*this);
    // Respect mutation chances.
    mutator.mutateWeights(*pNet, weightMutationChance,
                          weightMutationFactor);
    mutator.mutateConns(*pNet, connsMutationChance);
    mutator.mutateActFuncs(*pNet, actFuncMutationChance);

    // evaluate error here
    fitness = evaluateNetwork(fitnessFunction,
                              pNet, X, Y, length,
                              preds);

    insertSorted(sortedPopulation, sortedFitness, fitness, pNet);
  }

  // Save the best network in the population
  GeneticNetwork *best = sortedPopulation.front();

  // For each generation
  unsigned int curGen;
  for (curGen = 0; curGen < generations; curGen++) {
    time_t start, end;
    time(&start);

    //for (i = 0; i < num_threads; i++) {
    //      try {
        //        threads.push_back(std::thread(
        breedNetworks(std::ref(*this),
                      std::ref(sortedPopulation),
                      std::ref(sortedFitness),
                      populationSize, curGen,
                      X, Y, length, rand.uint());
            //                            ));

        //      }
        //catch (const std::system_error& e) {
        //std::cout << "Caught system_error with code " << e.code()
        //          << " meaning " << e.what() << '\n';
        //throw(e);
        //}
      //}

    // Wait for the threads to finish their work
    // for (auto t = threads.begin(); t != threads.end(); t++) {
    //   try {
    //     t->join();
    //   }
    //   catch (const std::system_error& e) {
    //     std::cout << "Caught system_error with code " << e.code()
    //               << " meaning " << e.what() << '\n';
    //     throw(e);
    //   }
    // }
    //cout << "\nJoined threads\n";
        //threads.clear();
    time(&end);
    //    std::cout << "gen time: " << difftime(end, start) << "s" << std::endl;

    //JGN_lockPopulation();

    // Print some stats about the current best
    best = sortedPopulation.front();

    // Save in log
    //cout << "\nLogging...\n";
    this->aLogPerf[curGen] = sortedFitness.front();
    //cout << "\nLogged\n";

    //JGN_unlockPopulation();

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

  this->cloneNetwork(*best);

  // And destroy population
  // do this last of all!
  best = NULL;
  for (netIt = sortedPopulation.begin(); netIt < sortedPopulation.end();
       netIt++) {
    delete *netIt;
  }
  delete[] preds;
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


void GeneticNetwork::cloneNetwork(GeneticNetwork &original) {
  std::copy(original.weights,
            original.weights + original.LENGTH * original.LENGTH,
            this->weights);

  std::copy(original.conns,
            original.conns + original.LENGTH * original.LENGTH,
            this->conns);

  std::copy(original.actFuncs,
            original.actFuncs + original.LENGTH,
            this->actFuncs);
}


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

double GeneticNetwork::getConnsMutationChance() const {
  return connsMutationChance;
}
void GeneticNetwork::setConnsMutationChance(double chance) {
  this->connsMutationChance = chance;
}

double GeneticNetwork::getActFuncMutationChance() const {
  return actFuncMutationChance;
}
void GeneticNetwork::setActFuncMutationChance(double chance) {
  this->actFuncMutationChance = chance;
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
