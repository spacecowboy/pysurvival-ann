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
#include "Statistics.hpp"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <omp.h>


GeneticNetwork::GeneticNetwork(const unsigned int numOfInputs,
                               const unsigned int numOfHidden,
                               const unsigned int numOfOutputs) :
  MatrixNetwork(numOfInputs, numOfHidden, numOfOutputs),
  populationSize(50),
  generations(100),
  weightMutationChance(0.15),
  weightMutationFactor(1.0),
  connsMutationChance(0.0),
  actFuncMutationChance(0.0),
  crossoverChance(1.0),
  selectionMethod(SelectionMethod::SELECTION_GEOMETRIC),
  crossoverMethod(CrossoverMethod::CROSSOVER_ONEPOINT),
  fitnessFunction(FitnessFunction::FITNESS_MSE),
  weightMutationHalfPoint(0),
  decayL2(0),
  decayL1(0),
  weightElimination(0),
  weightEliminationLambda(0),
  taroneWareStatistic(TaroneWareType::LOGRANK),
  minGroup(1)
{
  //printf("\n Gen Net Constructor\n");
}

double getClassFitness(const FitnessFunction func,
                       GeneticNetwork &net,
                       const std::vector<double> &X,
                       const std::vector<double> &Y,
                       const unsigned int length,
                       std::vector<double> &outputs)
{
  double retval, winningOutput, outVal;
  unsigned int i, j, nextGroup, groupCount = 0, winningGroup;
  // Index of each output neuron
  std::vector<unsigned int> activeOutputs(net.OUTPUT_COUNT, 0);
  // Store number of members in each group
  std::vector<unsigned int> groupCounts(net.OUTPUT_COUNT, 0);

  // Group membership
  std::vector<unsigned int> groups(length, 0);

  // Count active output neurons
  for (i = net.OUTPUT_START; i < net.OUTPUT_END; i++) {
    // Check diagonal connections
    if (net.conns.at(i * net.LENGTH + i) == 1) {
      // Active, remember index
      activeOutputs.at(groupCount) = i - net.OUTPUT_START;
      groupCount++;
    }
  }

  //printf("Length: %i, Groups: %i\n", length, groupCount);

  // Classify each pattern, increasing group counts etc as we go along
  for (i = 0; i < length; i++) {
    net.output(X.begin() + i * net.INPUT_COUNT, false,
                 outputs.begin() + i * net.OUTPUT_COUNT);
    // Start with first group
    nextGroup = 0;
    // Default value is first active neuron
    winningGroup = nextGroup;
    winningOutput = outputs.at(i * net.OUTPUT_COUNT + activeOutputs.at(nextGroup));
    for (j = 0; j < net.OUTPUT_COUNT; j++) {
      // Ignore inactive outputs
      if (j == activeOutputs.at(nextGroup)) {
        outVal = outputs.at(i * net.OUTPUT_COUNT + activeOutputs.at(nextGroup));
        // greater output, possible winner
        if (outVal > winningOutput) {
          winningGroup = nextGroup;
          winningOutput = outVal;
        }
        // Check next group, next time
        nextGroup++;
      } else {
        // This output is not active, ignore it
      }
    }

    // Increment count of group members
    groupCounts.at(winningGroup) += 1;
    // And note winner
    groups.at(i) = winningGroup;
  }


  // Evaluate performance
  // Each child is responsible for carrying information about fitness function
  switch (func) {
  case FitnessFunction::FITNESS_TARONEWARE_MEAN:
    retval = TaroneWareMeanPairwise(Y, groups, groupCounts, length,
                                    net.getTaroneWareStatistic());
    break;
  case FitnessFunction::FITNESS_TARONEWARE_HIGHLOW:
    retval = TaroneWareHighLow(Y, groups, groupCounts, length, groupCount,
                               net.getTaroneWareStatistic());
    // Penalize small groups
    if (groupCounts.at(0) < net.getMinGroup()) {
      retval = 0;
    }
    if (groupCount > 1 && groupCounts.at(1) < net.getMinGroup()) {
      retval = 0;
    }
    break;
  case FitnessFunction::FITNESS_SURV_KAPLAN_MAX:
  case FitnessFunction::FITNESS_SURV_KAPLAN_MIN:
    retval = SurvArea(Y, groups, groupCounts, length,
                      FitnessFunction::FITNESS_SURV_KAPLAN_MIN == func);

    // Penalize small groups
    if (groupCounts.at(0) < net.getMinGroup()) {
      retval = 0;
    }
    break;
  case FitnessFunction::FITNESS_SURV_RISKGROUP_HIGH:
  case FitnessFunction::FITNESS_SURV_RISKGROUP_LOW:
    retval = RiskGroup(Y, groups, groupCounts, length,
                       FitnessFunction::FITNESS_SURV_RISKGROUP_HIGH == func);
    // Penalize small groups
    if (groupCounts.at(0) < net.getMinGroup()) {
      retval = 0;
    }
    break;

  default:
    retval = 0;
    break;
  }

  return retval;
}

double evaluateNetwork(const FitnessFunction fitnessFunction,
                       GeneticNetwork &net,
                       const std::vector<double> &X,
                       const std::vector<double> &Y,
                       const unsigned int length,
                       std::vector<double> &outputs)
{
  if (FitnessFunction::FITNESS_TARONEWARE_MEAN == fitnessFunction ||
      FitnessFunction::FITNESS_TARONEWARE_HIGHLOW == fitnessFunction ||
      FitnessFunction::FITNESS_SURV_KAPLAN_MAX == fitnessFunction ||
      FitnessFunction::FITNESS_SURV_KAPLAN_MIN == fitnessFunction ||
      FitnessFunction::FITNESS_SURV_RISKGROUP_HIGH == fitnessFunction ||
      FitnessFunction::FITNESS_SURV_RISKGROUP_LOW == fitnessFunction){
    // Need to support a slightly different API for groups
    return getClassFitness(fitnessFunction,
                           net,
                           X, Y, length,
                           outputs);
  }
  unsigned int i;

  for (i = 0; i < length; i++) {
    net.output(X.begin() + i * net.INPUT_COUNT, false,
                 outputs.begin() + i * net.OUTPUT_COUNT);
  }

  return getFitness(fitnessFunction,
                    Y, length,
                    net.OUTPUT_COUNT,
                    outputs);
}

void insertSorted(std::vector<GeneticNetwork*>  &sortedPopulation,
                  std::vector<double> &sortedFitness,
                  const double fitness,
                  GeneticNetwork * const net)
{
  std::vector<GeneticNetwork*>::iterator netIt;
  std::vector<double>::iterator fitnessIt;
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

void insertLast(std::vector<GeneticNetwork*>  &sortedPopulation,
                std::vector<double> &sortedFitness,
                GeneticNetwork * const net)
{
  //printf("Inserting last, error = %f\n", error);
  sortedPopulation.push_back(net);
  sortedFitness.push_back(-99999999999.0);
}

GeneticNetwork *popLastNetwork(std::vector<GeneticNetwork*> &sortedPopulation,
                               std::vector<double> &sortedFitness)
{
  GeneticNetwork *pChild = sortedPopulation.back();
  sortedPopulation.pop_back();
  sortedFitness.pop_back();

  return pChild;
}

GeneticNetwork *popNetwork(unsigned int i,
                           std::vector<GeneticNetwork*> &sortedPopulation,
                           std::vector<double> &sortedFitness)
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
 */
void breedNetworks(GeneticNetwork &self,
                   std::vector<GeneticNetwork*> &sortedPopulation,
                   std::vector<double> &sortedFitness,
                   const unsigned int childCount,
                   const unsigned int curGen,
                   const std::vector<double> &X,
                   const std::vector<double> &Y,
                   const unsigned int length)
{
#pragma omp parallel default(none) shared(self, X, Y, sortedPopulation, sortedFitness)
  {
    unsigned int motherIndex, fatherIndex;

    GeneticNetwork *pBrother, *pSister, *pMother, *pFather;

    double bFitness, sFitness, mFitness, fFitness;
    std::vector<double> outputs(self.OUTPUT_COUNT * length, 0.0);

    Random rand;

    GeneticSelector selector(rand);
    GeneticMutator mutator(rand);
    GeneticCrosser crosser(rand);

    unsigned int threadChild;
#pragma omp for
    for (threadChild = 0; threadChild < childCount; threadChild++) {
      // We recycle the worst networks
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

      // Do dropout if configured
      pBrother->dropoutConns();

      pSister->dropoutConns();

      pMother->dropoutConns();

      pFather->dropoutConns();

      // Evaluate fitness
      bFitness = evaluateNetwork(self.fitnessFunction,
                                 *pBrother, X, Y, length,
                                 outputs);
      sFitness = evaluateNetwork(self.fitnessFunction,
                                 *pSister, X, Y, length,
                                 outputs);
      mFitness = evaluateNetwork(self.fitnessFunction,
                                 *pMother, X, Y, length,
                                 outputs);
      fFitness = evaluateNetwork(self.fitnessFunction,
                                 *pFather, X, Y, length,
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
    }
    // End for
  }
  // End parallel
}

double GeneticNetwork::getPredictionFitness(const std::vector<double> &X,
                                            const std::vector<double> &Y,
                                            const unsigned int length) {
  std::vector<double> outputs(this->OUTPUT_COUNT * length, 0.0);
  return evaluateNetwork(this->getFitnessFunction(),
                         *this,
                         X,
                         Y,
                         length,
                         outputs);
}

void GeneticNetwork::learn(const std::vector<double> &X,
                           const std::vector<double> &Y,
                           const unsigned int length) {
  // Reset LOG
  initLog(generations * this->OUTPUT_COUNT);

  // Create a population of networks
  std::vector<GeneticNetwork*> sortedPopulation;
  std::vector<double> sortedFitness;

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
  std::vector<GeneticNetwork*>::iterator netIt;
  std::vector<double>::iterator errorIt;

  double fitness = 0;
  // predictions
  std::vector<double> preds(OUTPUT_COUNT * length, 0.0);

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

    // Do dropout if configured
    pNet->dropoutConns();

    // evaluate error here
    fitness = evaluateNetwork(fitnessFunction,
                              *pNet, X, Y, length,
                              preds);

    insertSorted(sortedPopulation, sortedFitness, fitness, pNet);
  }

  // Save the best network in the population
  GeneticNetwork *best = sortedPopulation.front();

  // For each generation
  unsigned int curGen;
  for (curGen = 0; curGen < generations; curGen++) {
    breedNetworks(*this,
                  sortedPopulation,
                  sortedFitness,
                  populationSize, curGen,
                  X, Y, length);
    best = sortedPopulation.front();

    // Save in log for all neurons
    for (unsigned int node = 0; node < OUTPUT_COUNT; node++) {
      this->aLogPerf.at(OUTPUT_COUNT * curGen + node) = sortedFitness.front();
    }

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

  // Undo dropout again if configured
  dropoutReset();

  // And destroy population
  // do this last of all!
  best = nullptr;
  for (netIt = sortedPopulation.begin(); netIt < sortedPopulation.end();
       netIt++) {
    delete *netIt;
  }
}


void GeneticNetwork::cloneNetwork(GeneticNetwork &original) {
  // Need to have the same statistic set on all
  this->setTaroneWareStatistic(original.getTaroneWareStatistic());
  this->setMinGroup(original.getMinGroup());

  // Now copy all structure
  std::copy(original.weights.begin(),
            original.weights.end(),
            this->weights.begin());

  std::copy(original.conns.begin(),
            original.conns.end(),
            this->conns.begin());

  std::copy(original.actFuncs.begin(),
            original.actFuncs.end(),
            this->actFuncs.begin());

  std::copy(original.dropoutProbs.begin(),
            original.dropoutProbs.end(),
            this->dropoutProbs.begin());

  std::copy(original.dropoutStatus.begin(),
            original.dropoutStatus.end(),
            this->dropoutStatus.begin());
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

TaroneWareType GeneticNetwork::getTaroneWareStatistic() const {
  return taroneWareStatistic;
}
void GeneticNetwork::setTaroneWareStatistic(TaroneWareType stat) {
  this->taroneWareStatistic = stat;
}

unsigned int GeneticNetwork::getMinGroup() const {
  return minGroup;
}
void GeneticNetwork::setMinGroup(unsigned int n) {
  this->minGroup = n;
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
