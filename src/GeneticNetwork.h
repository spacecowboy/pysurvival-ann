#ifndef _GENETICNETWORK_H_
#define _GENETICNETWORK_H_

#include "FFNetwork.h"
#include "FFNeuron.h"
#include "GeneticFitness.h"
#include "boost/random.hpp"
#include <mutex>
#include <vector>

using namespace std;

enum selection_method_t { SELECTION_GEOMETRIC,
                          SELECTION_ROULETTE,
                          SELECTION_TOURNAMENT };

enum crossover_method_t { CROSSOVER_NEURON,
                          CROSSOVER_TWOPOINT };

enum insert_method_t { INSERT_ALL,
                       INSERT_FITTEST };

class GeneticNetwork: public FFNetwork {
 public:
    // Used to lock stuff to become thread safe
  std::mutex lockMutex;
  // If the network should resume from its existing weights
  // If false, will generate and independent population
  bool resume;

  // Training variables
  // How many networks should be created/mutated and compared in one generation
  unsigned int populationSize;
  // Train for this many generations. E.g. total number of networks
  // created/mutated is
  // populationSize * generations
  unsigned int generations;
  // If uniformRandom < weightMutationChance then mutate weight by
  // mutationChange
  double weightMutationChance;
  // Scales the mutation variance.
  double weightMutationFactor;

  // Chance of doing crossover
  double crossoverChance;

  selection_method_t selectionMethod;
  crossover_method_t crossoverMethod;
  insert_method_t insertMethod;
  fitness_function_t fitnessFunctionType;
  fitness_func_ptr pFitnessFunction;

  // If this is non zero, it is interpreted as the generation where the stddev
  // should have decreased to half its value.
  // This calculation is NOT done if this is zero, which it is by default.
  unsigned int weightMutationHalfPoint;

  // Simple weight decay factor for the L2 norm: lambda * Sum(W^2)
  double decayL2;
  // Weight decay factor for the L1 norm: lambda * Sum(abs(W))
  double decayL1;

  // Preceding factor (g) in P = g * sum()
  double weightElimination;
  // Factor (l) for soft weight elimination: P = sum( w^2 / (l^2 + w^2) )
  double weightEliminationLambda;

  // Methods
  GeneticNetwork(const unsigned int numOfInputs, const unsigned int numOfHidden,
                 const unsigned int numOfOutputs);

  virtual void initNodes();

  /*
   * Evaluates a network, including possible weight decays
   */
virtual double evaluateNetwork(GeneticNetwork &net,
                                   const double * const X,
                                   const double * const Y,
                                   const unsigned int length,
                                   double * const outputs);

/*
 * Expects the X and Y to be of equal number of rows.
 */
virtual void learn(const double * const X,
                   const double * const Y,
                   const unsigned int length);

// Make this network into a mixture of the mother and father
virtual void crossover(
    boost::variate_generator<boost::mt19937&,
    boost::uniform_real<> > &uniform,
    GeneticNetwork &mother, GeneticNetwork &father);

// Randomly mutates the weights of the network.
// Expects a gaussian distribution with
// mean 0 and stdev 1.
virtual void mutateWeights(
    boost::variate_generator<boost::mt19937&,
    boost::normal_distribution<double> > &gaussian,
    boost::variate_generator<boost::mt19937&,
    boost::uniform_real<> > &uniform,
    const double mutationChance, const double stdDev,
    const int deviationHalfPoint, const int epoch,
    const bool independent);

// Makes this network into a clone of the original. Assumes equal iteration.
virtual void cloneNetwork(GeneticNetwork &original);
// Makes this network into a clone of the original. Does NOT assume same order.
virtual void cloneNetworkSlow(GeneticNetwork &original);

// Used to build initial population
virtual GeneticNetwork*
getGeneticNetwork(GeneticNetwork &cloner,
                      boost::variate_generator<boost::mt19937&,
                                               boost::normal_distribution<double> >
                      &gaussian,
                      boost::variate_generator<boost::mt19937&,
                                               boost::uniform_real<> >
                      &uniform);

// Insert network back into the population
// Method because of thread stuff
void insertSorted(vector<GeneticNetwork*>  &sortedPopulation,
                  vector<double> &sortedErrors,
                  const double error,
                  GeneticNetwork * const net);
// Clone Parents thread safe
void cloneParents(GeneticNetwork &mother,
                  GeneticNetwork &father,
                  vector<GeneticNetwork*> &sortedPopulation,
                  const unsigned int motherIndex,
                  const unsigned int fatherIndex);

GeneticNetwork *popLastNetwork(
    vector<GeneticNetwork*> &sortedPopulation,
    vector<double> &sortedErrors);

// Learning work done by threads here
/*
void breedNetworks(
    boost::variate_generator<boost::mt19937&,
    boost::normal_distribution<double> > &gaussian,
    boost::variate_generator<boost::mt19937&,
    boost::geometric_distribution<int, double> > &geometric,
    boost::variate_generator<boost::mt19937&,
    boost::uniform_real<> > &uniform,
    vector<GeneticNetwork*> &sortedPopulation,
    vector<double> &sortedErrors,
    const unsigned int childCount,
    const unsigned int curGen,
    const double * const X, const double * const Y,
    const unsigned int length);
*/
  unsigned int getGenerations() const;
  void setGenerations(unsigned int generations);
  unsigned int getPopulationSize() const;
  void setPopulationSize(unsigned int populationSize);
  double getWeightMutationChance() const;
  void setWeightMutationChance(double weightMutationChance);
  unsigned int getWeightMutationHalfPoint() const;
  void setWeightMutationHalfPoint(unsigned int weightMutationHalfPoint);
  double getWeightMutationFactor() const;
  void setWeightMutationFactor(double weightMutationFactor);

  double getDecayL1() const;
  void setDecayL1(double val);
  double getDecayL2() const;
  void setDecayL2(double val);

  double getWeightElimination() const;
  void setWeightElimination(double val);
  double getWeightEliminationLambda() const;
  void setWeightEliminationLambda(double val);

  bool getResume() const;
  void setResume(bool val);

  double getCrossoverChance() const;
  void setCrossoverChance(double val);

  selection_method_t getSelectionMethod() const;
  void setSelectionMethod(long val);

  crossover_method_t getCrossoverMethod() const;
  void setCrossoverMethod(long val);

  insert_method_t getInsertMethod() const;
  void setInsertMethod(long val);

  fitness_function_t getFitnessFunction() const;
  void setFitnessFunction(long val);
};

class GeneticNeuron: public Neuron {
public:
  GeneticNeuron(int id);
  GeneticNeuron(int id, double (*activationFunction)(double),
                double (*activationDerivative)(double));
  virtual ~GeneticNeuron();

  // Makes this neuron into a copy of the original. Assumes equal structure
  // Only copies the weights for the connections with equal number
  virtual void cloneNeuron(Neuron *original);
  // Does not assume equal iteration order
  virtual void cloneNeuronSlow(Neuron *original);
  // With some probability will change the weights.
  // If independant is true, will replace all weights and make sure
  // the vector is scaled by the l2 norm.
  virtual void mutateWeights(boost::variate_generator<boost::mt19937&,
                             boost::normal_distribution<double> > &gaussian,
                             boost::variate_generator<boost::mt19937&,
                             boost::uniform_real<> > &uniform,
                             const double mutationChance, const double stdDev,
                             const bool independent, const bool l2scale);
};

class GeneticBias: public GeneticNeuron {
public:
 GeneticBias() :
  GeneticNeuron(-1) {
  }
  virtual double output() {
    return 1;
  }
  virtual double output(double * inputs) {
    return 1;
  }
  virtual double outputDeriv() {
    return 0;
  }
};

// Calculate the sum of all weights squared (L2 norm)
double weightSquaredSum(FFNetwork &net);

// Calculate the sum of absolute values of weights (L1 norm)
double weightAbsoluteSum(FFNetwork &net);

// Calculate the sum of soft weight elimination terms
double weightEliminationSum(FFNetwork &net, double lambda);


#endif /* _GENETICNETWORK_H_ */
