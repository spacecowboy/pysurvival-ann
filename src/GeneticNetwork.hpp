#ifndef _GENETICNETWORK_HPP_
#define _GENETICNETWORK_HPP_

#include "MatrixNetwork.hpp"
#include "GeneticFitness.hpp"
#include "GeneticSelection.hpp"
#include "GeneticCrossover.hpp"
#include "Statistics.hpp"
#include <vector>


class GeneticNetwork: public MatrixNetwork {
 public:
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
  // Chance to flip a bit
  double connsMutationChance;
  // Chance to change activationFunction
  // ex: 10% means 90% chance to stay the same (linear for ex.),
  // and 5% chance to select logsig and 5% tanh.
  double actFuncMutationChance;

  // Chance of doing crossover
  double crossoverChance;

  SelectionMethod selectionMethod;
  CrossoverMethod crossoverMethod;
  FitnessFunction fitnessFunction;

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

  // Statistic used if fitness function is TaroneWare
  TaroneWareType taroneWareStatistic;
  // Used to penalize small groups
  unsigned int minGroup;


  // Methods
  GeneticNetwork(const unsigned int numOfInputs,
                 const unsigned int numOfHidden,
                 const unsigned int numOfOutputs);

/*
 * Expects the X and Y to be of equal number of rows.
 */
  virtual void learn(const std::vector<double> &X,
                     const std::vector<double> &Y,
                     const unsigned int length);

  /**
   * Performs output, then returns the fitness of those predictions.
   */
  double getPredictionFitness(const std::vector<double> &X,
                              const std::vector<double> &Y,
                              const unsigned int length);

// Makes this network into a clone of the original. Assumes equal iteration.
  virtual void cloneNetwork(GeneticNetwork &original);

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

  double getConnsMutationChance() const;
  void setConnsMutationChance(double chance);
  double getActFuncMutationChance() const;
  void setActFuncMutationChance(double chance);

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

  SelectionMethod getSelectionMethod() const;
  void setSelectionMethod(SelectionMethod val);

  CrossoverMethod getCrossoverMethod() const;
  void setCrossoverMethod(CrossoverMethod val);

  //  insert_method_t getInsertMethod() const;
  //  void setInsertMethod(long val);

  FitnessFunction getFitnessFunction() const;
  void setFitnessFunction(FitnessFunction val);

  TaroneWareType getTaroneWareStatistic() const;
  void setTaroneWareStatistic(TaroneWareType stat);

  unsigned int getMinGroup() const;
  void setMinGroup(unsigned int n);

};

// Calculate the sum of all weights squared (L2 norm)
//double weightSquaredSum(FFNetwork &net);

// Calculate the sum of absolute values of weights (L1 norm)
//double weightAbsoluteSum(FFNetwork &net);

// Calculate the sum of soft weight elimination terms
//double weightEliminationSum(FFNetwork &net, double lambda);


#endif /* _GENETICNETWORK_HPP_ */
