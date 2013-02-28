#ifndef GENETICSURVIVALNETWORK_H_
#define GENETICSURVIVALNETWORK_H_

#include "FFNetwork.h"
#include "FFNeuron.h"
#include "boost/random.hpp"

class GeneticSurvivalNetwork: public FFNetwork {
 public:
  // Training variables
  // How many networks should be created/mutated and compared in one generation
  unsigned int populationSize;
  // Train for this many generations. E.g. total number of networks created/mutated is
  // populationSize * generations
  unsigned int generations;
  // If uniformRandom < weightMutationChance then mutate weight by mutationChange
  double weightMutationChance;
  // This is the stddev in: mutationChange = random.gaussianDistribution(mean=0, stddev)
  // This is the maximum the stddev can be. If weightMutationHalfPoint is set, it can be less
  double weightMutationStdDev;

  // If this is non zero, it is interpreted as the generation where the stddev should have
  // decreased to half its value.
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
  GeneticSurvivalNetwork(unsigned int numOfInputs, unsigned int numOfHidden);
  void initNodes();

  /*
   * Evaluates a network, including possible weight decays
   */
  double evaluateNetwork(GeneticSurvivalNetwork *net, double *X, double *Y,
                         unsigned int length, double *outputs);

  /*
   * Expects the X and Y to be of equal number of rows. Y has 2 columns,
   * first being survival time, second being event (1 or 0)
   */
  void learn(double *X, double *Y, unsigned int length);

  // Make this network into a mixture of the mother and father
  void crossover(
                 boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
                 GeneticSurvivalNetwork *mother, GeneticSurvivalNetwork *father);

  // Randomly mutates the weights of the network. Expects a gaussian distribution with
  // mean 0 and stdev 1.
  void mutateWeights(
                     boost::variate_generator<boost::mt19937&,
                     boost::normal_distribution<double> > *gaussian,
                     boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
                     double mutationChance, double stdDev,
                     int deviationHalfPoint, int epoch);

  // Makes this network into a clone of the original. Assumes equal iteration.
  void cloneNetwork(GeneticSurvivalNetwork *original);
  // Makes this network into a clone of the original. Does NOT assume same order.
  void cloneNetworkSlow(GeneticSurvivalNetwork *original);
  unsigned int getGenerations() const;
  void setGenerations(unsigned int generations);
  unsigned int getPopulationSize() const;
  void setPopulationSize(unsigned int populationSize);
  double getWeightMutationChance() const;
  void setWeightMutationChance(double weightMutationChance);
  unsigned int getWeightMutationHalfPoint() const;
  void setWeightMutationHalfPoint(unsigned int weightMutationHalfPoint);
  double getWeightMutationStdDev() const;
  void setWeightMutationStdDev(double weightMutationStdDev);

  double getDecayL1() const;
  void setDecayL1(double val);
  double getDecayL2() const;
  void setDecayL2(double val);

  double getWeightElimination() const;
  void setWeightElimination(double val);
  double getWeightEliminationLambda() const;
  void setWeightEliminationLambda(double val);

};

class GeneticSurvivalNeuron: public Neuron {
public:
	GeneticSurvivalNeuron(int id);
	GeneticSurvivalNeuron(int id, double (*activationFunction)(double),
			double (*activationDerivative)(double));
	virtual ~GeneticSurvivalNeuron();

	// Makes this neuron into a copy of the original. Assumes equal structure
	// Only copies the weights for the connections with equal number
	void cloneNeuron(Neuron *original);
    // Does not assume equal iteration order
    void cloneNeuronSlow(Neuron *original);
	// With some probability will change the weights (not replace them!)
	void mutateWeights(
			boost::variate_generator<boost::mt19937&,
					boost::normal_distribution<double> > *gaussian,
			boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
			double mutationChance, double stdDev);
};

class GeneticSurvivalBias: public GeneticSurvivalNeuron {
public:
	GeneticSurvivalBias() :
			GeneticSurvivalNeuron(-1) {
	}
	virtual double output() {
		return 1;
	}
	virtual double output(double *inputs) {
		return 1;
	}
	virtual double outputDeriv() {
		return 0;
	}
};

GeneticSurvivalNetwork* getGeneticSurvivalNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden);

GeneticSurvivalNetwork* getGeneticSurvivalNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden,
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform);

#endif /* GENETICSURVIVALNETWORK_H_ */
