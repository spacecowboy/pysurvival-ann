/*
  A network which trains by use of the cascade correlation algorithm.
  The output nodes are trained by a genetic algorithm.
  The hidden nodes are trained with RPROP to maximize correlation.
*/
#ifndef _COXCASCADENETWORK_H_
#define _COXCASCADENETWORK_H_

#include "FFNeuron.h"
#include "CascadeNetwork.h"
#include "boost/random.hpp"

class GeneticCascadeNeuron : public Neuron {
protected:
  void setup();
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

  // Random stuff
  // Geometric distribution for selecting parents
  //boost::variate_generator<boost::mt19937&,
  //  boost::geometric_distribution<int, double> > *geometric;

  boost::mt19937 rng;

  // Normal distribution for weight mutation, 0 mean and 1 stddev
  // We can then get any normal distribution with y = mean + stddev * x
  boost::normal_distribution<double> dist_normal;
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > gaussian;

  // Uniform distribution, 0 to 1 (inclusive)
  boost::uniform_int<> dist_uniform;
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform;

  /*
    Returns a joint vector of weights. Ordered as:
    Input weights
    Neuron weights
  */
  void getAllWeights(std::vector<double> *weights);
  /*
    Sets the weights in the connections. Expects the same order as in
    getAllWeights
  */
  void setAllWeights(std::vector<double> *weights);

  double outputWithVector(std::vector<double> *weights,
                          double *X);

  double evaluateWithVector(std::vector<double> *weights,
                            double *X, double *Y, unsigned int rows,
                            double *outputs);

public:
  GeneticCascadeNeuron(int id);
  GeneticCascadeNeuron(int id, double (*activationFunction)(double),
			double (*activationDerivative)(double));
  virtual ~GeneticCascadeNeuron();

  virtual void learn(double *X, double *Y,
                     unsigned int length);

};

class GeneticCascadeNetwork : public CascadeNetwork {
 public:
  GeneticCascadeNetwork(unsigned int numOfInputs);
  virtual ~GeneticCascadeNetwork();
  virtual void initNodes();

  virtual void trainOutputs(double *X, double *Y, unsigned int rows);

  virtual void calcErrors(double *X, double *Y, unsigned int rows,
                            double *patError, double *error, double *outputs);
};

class GeneticLadderNetwork : public GeneticCascadeNetwork {
 protected:
  std::vector<GeneticCascadeNeuron*> *hiddenGeneticCascadeNeurons;
 public:
  GeneticLadderNetwork(unsigned int numOfInputs);
  virtual ~GeneticLadderNetwork();
  virtual void initNodes();
  virtual void learn(double *X, double *Y, unsigned int rows);
  virtual void calcErrors(double *X, double *Y, unsigned int rows,
                          double *patError, double *error, double *outputs);

  virtual unsigned int getNumOfHidden() const;
  virtual Neuron* getHiddenNeuron(unsigned int id) const;
  virtual bool getNeuronWeightFromHidden(unsigned int fromId, int toId, double *weight);
  virtual bool getInputWeightFromHidden(unsigned int fromId, unsigned int toIndex, double *weight);

};


#endif /* _COXGENETICCASCADENETWORK_H_ */
