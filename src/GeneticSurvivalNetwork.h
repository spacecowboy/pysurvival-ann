/*
 * GeneticNetwork.h
 *
 *  Created on: 11 sep 2012
 *      Author: jonas
 */

#ifndef GENETICSURVIVALNETWORK_H_
#define GENETICSURVIVALNETWORK_H_

#include "FFNetwork.h"
#include "FFNeuron.h"

class GeneticSurvivalNetwork: public FFNetwork {
private:
	// Training variables
	// How many networks should be created/mutated and compared in one generation
	unsigned int populationSize;
	// Train for this many generations. E.g. total number of networks created/mutated is
	// populationSize * generations
	unsigned int generations;
	// If uniformRandom < weightMutationChance then mutate weight by mutationChange
	double weightMutationChance;
	// This is the mean in: mutationChange = random.exponentialDistribution(mean)
	// This is the maximum the mean can be. If weightMutationHalfPoint is set, it can be less
	// In C++ they define lambda = 1 / mean. This variable IS the mean. This is NOT lambda.
	double weightMutationMean;

	// If this is non zero, it is interpreted as the generation where the mean should have
	// decreased to half its value. The mean is calculated according to the logsig func:
	// weightMutationMean / (1 + exp(generation - weightMutationHalfPoint))
	// This calculation is NOT done if this is zero, which it is by default.
	unsigned int weightMutationHalfPoint;

public:
	GeneticSurvivalNetwork(unsigned int numOfInputs, unsigned int numOfHidden);
	virtual void initNodes();
	/*
	 * Expects the X and Y to be of equal number of rows. Y has 2 columns,
	 * first being survival time, second being event (1 or 0)
	 */
	void learn(double **X, double **Y, unsigned int length);
};

class GeneticSurvivalNeuron : public Neuron {
public:
	GeneticSurvivalNeuron();
	GeneticSurvivalNeuron(double (*activationFunction)(double),
			double (*activationDerivative)(double));
	virtual ~GeneticSurvivalNeuron();
};

class GeneticSurvivalBias: public GeneticSurvivalNeuron {
public:
	GeneticSurvivalBias() :
		GeneticSurvivalNeuron() {
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

#endif /* GENETICSURVIVALNETWORK_H_ */
