//============================================================================
// Name        : FFNeuron.h
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#ifndef FFNEURON_H_
#define FFNEURON_H_

#include <vector>

/*
 * The neuron is not intended to be public. Users are only intended to use Network.
 */
class Neuron {
protected:
	double cachedOutput;
	double cachedInputSum;
	// Function pointers
	double (*activationFunction)(double);
	double (*activationDerivative)(double);
    int neuronId;

public:
	// Connections is a vector of neuron-weight pairs
	std::vector<std::pair<Neuron*, double> > *neuronConnections;
	// If connected to the input values, index-weight pairs
	std::vector<std::pair<unsigned int, double> > *inputConnections;

	Neuron(int id);
	Neuron(int id, double (*activationFunction)(double),
			double (*activationDerivative)(double));
	virtual ~Neuron();

	/*
	 * Connect this neuron to the specified neuron and weight
	 */
	virtual void connectToNeuron(Neuron *neuron, double weight);

	/*
	 * connect this neuron to the specified input with weight
	 */
	virtual void connectToInput(unsigned int index, double weight);

	/*
	 * Returns the value which was calculated by the previous call to output(*inputs)
	 */
	virtual double output();
	/*
	 * Calculate the output of this neuron.
	 * Assumes connected nodes have already computed their outputs.
	 */
	virtual double output(const double * const inputs);
	/*
	 * Returns the derivative of the output which was calculated by the previous call to output(*inputs)
	 */
	virtual double outputDeriv();

	void setActivationFunction(double (*activationFunction)(double),
			double (*activationDerivative)(double));

    /*
     * Return the id of this neuron. Is only set during construction.
     * Ideally it equals its place in the iterator.
     *
     * -1 is reserved for bias nodes.
     */
    virtual int getId();
    virtual void setId(int id);

    /*
     * Returns true if a connection exists, false otherwise
     */
    bool getNeuronWeight(int targetId, double *weight);
    bool getInputWeight(unsigned int inputIndex, double *weight);

    /*
     * Number of connections from this neuron.
     * Sum of neuron connections and input connections
     */
    unsigned int getNumOfConnections();

    /*
     * Sum of weights
     */
    double getWeightsSquaredSum();
    double getWeightsAbsoluteSum();
    /*
     * Soft weight elimination
     * P = sum( w^2 / [lambda^2 + w^2])
     * Returns zero if lambda is zero.
     */
    double getWeightEliminationSum(double lambda);
};

/*
 * Special neuron which always outputs 1. Has no derivative but returns 0 for compatibility.
 */
class Bias: public Neuron {
public:
	Bias() :
			Neuron(-1) {
	}

	virtual double output() {
		return 1;
	}
	virtual double output(const double * const inputs) {
		return 1;
	}
	virtual double outputDeriv() {
		return 0;
	}
};

#endif /* FFNEURON_H_ */
