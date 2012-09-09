/*
 * ANN.h
 *
 *  Created on: 7 sep 2012
 *      Author: jonas
 */

#ifndef ANN_H_
#define ANN_H_

#include <vector>
#include "activationfunctions.h"

/*
 * The network is the public class users are intended to work with.
 * Default implementation has one output node.
 */
class FFNetwork {
private:
	int numOfInputs;
public:
	double output(double *inputs);
	void learn(double **data);
};

/*
 * The neuron is not intended to be public. Users are only intended to use Network.
 */
class Neuron {
private:
	// Connections is a vector of neuron-weight pairs
	std::vector<std::pair<Neuron*, double> > *neuronConnections;
	// If connected to the input values, index-weight pairs
	std::vector<std::pair<int, double> > *inputConnections;

	double cachedOutput;
	// Function pointers
	double (*activationFunction)(double);
	double (*activationDerivative)(double);

	/*
	 * Calculate what value to pass into the activation function
	 */
	double inputSum(double *inputs);

public:
	Neuron();
	virtual ~Neuron();

	/*
	 * Connect this neuron to the specified neuron and weight
	 */
	void connectTo(Neuron *neuron, double weight);

	/*
	 * connect this neuron to the specified input with weight
	 */
	void connectTo(int index, double weight);

	/*
	 * Returns the value which was calculated by the previous call to output(*inputs)
	 */
	virtual double output();
	/*
	 * Traverse the network as required and calculate the output of this neuron
	 */
	virtual double output(double *inputs);
	/*
	 * Returns the derivative of the output which was calculated by the previous call to output(*inputs)
	 */
	virtual double outputDeriv();

	void setActivationFunction(double (*activationFunction)(double),
			double (*activationDerivative)(double)) {
		this->activationFunction = activationFunction;
		this->activationDerivative = activationDerivative;
	}

};

/*
 * Special neuron which always outputs 1. Has no derivative but returns 0 for compatibility.
 */
class Bias: Neuron {
public:
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

#endif /* ANN_H_ */
