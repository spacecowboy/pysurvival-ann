//============================================================================
// Name        : FFNetwork.h
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#ifndef FFNETWORK_H_
#define FFNETWORK_H_

// Forward-declare the Neuron class
class Neuron;
class Bias;
//#include "FFNeuron.h"

/*
 * The network is the public base class users are intended to work with.
 * Has one output node. Constructor takes number of input and hidden nodes.
 * The constructor will create all the neurons but not connect them. The user
 * or a factory function is supposed to connect them as required.
 *
 * Note that this is an abstract class. Any inheriting classes must implement the
 * learn method. A network is defined first by its structure but also by its
 * training algorithm.
 */
class FFNetwork {
private:
	unsigned int numOfInputs;
	unsigned int numOfHidden;
	Neuron **hiddenNeurons;
	Neuron *outputNeuron;
	Bias *bias;

	virtual void initNodes();
public:
	FFNetwork(unsigned int numOfInputs, unsigned int numOfHidden);
	virtual ~FFNetwork();

	double output(double *inputs);

	//virtual void learn(double **data) = 0;
	/*
	 * Connects the first argument to the second argument. Meaning that the following
	 * will be true: First.output = w * Second.output
	 */
	void connectHToH(unsigned int firstIndex, unsigned int secondIndex, double weight);
	/*
	 * Connect the hidden neuron (first argument) to the specified input index (second
	 * argument)
	 */
	void connectHToI(unsigned int hiddenIndex, unsigned int inputIndex, double weight);

	/*
	 * Connect the hidden neuront o the bias
	 */
	void connectHToB(unsigned int hiddenIndex, double weight);

	/*
	 * Connects the output neuron to the specified hidden neuron.
	 */
	void connectOToH(unsigned int hiddenIndex, double weight);

	/*
	 * Connect the output neuron to the specified input index.
	 */
	void connectOToI(unsigned int inputIndex, double weight);

	/*
	 * Connect the output neuron to the bias
	 */
	void connectOToB(double weight);
};

#endif /* FFNETWORK_H_ */
