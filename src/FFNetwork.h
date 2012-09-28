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
protected:
	unsigned int numOfInputs;
	unsigned int numOfHidden;
	unsigned int numOfOutput;
	Neuron **hiddenNeurons;
	Neuron **outputNeurons;
	Neuron *bias;

public:
	FFNetwork(unsigned int numOfInputs, unsigned int numOfHidden, unsigned int numOfOutput);
	virtual ~FFNetwork();

	/**
	 * Derived classes must implement this and initialize internal neuron lists
	 */
	virtual void initNodes();

	/**
	 * Returns the pointer given as output, so pay no attention to return object
	 * if not wanted.
	 */
	double *output(double *inputs, double *output);

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
	void connectOToH(unsigned int outputIndex, unsigned int hiddenIndex, double weight);

	/*
	 * Connect the output neuron to the specified input index.
	 */
	void connectOToI(unsigned int outputIndex, unsigned int inputIndex, double weight);

	/*
	 * Connect the output neuron to the bias
	 */
	void connectOToB(unsigned int outputIndex, double weight);
	Neuron** getHiddenNeurons() const;
	unsigned int getNumOfHidden() const;
	unsigned int getNumOfInputs() const;
	unsigned int getNumOfOutputs() const;
	Neuron** getOutputNeurons() const;

	Neuron* getBias() const {
		return bias;
	}
};

/*
 * Utility functions
 */

/*
 * Save network to file in semi-human-readable format
 */
//void saveFFNetwork(FFNetwork *ann, std::string filename);

/*
 * Load an exact copy of a network that was saved by saveFFNetwork
 */
//FFNetwork* loadFFNetwork(std::string filename);

#endif /* FFNETWORK_H_ */
