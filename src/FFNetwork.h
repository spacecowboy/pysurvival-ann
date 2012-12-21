//============================================================================
// Name        : FFNetwork.h
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#ifndef FFNETWORK_H_
#define FFNETWORK_H_

#include <vector>
//#include "activationfunctions.h"

// Forward-declare
//class Neuron;
//class Bias;

//enum ActivationFuncEnum;
#include "FFNeuron.h"

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
	Neuron **hiddenNeurons;
	Neuron **outputNeurons;
	Neuron *bias;

	unsigned int numOfInputs;
	unsigned int numOfHidden;
	unsigned int numOfOutput;

    int hiddenActivationFunction;
    int outputActivationFunction;

public:
	FFNetwork();
	FFNetwork(unsigned int numOfInputs, unsigned int numOfHidden,
			unsigned int numOfOutput);

	virtual ~FFNetwork();

	/**
	 * Derived classes must implement this and initialize internal neuron lists
	 */
	virtual void initNodes();

    /**
     * Re-creates all neurons to reset connections
     */
    virtual void resetNodes();
    /**
     * Deletes all neurons from memory.
     */
    virtual void deleteNeurons();

	/**
	 * Returns the pointer given as output, so pay no attention to return object
	 * if not wanted.
	 */
	virtual double *output(double *inputs, double *output);

    /**
     * Sets the activation function of the output layer
     */
    void setOutputActivationFunction(int func);
    int getOutputActivationFunction();

    /**
     * Sets the activation function of the hidden layer
     */
    void setHiddenActivationFunction(int func);
    int getHiddenActivationFunction();

	/*
	 * Connects the first argument to the second argument. Meaning that the following
	 * will be true: First.output = w * Second.output
	 */
	void connectHToH(unsigned int firstIndex, unsigned int secondIndex,
			double weight);
	/*
	 * Connect the hidden neuron with id (first argument) to the specified input index (second
	 * argument)
	 */
	void connectHToI(unsigned int hiddenIndex, unsigned int inputIndex,
			double weight);

	/*
	 * Connect the hidden neuront o the bias
	 */
	void connectHToB(unsigned int hiddenIndex, double weight);

	/*
	 * Connects the output neuron to the specified hidden neuron.
	 */
	void connectOToH(unsigned int outputIndex, unsigned int hiddenIndex,
			double weight);

	/*
	 * Connect the output neuron to the specified input index.
	 */
	void connectOToI(unsigned int outputIndex, unsigned int inputIndex,
			double weight);

	/*
	 * Connect the output neuron to the bias
	 */
	void connectOToB(unsigned int outputIndex, double weight);

    virtual Neuron* getHiddenNeuron(unsigned int id) const;
	virtual unsigned int getNumOfHidden() const;
	virtual unsigned int getNumOfInputs() const;
	virtual unsigned int getNumOfOutputs() const;
	virtual Neuron* getOutputNeuron(unsigned int id) const;

	Neuron* getBias() const {
		return bias;
	}

    /*
     * Return the weight that toId is connected to fromId with
     */
    virtual bool getNeuronWeightFromHidden(unsigned int fromId, int toId, double *weight);
    virtual bool getInputWeightFromHidden(unsigned int fromId, unsigned int toIndex, double *weight);

    bool getNeuronWeightFromOutput(unsigned int fromId, int toId, double *weight);
    bool getInputWeightFromOutput(unsigned int fromId, unsigned int toIndex, double *weight);

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
