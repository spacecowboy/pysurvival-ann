/*
 * RPropNetwork.h
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#ifndef RPROPNETWORK_H_
#define RPROPNETWORK_H_

#include "FFNeuron.h"
#include "FFNetwork.h"
#include <vector>

class RPropNeuron: public Neuron {
protected:
	double localError;

	std::vector<double> *prevNeuronUpdates;
	std::vector<double> *prevInputUpdates;
	std::vector<double> *neuronUpdates;
	std::vector<double> *inputUpdates;

	std::vector<double> *prevNeuronDerivs;
	std::vector<double> *prevInputDerivs;

public:
	RPropNeuron();
	RPropNeuron(double (*activationFunction)(double),
			double (*activationDerivative)(double));
	virtual ~RPropNeuron();

	/*
	 * Connect this neuron to the specified neuron and weight
	 */
	virtual void connectToNeuron(Neuron *neuron, double weight);

	/*
	 * connect this neuron to the specified input with weight
	 */
	virtual void connectToInput(unsigned int index, double weight);

	/*
	 * A neuron at a higher layer should call this method on its immediate
	 * connections in previous layers to add error to the target neuron.
	 *
	 * Or called directly with network error derivative if this is an output node.
	 */
	void addLocalError(double error);

	/*
	 * Once the sum is completed by all calls to addLocalError, this method
	 * should be called to calculate what the local derivative is
	 * This also resets the localError to 0 so it's ready for next iteration.
	 *
	 * The result of the local derivative is added to (will not replace) private
	 * variable.
	 *
	 * This method calls addLocalError on connected nodes.
	 */
	void calcLocalDerivative(double *inputs);

	/*
	 * Once you've calculated the derivative sum for the batch in question, call
	 * this to apply the weight update. Will clear private variables afterward except
	 * for previousValues which are given the values calculated this round.
	 */
	void applyWeightUpdates();
};

class RPropBias: public RPropNeuron {
public:
	RPropBias() :
			RPropNeuron() {
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

class RPropNetwork: public FFNetwork {
protected:
	unsigned int maxEpochs;
	double maxError;
	int printEpoch;

public:
	RPropNetwork(unsigned int numOfInputs, unsigned int numOfHidden);
	virtual void initNodes();
	/*
	 * Uses the RProp algorithm to train the network. X is an array of input arrays.
	 * Y is an array of target outputs.
	 */
	void learn(double **X, double *Y, unsigned int length);
	unsigned int getMaxEpochs() const;
	void setMaxEpochs(unsigned int maxEpochs);
	double getMaxError() const;
	void setMaxError(double maxError);
	int getPrintEpoch() const;
	void setPrintEpoch(int printEpoch);
};

/*
 * Returns a FeedForward network that is trained by the RProp algorithm. The network is
 * composed of one layer of hidden units fully connected to inputs. The single output
 * unit is fully connected to the hidden layer. All neurons are connected to the bias.
 */
RPropNetwork* getRPropNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden);

#endif /* RPROPNETWORK_H_ */
