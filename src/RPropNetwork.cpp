/*
 * RPropNetwork.cpp
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#include "RPropNetwork.h"
#include "FFNeuron.h"
#include "activationfunctions.h"
#include "drand.h"
#include <cmath>
#include <stdio.h>

signed int sign(double x) {
	if (x >= 0)
		return 1;
	else
		return -1;
}

/*
 * Returns a single layer FeedForward Network that is trained by the RProp algorithm.
 * It is up to the user to free this network from the heap once done.
 */
RPropNetwork* getRPropNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden) {
	RPropNetwork *net = new RPropNetwork(numOfInputs, numOfHidden);

	// Set random seed
	setSeed();

	unsigned int i, j;
	// Connect hidden to input and bias
	// Connect output to hidden
	for (i = 0; i < numOfHidden; i++) {
		net->connectHToB(i, dRand());
		net->connectOToH(i, dRand());
		// Inputs
		for (j = 0; j < numOfInputs; j++) {
			net->connectHToI(i, j, dRand());
		}
	}
	// Connect output to bias
	net->connectOToB(dRand());

	return net;
}

/*
 * Derivative of SumSquareError
 */
double SSEDeriv(double target, double output) {
	return target - output;
}

double SSE(double target, double output) {
	return std::pow(target - output, 2) / 2;
}

RPropNetwork::RPropNetwork(unsigned int numOfInputs, unsigned int numOfHidden) :
		FFNetwork(numOfInputs, numOfHidden) {
	maxEpochs = 1000;
	maxError = 0.0000001;
	printf("rprop network constr\n");
	initNodes();
}

void RPropNetwork::initNodes() {
	printf("rprop initnodes\n");
	this->hiddenNeurons = new Neuron*[this->numOfHidden];
	unsigned int i;
	for (i = 0; i < this->numOfHidden; i++) {
		this->hiddenNeurons[i] = new RPropNeuron(&hyperbole, &hyperboleDeriv);
	}

	this->outputNeuron = new RPropNeuron(&sigmoid, &sigmoidDeriv);

	this->bias = new RPropBias;
}

void RPropNetwork::learn(double **X, double *Y, unsigned int length) {
	double error = 0, errorDeriv = 0;
	unsigned int epoch = 0;

	int i, n;

	do {
		printf("epoch: %d, error: %f\n", epoch, error);
		// Evaluate for each value in input vector
		for (i = 0; i < (int) length; i++) {
			// First let all neurons evaluate
			errorDeriv = SSEDeriv(Y[i], output(X[i]));
			error = SSE(Y[i], outputNeuron->output());
			// set error deriv on output node
			static_cast<RPropNeuron*>(outputNeuron)->addLocalError(errorDeriv);
			// Calculate local derivatives at all neurons
			// and propagate
			static_cast<RPropNeuron*>(outputNeuron)->calcLocalDerivative(X[i]);
			for (n = numOfHidden - 1; n >= 0; n--) {
				static_cast<RPropNeuron*>(hiddenNeurons[n])->calcLocalDerivative(
						X[i]);
			}
		}
		// Apply weight updates
		static_cast<RPropNeuron*>(outputNeuron)->applyWeightUpdates();
		for (n = numOfHidden - 1; n >= 0; n--) {
			static_cast<RPropNeuron*>(hiddenNeurons[n])->applyWeightUpdates();
		}
		epoch += 1;
	} while (epoch < maxEpochs && error > maxError);
}

/*
 * -----------------------
 * RPropNeuron definitions
 * -----------------------
 */

RPropNeuron::RPropNeuron() :
		Neuron() {
	printf("rprop init\n");

	localError = 0;

	prevNeuronUpdates = new std::vector<double>;
	prevInputUpdates = new std::vector<double>;
	prevNeuronDerivs = new std::vector<double>;
	prevInputDerivs = new std::vector<double>;

	neuronUpdates = new std::vector<double>;
	inputUpdates = new std::vector<double>;
}

RPropNeuron::RPropNeuron(double (*activationFunction)(double),
		double (*activationDerivative)(double)) :
		Neuron(activationFunction, activationDerivative) {
	printf("rprop init\n");

	localError = 0;

	prevNeuronUpdates = new std::vector<double>;
	prevInputUpdates = new std::vector<double>;
	prevNeuronDerivs = new std::vector<double>;
	prevInputDerivs = new std::vector<double>;

	neuronUpdates = new std::vector<double>;
	inputUpdates = new std::vector<double>;
}

RPropNeuron::~RPropNeuron() {
	delete prevNeuronUpdates;
	delete prevNeuronDerivs;
	delete prevInputDerivs;
	delete prevInputUpdates;
	delete neuronUpdates;
	delete inputUpdates;
}

void RPropNeuron::connectToInput(unsigned int index, double weight) {
	this->Neuron::connectToInput(index, weight);

	prevInputUpdates->push_back(0.1);
	prevInputDerivs->push_back(1);
	inputUpdates->push_back(0);
}

void RPropNeuron::connectToNeuron(Neuron *neuron, double weight) {
	this->Neuron::connectToNeuron(neuron, weight);

	prevNeuronUpdates->push_back(0.1);
	prevNeuronDerivs->push_back(1);
	neuronUpdates->push_back(0);
}

void RPropNeuron::addLocalError(double error) {
	localError += error;
}

void RPropNeuron::calcLocalDerivative(double *inputs) {
	unsigned int inputIndex, i;
	localError *= outputDeriv();

	//Propagate the error backwards
	//And calculate weight updates
	for (i = 0; i < neuronConnections->size(); i++) {
		// Propagate backwards
		// I know it's an RPropNeuron
		static_cast<RPropNeuron*>(neuronConnections->at(i).first)->addLocalError(
				localError * neuronConnections->at(i).second);

		// Calculate and add weight update
		(neuronUpdates->begin())[i] += (localError
				* (neuronConnections->at(i).first->output()));
	}
	// Calculate weight updates to inputs also
	for (i = 0; i < inputConnections->size(); i++) {
		inputIndex = inputConnections->at(i).first;
		(inputUpdates->begin()[i]) += (localError * inputs[inputIndex]);
	}

	// Set to zero for next iteration
	localError = 0;
}

void RPropNeuron::applyWeightUpdates() {
	unsigned int i;
	double prevUpdate, prevDeriv, weightUpdate, deriv;
	// Maximum and minimm weight changes
	double dMax = 50, dMin = 0.00001;
	// How much to adjust updates
	double dPos = 1.2, dNeg = 0.5;

	for (i = 0; i < neuronConnections->size(); i++) {
		prevUpdate = prevNeuronUpdates->at(i);
		prevDeriv = prevNeuronDerivs->at(i);
		deriv = neuronUpdates->at(i);

		if (prevDeriv * deriv > 0) {
			// We are on the right track, increase speed!
			weightUpdate = std::abs(prevUpdate) * dPos;
			// But not too fast!
			if (weightUpdate > dMax)
				weightUpdate = dMax;
			weightUpdate *= sign(deriv);
			neuronConnections->at(i).second += weightUpdate;
		} else if (prevDeriv * deriv < 0) {
			// Shit, we overshot the target!
			weightUpdate = std::abs(prevUpdate) * dNeg;
			if (weightUpdate < dMin)
				weightUpdate = dMin;
			weightUpdate *= sign(deriv);
			// Go back
			neuronConnections->at(i).second -= prevUpdate;
			// Next time forget about this disastrous direction
			deriv = 0;
		} else {
			// Previous round we overshot, go forward
			weightUpdate = std::abs(prevUpdate) * sign(deriv);
			neuronConnections->at(i).second += weightUpdate;
		}

		prevNeuronDerivs->begin()[i] = deriv;
		prevNeuronUpdates->begin()[i] = weightUpdate;
		neuronUpdates->begin()[i] = 0;
	}
	// Calculate weight updates to inputs also
	for (i = 0; i < inputConnections->size(); i++) {
		prevUpdate = prevInputUpdates->at(i);
		prevDeriv = prevInputDerivs->at(i);
		deriv = inputUpdates->at(i);

		if (prevDeriv * deriv > 0) {
			// We are on the right track, increase speed!
			weightUpdate = std::abs(prevUpdate) * dPos;
			// But not too fast!
			if (weightUpdate > dMax)
				weightUpdate = dMax;
			weightUpdate *= sign(deriv);
			inputConnections->at(i).second += weightUpdate;
		} else if (prevDeriv * deriv < 0) {
			// Shit, we overshot the target!
			weightUpdate = std::abs(prevUpdate) * dNeg;
			if (weightUpdate < dMin)
				weightUpdate = dMin;
			weightUpdate *= sign(deriv);
			// Go back
			inputConnections->at(i).second -= prevUpdate;
			// Next time forget about this disastrous direction
			deriv = 0;
		} else {
			// Previous round we overshot, go forward
			weightUpdate = std::abs(prevUpdate) * sign(deriv);
			inputConnections->at(i).second += weightUpdate;
		}

		prevInputDerivs->begin()[i] = deriv;
		prevInputUpdates->begin()[i] = weightUpdate;
		inputUpdates->begin()[i] = 0;
	}
}
