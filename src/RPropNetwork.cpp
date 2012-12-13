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
	RPropNetwork *net = new RPropNetwork(numOfInputs, numOfHidden, 1);
	net->initNodes();
	unsigned int i, j;
	// Connect hidden to input and bias
	// Connect output to hidden
	for (i = 0; i < numOfHidden; i++) {
		net->connectHToB(i, dRand());
		net->connectOToH(0, i, dRand());
		// Inputs
		for (j = 0; j < numOfInputs; j++) {
			net->connectHToI(i, j, dRand());
		}
	}
	// Connect output to bias
	net->connectOToB(0, dRand());

	return net;
}

/*
 * Derivative of SumSquareError
 */
double SSEDeriv(double target, double output) {
	return target - output;
}

double *SSEDerivs(double *target, double *output, int length) {
	double *derivs = new double[length];
	for (int i = 0; i < length; i++) {
		derivs[i] = target[i] - output[i];
	}
	return derivs;
}

double SSE(double target, double output) {
	return std::pow(target - output, 2) / 2;
}

double *SSEs(double *target, double *output, int length) {
	double *errors = new double[length];
	for (int i = 0; i < length; i++) {
		errors[i] = std::pow(target[i] - output[i], 2) / 2;
	}
	return errors;
}

RPropNetwork::RPropNetwork(unsigned int numOfInputs, unsigned int numOfHidden,
		unsigned int numOfOutput) :
		FFNetwork(numOfInputs, numOfHidden, numOfOutput) {
	maxEpochs = 10000;
	maxError = 0.0000001;
	printEpoch = 100;
}

void RPropNetwork::initNodes() {
	this->hiddenNeurons = new Neuron*[this->numOfHidden];
	unsigned int i;
	for (i = 0; i < this->numOfHidden; i++) {
      this->hiddenNeurons[i] = new RPropNeuron(i, &hyperbole, &hyperboleDeriv);
	}

	this->outputNeurons = new Neuron*[this->numOfOutput];
	for (i = 0; i < this->numOfOutput; i++) {
      this->outputNeurons[i] = new RPropNeuron(i, &sigmoid, &sigmoidDeriv);
	}

	this->bias = new RPropBias;
}

unsigned int RPropNetwork::getMaxEpochs() const {
	return maxEpochs;
}

void RPropNetwork::setMaxEpochs(unsigned int maxEpochs) {
	this->maxEpochs = maxEpochs;
}

double RPropNetwork::getMaxError() const {
	return maxError;
}

int RPropNetwork::getPrintEpoch() const {
	return printEpoch;
}

void RPropNetwork::setPrintEpoch(int printEpoch) {
	this->printEpoch = printEpoch;
}

void RPropNetwork::setMaxError(double maxError) {
	this->maxError = maxError;
}

void RPropNetwork::learn(double *X, double *Y, unsigned int rows) {
	double error[numOfOutput];
	for (unsigned int i = 0; i < numOfOutput; i++)
		error[i] = 0;

	double *outputs = new double[numOfOutput];
	double deriv = 0;
	unsigned int epoch = 0;

	int i, n;

	do {
      if (printEpoch > 0 && epoch % (int) printEpoch == 0)
			printf("epoch: %d, error: %f\n", epoch, error[0]);
		// Evaluate for each value in input vector
      for (i = 0; i < (int) rows; i++) {
			// First let all neurons evaluate
			output(X + i*numOfInputs, outputs);
			for (n = 0; n < (int) numOfOutput; n++) {
				deriv = SSEDeriv(Y[i*numOfOutput + n], outputs[n]);
				error[n] = SSE(Y[i*numOfOutput + n], outputs[n]);
				// set error deriv on output node
				static_cast<RPropNeuron*>(outputNeurons[n])->addLocalError(
						deriv);
				// Calculate local derivatives at output and propagate
				static_cast<RPropNeuron*>(outputNeurons[n])->calcLocalDerivative(
						X + i*numOfInputs);
			}

			// Calculate local derivatives at all neurons
			// and propagate
			for (n = numOfHidden - 1; n >= 0; n--) {
				static_cast<RPropNeuron*>(hiddenNeurons[n])->calcLocalDerivative(
						X + i*numOfInputs);
			}
		}
		// Apply weight updates
		for (n = 0; n < (int) numOfOutput; n++) {
			static_cast<RPropNeuron*>(outputNeurons[n])->applyWeightUpdates();
		}
		for (n = numOfHidden - 1; n >= 0; n--) {
			static_cast<RPropNeuron*>(hiddenNeurons[n])->applyWeightUpdates();
		}
		epoch += 1;
	} while (epoch < maxEpochs && error[0] > maxError);

	delete[] outputs;
}

/*
 * -----------------------
 * RPropNeuron definitions
 * -----------------------
 */

RPropNeuron::RPropNeuron(int id) :
		Neuron(id) {
	localError = 0;

	prevNeuronUpdates = new std::vector<double>;
	prevInputUpdates = new std::vector<double>;
	prevNeuronDerivs = new std::vector<double>;
	prevInputDerivs = new std::vector<double>;

	neuronUpdates = new std::vector<double>;
	inputUpdates = new std::vector<double>;
}

RPropNeuron::RPropNeuron(int id, double (*activationFunction)(double),
		double (*activationDerivative)(double)) :
  Neuron(id, activationFunction, activationDerivative) {
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
