//============================================================================
// Name        : FFNetwork.cpp
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#include "FFNetwork.h"
#include "FFNeuron.h"
#include "activationfunctions.h"
#include <stdexcept>

FFNetwork::FFNetwork(unsigned int numOfInputs, unsigned int numOfHidden) {
	this->numOfInputs = numOfInputs;
	this->numOfHidden = numOfHidden;

	initNodes();
}

/**
 * Create hidden nodes with tanh-function and output node with
 * sigmoid-function.
 */
void FFNetwork::initNodes() {
	this->hiddenNeurons = new Neuron*[this->numOfHidden];
	unsigned int i;
	for (i = 0; i < this->numOfHidden; i++) {
		this->hiddenNeurons[i] = new Neuron(&hyperbole, &hyperboleDeriv);
	}

	this->outputNeuron = new Neuron(&sigmoid, &sigmoidDeriv);

	bias = new Bias;
}

FFNetwork::~FFNetwork() {
	delete this->bias;
	delete this->outputNeuron;

	unsigned int i;
	for (i = 0; i < this->numOfHidden; i++) {
		delete this->hiddenNeurons[i];
	}
	delete[] this->hiddenNeurons;
}

double FFNetwork::output(double *inputs) {
	// Iterate over the neurons in order and calculate their outputs.
	unsigned int i;
	for (i = 0; i < numOfHidden; i++) {
		hiddenNeurons[i]->output(inputs);
	}
	// Finally the output neuron
	return outputNeuron->output(inputs);
}

void FFNetwork::connectOToB(double weight) {
	outputNeuron->connectToNeuron(bias, weight);
}

void FFNetwork::connectOToI(unsigned int inputIndex, double weight) {
	if (inputIndex >= numOfInputs) {
		throw std::invalid_argument(
				"Can not connect to inputIndex which is greater than number of inputs!\n");
	}
	outputNeuron->connectToInput(inputIndex, weight);
}

void FFNetwork::connectOToH(unsigned int hiddenIndex, double weight) {
	if (hiddenIndex >= numOfHidden) {
		throw std::invalid_argument(
				"Can not connect to hiddenIndex which is greater than number of hidden!\n");
	}
	outputNeuron->connectToNeuron(hiddenNeurons[hiddenIndex], weight);
}

void FFNetwork::connectHToB(unsigned int hiddenIndex, double weight) {
	if (hiddenIndex >= numOfHidden) {
		throw std::invalid_argument(
				"Can not connect iddenIndex which is greater than number of hidden!\n");
	}

	hiddenNeurons[hiddenIndex]->connectToNeuron(bias, weight);
}

void FFNetwork::connectHToI(unsigned int hiddenIndex, unsigned int inputIndex,
		double weight) {
	if (hiddenIndex >= numOfHidden) {
		throw std::invalid_argument(
				"Can not connect to hiddenIndex which is greater than number of hidden!\n");
	}
	if (inputIndex >= numOfInputs) {
		throw std::invalid_argument(
				"Can not connect to inputIndex which is greater than number of inputs!\n");
	}
	hiddenNeurons[hiddenIndex]->connectToInput(inputIndex, weight);
}

void FFNetwork::connectHToH(unsigned int firstIndex, unsigned int secondIndex,
		double weight) {
	if (firstIndex >= numOfHidden || secondIndex >= numOfHidden) {
		throw std::invalid_argument(
				"Can not connect hiddenIndex which is greater than number of hidden!\n");
	}
	hiddenNeurons[firstIndex]->connectToNeuron(hiddenNeurons[secondIndex],
			weight);
}
