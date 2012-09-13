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

Neuron** FFNetwork::getHiddenNeurons() const {
	return hiddenNeurons;
}

unsigned int FFNetwork::getNumOfHidden() const {
	return numOfHidden;
}

unsigned int FFNetwork::getNumOfInputs() const {
	return numOfInputs;
}

Neuron* FFNetwork::getOutputNeuron() const {
	return outputNeuron;
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
