//============================================================================
// Name        : FFNeuron.cpp
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#include "FFNeuron.h"
#include "activationfunctions.h"

// -------------------
// Neuron definitions
// -------------------
Neuron::Neuron() {
	cachedOutput = 0;
	neuronConnections = new std::vector<std::pair<Neuron*, double>>;
	inputConnections = new std::vector<std::pair<unsigned int, double>>;
	setActivationFunction(&sigmoid, &sigmoidDeriv);
}

Neuron::Neuron(double (*activationFunction)(double),
		double (*activationDerivative)(double)) {
	cachedOutput = 0;
	neuronConnections = new std::vector<std::pair<Neuron*, double>>;
	inputConnections = new std::vector<std::pair<unsigned int, double>>;
	setActivationFunction(activationFunction, activationDerivative);
}

Neuron::~Neuron() {
	delete neuronConnections;
	delete inputConnections;
}

void Neuron::connectToInput(unsigned int index, double weight) {
	inputConnections->push_back(std::pair<unsigned int, double>(index, weight));
}

void Neuron::connectToNeuron(Neuron *neuron, double weight) {
	neuronConnections->push_back(std::pair<Neuron*, double>(neuron, weight));
}

double Neuron::output() {
	return cachedOutput;
}

double Neuron::output(double *inputs) {
	// First calculate the input sum
	cachedInputSum = 0;

	// Iterate over input connections first
	std::vector<std::pair<unsigned int, double>>::iterator ic;

	for (ic = inputConnections->begin(); ic < inputConnections->end(); ic++) {
		cachedInputSum += inputs[ic->first] * ic->second;
	}

	// Then take neuron connections
	std::vector<std::pair<Neuron*, double>>::iterator nc;

	for (nc = neuronConnections->begin(); nc < neuronConnections->end(); nc++) {
		cachedInputSum += nc->first->output() * nc->second;
	}

	cachedOutput = activationFunction(cachedInputSum);
	return cachedOutput;
}

double Neuron::outputDeriv() {
	return activationDerivative(cachedOutput);
}

void Neuron::setActivationFunction(double (*activationFunction)(double),
		double (*activationDerivative)(double)) {
	this->activationFunction = activationFunction;
	this->activationDerivative = activationDerivative;
}
