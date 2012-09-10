//============================================================================
// Name        : FFNeuron.cpp
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#include <stdio.h>

#include "FFNeuron.h"

// ------------------
// Network definitions
//

// -------------------
// Neuron definitions
// -------------------
Neuron::Neuron() {
	init();
	setActivationFunction(&sigmoid, &sigmoidDeriv);
}

Neuron::Neuron(double (*activationFunction)(double),
		double (*activationDerivative)(double)) {
	init();
	setActivationFunction(activationFunction, activationDerivative);
}

void Neuron::init() {
	cachedOutput = 0;
	neuronConnections = new std::vector<std::pair<Neuron*, double>>;
	inputConnections = new std::vector<std::pair<int, double>>;
}

Neuron::~Neuron() {
	delete neuronConnections;
	delete inputConnections;
}

void Neuron::connectTo(int index, double weight) {
	inputConnections->push_back(std::pair<int, double>(index, weight));
}

void Neuron::connectTo(Neuron *neuron, double weight) {
	neuronConnections->push_back(std::pair<Neuron*, double>(neuron, weight));
}

double Neuron::output() {
	return cachedOutput;
}

double Neuron::output(double *inputs) {
	// First calculate the input sum
	cachedInputSum = 0;

	// Iterate over input connections first
	std::vector<std::pair<int, double>>::iterator ic;

	for (ic = inputConnections->begin(); ic < inputConnections->end(); ic++) {
		cachedInputSum += inputs[ic->first] * ic->second;
		printf("input iterator sum = %f\n", cachedInputSum);
	}

	// Then take neuron connections
	std::vector<std::pair<Neuron*, double>>::iterator nc;

	for (nc = neuronConnections->begin(); nc < neuronConnections->end(); nc++) {
		cachedInputSum += nc->first->output(inputs) * nc->second;
		printf("neuron iterator sum = %f\n", cachedInputSum);
	}

	cachedOutput = activationFunction(cachedInputSum);
	printf("Output is %f\n", cachedOutput);
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

int main(int argc, char* argv[]) {

	//Bias b;
	//printf("Bias output is %f\n", b.output());
	double x[2] = { -1.0, 1.0 };
	Neuron *n = new Neuron;
	n->connectTo(0, 0.5);
	n->connectTo(1, 0.7);

	Neuron *o = new Neuron;
	o->connectTo(n, -1);
	printf("Neuron output is %f\n", o->output(x));
	printf("Neuron outputDeriv is %f\n", o->outputDeriv());

	delete o;
	delete n;
}
