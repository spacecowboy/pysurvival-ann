//============================================================================
// Name        : FFNeuron.cpp
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#include "FFNeuron.h"
#include "activationfunctions.h"
#include <math.h> /* pow, fabs */

// -------------------
// Neuron definitions
// -------------------
Neuron::Neuron(int id) {
  neuronId = id;
	cachedOutput = 0;
	neuronConnections = new std::vector<std::pair<Neuron*, double>>;
	inputConnections = new std::vector<std::pair<unsigned int, double>>;
	setActivationFunction(&sigmoid, &sigmoidDeriv);
}

Neuron::Neuron(int id, double (*activationFunction)(double),
		double (*activationDerivative)(double)) {
  neuronId = id;
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

int Neuron::getId() {
  return neuronId;
}

void Neuron::setId(int id) {
  if (id >= 0)
    neuronId = id;
}

/*
 * Returns true if a connection exists, false otherwise
 */
bool Neuron::getNeuronWeight(int targetId, double *weight) {
  bool retval = false;
  unsigned int i;
  Neuron *neuron;
  for (i = 0; i < neuronConnections->size(); i++) {
		neuron = neuronConnections->at(i).first;
        if (neuron->getId() == targetId) {
          retval = true;
          *weight = neuronConnections->at(i).second;
          break;
        }
	}
  return retval;
}

bool Neuron::getInputWeight(unsigned int targetIndex, double *weight) {
  bool retval = false;
  unsigned int i, inputIndex;
  for (i = 0; i < inputConnections->size(); i++) {
		inputIndex = inputConnections->at(i).first;
        if (targetIndex == inputIndex) {
          retval = true;
          *weight = inputConnections->at(i).second;
          break;
        }
	}
  return retval;
}

unsigned int Neuron::getNumOfConnections() {
	 return neuronConnections->size() + inputConnections->size();
}

double Neuron::getWeightsSquaredSum() {
	 double sum = 0;
	 unsigned int i;
	 for (i = 0; i < neuronConnections->size(); i++) {
		  sum += pow(neuronConnections->at(i).second, 2);
	 }
	 for (i = 0; i < inputConnections->size(); i++) {
		  sum += pow(inputConnections->at(i).second, 2);
	 }

	 return sum;
}

double Neuron::getWeightsAbsoluteSum() {
	 double sum = 0;
	 unsigned int i;
	 for (i = 0; i < neuronConnections->size(); i++) {
		  sum += fabs(neuronConnections->at(i).second);
	 }
	 for (i = 0; i < inputConnections->size(); i++) {
		  sum += fabs(inputConnections->at(i).second);
	 }

	 return sum;
}

double Neuron::getWeightEliminationSum(double lambda) {
     double sum = 0, w2 = 0;
     unsigned int i;
     if (lambda == 0) {
          return 1.0;
     }
	 else {
		  double l2 = pow(lambda, 2);
		  for (i = 0; i < neuronConnections->size(); i++) {
			   w2 = pow(neuronConnections->at(i).second, 2);
			   sum += w2 / (l2 + w2);
		  }
		  for (i = 0; i < inputConnections->size(); i++) {
			   w2 = pow(inputConnections->at(i).second, 2);
			   sum += w2 / (l2 + w2);
		  }
	 }

	 return sum;
}
