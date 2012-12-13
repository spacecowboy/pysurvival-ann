//============================================================================
// Name        : FFNetwork.cpp
// Author      : jonas
// Date        : 2012-09-09
// Copyright   :
// Description :
//============================================================================

#include <Python.h>
#include "FFNetwork.h"
#include "FFNeuron.h"
#include "activationfunctions.h"
#include <stdexcept>
//#include <iostream>
//#include <fstream>
#include <vector>
using namespace std;

FFNetwork::FFNetwork() {
		this->numOfInputs = 0;
		this->numOfHidden = 0;
		this->numOfOutput = 0;
}

FFNetwork::FFNetwork(unsigned int numOfInputs, unsigned int numOfHidden,
		unsigned int numOfOutput) {
	this->numOfInputs = numOfInputs;
	this->numOfHidden = numOfHidden;
	this->numOfOutput = numOfOutput;

    this->hiddenActivationFunction = TANH;
    this->outputActivationFunction = LOGSIG;
}

FFNetwork::~FFNetwork() {
	delete this->bias;

	unsigned int i;
	for (i = 0; i < this->numOfHidden; i++) {
		delete this->hiddenNeurons[i];
	}
	delete[] this->hiddenNeurons;

	for (i = 0; i < this->numOfOutput; i++) {
		delete this->outputNeurons[i];
	}
	delete[] this->outputNeurons;
}


void FFNetwork::initNodes() {
  hiddenActivationFunction = TANH;
	this->hiddenNeurons = new Neuron*[this->numOfHidden];
	unsigned int i;
	for (i = 0; i < this->numOfHidden; i++) {
      this->hiddenNeurons[i] = new Neuron(i, &hyperbole, &hyperboleDeriv);
	}

    outputActivationFunction = LOGSIG;
	this->outputNeurons = new Neuron*[this->numOfOutput];
	for (i = 0; i < this->numOfOutput; i++) {
      this->outputNeurons[i] = new Neuron(i, &sigmoid, &sigmoidDeriv);
	}

	this->bias = new Bias();
}


double *FFNetwork::output(double *inputs, double *output) {
	//double *output = new double[numOfOutput];
	// Iterate over the neurons in order and calculate their outputs.
	unsigned int i;
	for (i = 0; i < numOfHidden; i++) {
		hiddenNeurons[i]->output(inputs);
	}
	// Finally the output neurons
	for (i = 0; i < numOfOutput; i++) {
		output[i] = outputNeurons[i]->output(inputs);
	}

	return output;
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

unsigned int FFNetwork::getNumOfOutputs() const {
	return numOfOutput;
}

Neuron** FFNetwork::getOutputNeurons() const {
	return outputNeurons;
}

void FFNetwork::connectOToB(unsigned int outputIndex, double weight) {
	if (outputIndex >= numOfOutput) {
		throw invalid_argument(
				"Can not connect to outputIndex which is greater than number of outputs!\n");
	}
	outputNeurons[outputIndex]->connectToNeuron(bias, weight);
}

void FFNetwork::connectOToI(unsigned int outputIndex, unsigned int inputIndex,
		double weight) {
	if (inputIndex >= numOfInputs) {
		throw invalid_argument(
				"Can not connect to inputIndex which is greater than number of inputs!\n");
	}
	if (outputIndex >= numOfOutput) {
		throw invalid_argument(
				"Can not connect to outputIndex which is greater than number of outputs!\n");
	}
	outputNeurons[outputIndex]->connectToInput(inputIndex, weight);
}

void FFNetwork::connectOToH(unsigned int outputIndex, unsigned int hiddenIndex,
		double weight) {
	if (hiddenIndex >= numOfHidden) {
		throw invalid_argument(
				"Can not connect to hiddenIndex which is greater than number of hidden!\n");
	}
	if (outputIndex >= numOfOutput) {
		throw invalid_argument(
				"Can not connect to outputIndex which is greater than number of outputs!\n");
	}
	outputNeurons[outputIndex]->connectToNeuron(hiddenNeurons[hiddenIndex],
			weight);
}

void FFNetwork::connectHToB(unsigned int hiddenIndex, double weight) {
	if (hiddenIndex >= numOfHidden) {
		throw invalid_argument(
				"Can not connect iddenIndex which is greater than number of hidden!\n");
	}

	hiddenNeurons[hiddenIndex]->connectToNeuron(bias, weight);
}

void FFNetwork::connectHToI(unsigned int hiddenIndex, unsigned int inputIndex,
		double weight) {
	if (hiddenIndex >= numOfHidden) {
		throw invalid_argument(
				"Can not connect to hiddenIndex which is greater than number of hidden!\n");
	}
	if (inputIndex >= numOfInputs) {
		throw invalid_argument(
				"Can not connect to inputIndex which is greater than number of inputs!\n");
	}
	hiddenNeurons[hiddenIndex]->connectToInput(inputIndex, weight);
}

void FFNetwork::connectHToH(unsigned int firstIndex, unsigned int secondIndex,
		double weight) {
	if (firstIndex >= numOfHidden || secondIndex >= numOfHidden) {
		throw invalid_argument(
				"Can not connect hiddenIndex which is greater than number of hidden!\n");
	}
	hiddenNeurons[firstIndex]->connectToNeuron(hiddenNeurons[secondIndex],
			weight);
}

bool FFNetwork::getNeuronWeightFromHidden(unsigned int fromId, int toId, double *weight) {
  if (fromId >= numOfHidden || toId > -1 && (unsigned int) toId >= numOfHidden) {
    throw invalid_argument("Id was larger than number of nodes");
  }

  return hiddenNeurons[fromId]->getNeuronWeight(toId, weight);
}

bool FFNetwork::getInputWeightFromHidden(unsigned int fromId, unsigned int toIndex, double *weight) {
  if (fromId >= numOfHidden || toIndex >= numOfInputs) {
    throw invalid_argument("Id was larger than number of nodes or index was greater than number of inputs");
  }

  return hiddenNeurons[fromId]->getInputWeight(toIndex, weight);
}

bool FFNetwork::getNeuronWeightFromOutput(unsigned int fromId, int toId, double *weight) {
  if (fromId >= numOfOutput || toId > -1 && (unsigned int) toId >= numOfHidden) {
    throw invalid_argument("Id was larger than number of nodes");
  }

  return outputNeurons[fromId]->getNeuronWeight(toId, weight);
}

bool FFNetwork::getInputWeightFromOutput(unsigned int fromId, unsigned int toIndex, double *weight) {
  if (fromId >= numOfOutput || toIndex >= numOfInputs) {
    throw invalid_argument("Id was larger than number of nodes or index was greater than number of inputs");
  }

  return outputNeurons[fromId]->getInputWeight(toIndex, weight);
}


void FFNetwork::setOutputActivationFunction(int func) {
  unsigned int i;
  for (i = 0; i < numOfOutput; i++) {
    switch(func) {
    case LINEAR:
      this->outputActivationFunction = LINEAR;
      outputNeurons[i]->setActivationFunction(linear, linearDeriv);
      break;
    case LOGSIG:
      this->outputActivationFunction = LOGSIG;
      outputNeurons[i]->setActivationFunction(sigmoid, sigmoidDeriv);
    break;
    case TANH:
    default:
      outputActivationFunction = TANH;
      outputNeurons[i]->setActivationFunction(hyperbole, hyperboleDeriv);
      break;
    }
  }
}

int FFNetwork::getOutputActivationFunction() {
  return outputActivationFunction;
}

void FFNetwork::setHiddenActivationFunction(int func) {
  unsigned int i;
  for (i = 0; i < numOfHidden; i++) {
    switch(func) {
    case LINEAR:
      hiddenActivationFunction = LINEAR;
      hiddenNeurons[i]->setActivationFunction(linear, linearDeriv);
      break;
    case LOGSIG:
      hiddenActivationFunction = LOGSIG;
      hiddenNeurons[i]->setActivationFunction(sigmoid, sigmoidDeriv);
    break;
    case TANH:
    default:
      hiddenActivationFunction = TANH;
      hiddenNeurons[i]->setActivationFunction(hyperbole, hyperboleDeriv);
      break;
    }
  }
}

int FFNetwork::getHiddenActivationFunction() {
  return hiddenActivationFunction;
}
