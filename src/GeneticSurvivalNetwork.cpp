/*
 * GeneticNetwork.cpp
 *
 *  Created on: 11 sep 2012
 *      Author: jonas
 */

#include "GeneticSurvivalNetwork.h"
#include "FFNeuron.h"
#include "FFNetwork.h"
#include "activationfunctions.h"
#include "drand.h"
#include <vector>
#include <stdio.h>

using namespace std;

GeneticSurvivalNetwork* getGeneticSurvivalNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden) {
	GeneticSurvivalNetwork *net = new GeneticSurvivalNetwork(numOfInputs,
			numOfHidden);

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

	// Set output node to linear activation function
	net->getOutputNeuron()->setActivationFunction(&linear, &linearDeriv);

	return net;
}

GeneticSurvivalNetwork::GeneticSurvivalNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden) :
		FFNetwork(numOfInputs, numOfHidden) {
	populationSize = 5;
	generations = 10;
	weightMutationChance = 0.15;
	weightMutationMean = 0.1;
	weightMutationHalfPoint = 0;

	initNodes();
}

void GeneticSurvivalNetwork::initNodes() {
	this->hiddenNeurons = new Neuron*[this->numOfHidden];
	unsigned int i;
	for (i = 0; i < this->numOfHidden; i++) {
		this->hiddenNeurons[i] = new GeneticSurvivalNeuron(&hyperbole,
				&hyperboleDeriv);
	}
	this->outputNeuron = new GeneticSurvivalNeuron(&sigmoid, &sigmoidDeriv);
	this->bias = new GeneticSurvivalBias;
}

void insertSorted(vector<GeneticSurvivalNetwork*> * const sortedPopulation,
		vector<double> * const sortedErrors, const double error,
		GeneticSurvivalNetwork * const net) {
	vector<GeneticSurvivalNetwork*>::iterator netIt;
	vector<double>::iterator errorIt;
	bool inserted = false;
	unsigned int j;

	netIt = sortedPopulation->begin();
	errorIt = sortedErrors->begin();
	// Insert in sorted position
	for (j = 0; j < sortedPopulation->size(); j++) {
		if (error < errorIt[j]) {
			//printf("Inserting at %d, error = %f\n", j, error);
			sortedPopulation->insert(netIt + j, net);
			sortedErrors->insert(errorIt + j, error);
			inserted = true;
			break;
		}
	}
	// If empty, or should be placed last in list
	if (!inserted) {
		//printf("Inserting last, error = %f\n", error);
		sortedPopulation->push_back(net);
		sortedErrors->push_back(error);
		inserted = true;
	}
}

/*
 * This version does not replace the entire population each generation. Two parents are selected at random to create a child.
 * This child is inserted into the list sorted on error. The worst network is destroyed if population exceeds limit.
 * One generation is considered to be the same number of matings as population size.
 * Networks to be mated are selected with the geometric distribution, probability of the top network to be chosen = 0.05
 * Mutation chance dictate the probability of every single weight being mutated.
 */
void GeneticSurvivalNetwork::learn(double **X, double **Y,
		unsigned int length) {
	// Create a population of networks
	printf("Creating population\n");
	vector<GeneticSurvivalNetwork*> sortedPopulation;
	vector<double> sortedErrors;
	// Have one throw away network used to create next child
	sortedPopulation.reserve(populationSize + 1);
	sortedErrors.reserve(populationSize + 1);
	// Rank and insert them in a sorted order
	double error;
	unsigned int i, j;
	vector<GeneticSurvivalNetwork*>::iterator netIt;
	vector<double>::iterator errorIt;
	for (i = 0; i < populationSize + 1; i++) {
		printf("Creating individual\n");
		GeneticSurvivalNetwork *net = getGeneticSurvivalNetwork(numOfInputs,
				numOfHidden);
		// TODO evaluate error here
		error = dRand();

		insertSorted(&sortedPopulation, &sortedErrors, error, net);
	}

	// Debug print the sorted list
	for (errorIt = sortedErrors.begin(); errorIt < sortedErrors.end();
			errorIt++) {
		printf("sortedError: %f\n", *errorIt);
	}

	// Save the best network in the population
	GeneticSurvivalNetwork *best = sortedPopulation.front();
	GeneticSurvivalNetwork *child;

	// For each generation
	unsigned int curGen, genChild, mother, father;
	for (curGen = 0; curGen < generations; curGen++) {
		for (genChild = 0; genChild < populationSize; genChild++) {
			// We recycle the worst network
			child = sortedPopulation.back();
			//printf("error at back: %f\n", sortedErrors.back());
			// Remove it from the list
			sortedPopulation.pop_back();
			sortedErrors.pop_back();
			// Select two networks
			// TODO selectParents(&mother, &father);

			// Create new child through crossover
			// TODO child->crossOver(mother, father)
			// Mutate child
			// TODO child->mutate(weightMutationChance, weightMutationMean, weightMutationHalfPoint, curGen)

			// TODO Evaluate child
			error = dRand();
			//printf("new child error: %f\n", error);
			// Insert child into the sorted list
			insertSorted(&sortedPopulation, &sortedErrors, error, child);
			// Save best network
			best = sortedPopulation.front();
		}
	}

	// Debug print the sorted list
	for (errorIt = sortedErrors.begin(); errorIt < sortedErrors.end();
			errorIt++) {
		printf("sortedError: %f\n", *errorIt);
	}

	// When done, make this network into the best network
	// TODO this->cloneNetwork(best);

	// And destroy population
	// do this last of all!
	best = NULL;
	for (netIt = sortedPopulation.begin(); netIt < sortedPopulation.end();
			netIt++) {
		printf("deleting population\n");
		delete *netIt;
	}
}

/*
 * ------------------------
 * Neuron definition
 * ------------------------
 */
GeneticSurvivalNeuron::GeneticSurvivalNeuron() :
		Neuron() {

}

GeneticSurvivalNeuron::GeneticSurvivalNeuron(
		double (*activationFunction)(double),
		double (*activationDerivative)(double)) :
		Neuron(activationFunction, activationDerivative) {

}

GeneticSurvivalNeuron::~GeneticSurvivalNeuron() {

}
