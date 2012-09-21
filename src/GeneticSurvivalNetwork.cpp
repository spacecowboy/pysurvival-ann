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
#include <vector>
#include <stdio.h>
#include "boost/random.hpp"
#include <time.h>
#include <math.h>

using namespace std;

GeneticSurvivalNetwork* getGeneticSurvivalNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden) {
	// Init random number stuff
	boost::mt19937 eng; // a core engine class
	eng.seed(time(NULL));
	// Uniform distribution 0 to 1 (inclusive)
	boost::uniform_int<> uni_dist(0, 1);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform(
			eng, uni_dist);

	return getGeneticSurvivalNetwork(numOfInputs, numOfHidden, &uniform);
}

GeneticSurvivalNetwork* getGeneticSurvivalNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden,
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> >* uniform) {

	GeneticSurvivalNetwork *net = new GeneticSurvivalNetwork(numOfInputs,
			numOfHidden);

	unsigned int i, j;
	// Connect hidden to input and bias
	// Connect output to hidden
	for (i = 0; i < numOfHidden; i++) {
		net->connectHToB(i, ((*uniform)() - 0.5) * 0.1);
		net->connectOToH(i, ((*uniform)() - 0.5) * 0.1);
		// Inputs
		for (j = 0; j < numOfInputs; j++) {
			net->connectHToI(i, j, ((*uniform)() - 0.5) * 0.1);
		}
	}
	// Connect output to bias
	net->connectOToB(((*uniform)() - 0.5) * 0.1);

	// Set output node to linear activation function
	net->getOutputNeuron()->setActivationFunction(&linear, &linearDeriv);

	return net;
}

GeneticSurvivalNetwork::GeneticSurvivalNetwork(unsigned int numOfInputs,
		unsigned int numOfHidden) :
		FFNetwork(numOfInputs, numOfHidden) {
	populationSize = 50;
	generations = 100;
	weightMutationChance = 0.15;
	weightMutationStdDev = 0.1;
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

void selectParents(
		boost::variate_generator<boost::mt19937&,
				boost::geometric_distribution<int, double> > *geometric,
		unsigned int maximum, unsigned int *mother, unsigned int *father) {

	*mother = (*geometric)();
	while (*mother >= maximum) {
		*mother = (*geometric)();
	}
	// Make sure they are not the same
	*father = *mother;
	while (*father == *mother || *father >= maximum) {
		*father = (*geometric)();
	}
}

void GeneticSurvivalNetwork::crossover(
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
		GeneticSurvivalNetwork* mother, GeneticSurvivalNetwork* father) {
	// Each individual node is replaced with some probability
	unsigned int n;
	for (n = 0; n < numOfHidden; n++) {
		if ((*uniform)() < 0.5)
			((GeneticSurvivalNeuron *) hiddenNeurons[n])->cloneNeuron(
					mother->hiddenNeurons[n]);
		else
			((GeneticSurvivalNeuron *) hiddenNeurons[n])->cloneNeuron(
					father->hiddenNeurons[n]);
	}
	// Then output node
	if ((*uniform)() < 0.5)
		((GeneticSurvivalNeuron *) outputNeuron)->cloneNeuron(
				mother->outputNeuron);
	else
		((GeneticSurvivalNeuron *) outputNeuron)->cloneNeuron(
				father->outputNeuron);

}

void GeneticSurvivalNetwork::mutateWeights(
		boost::variate_generator<boost::mt19937&,
				boost::normal_distribution<double> >* gaussian,
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
		double mutationChance, double stdDev, int deviationHalfPoint,
		int epoch) {

	double currentStdDev = stdDev;
	if (deviationHalfPoint > 0 && epoch > 0) {
		currentStdDev = stdDev * (1.0 - 0.5 * ((double) epoch / (double) deviationHalfPoint));
	}

	unsigned int n;
	for (n = 0; n < numOfHidden; n++) {
		((GeneticSurvivalNeuron*) hiddenNeurons[n])->mutateWeights(gaussian,
				uniform, mutationChance, currentStdDev);
	}
	((GeneticSurvivalNeuron*) outputNeuron)->mutateWeights(gaussian, uniform,
			mutationChance, currentStdDev);
}

unsigned int GeneticSurvivalNetwork::getGenerations() const {
	return generations;
}

void GeneticSurvivalNetwork::setGenerations(unsigned int generations) {
	this->generations = generations;
}

unsigned int GeneticSurvivalNetwork::getPopulationSize() const {
	return populationSize;
}

void GeneticSurvivalNetwork::setPopulationSize(unsigned int populationSize) {
	this->populationSize = populationSize;
}

double GeneticSurvivalNetwork::getWeightMutationChance() const {
	return weightMutationChance;
}

void GeneticSurvivalNetwork::setWeightMutationChance(
		double weightMutationChance) {
	this->weightMutationChance = weightMutationChance;
}

unsigned int GeneticSurvivalNetwork::getWeightMutationHalfPoint() const {
	return weightMutationHalfPoint;
}

void GeneticSurvivalNetwork::setWeightMutationHalfPoint(
		unsigned int weightMutationHalfPoint) {
	this->weightMutationHalfPoint = weightMutationHalfPoint;
}

double GeneticSurvivalNetwork::getWeightMutationStdDev() const {
	return weightMutationStdDev;
}

void GeneticSurvivalNetwork::setWeightMutationStdDev(
		double weightMutationStdDev) {
	this->weightMutationStdDev = weightMutationStdDev;
}

void GeneticSurvivalNetwork::cloneNetwork(GeneticSurvivalNetwork* original) {
	unsigned int n;
	for (n = 0; n < numOfHidden; n++) {
		((GeneticSurvivalNeuron *) hiddenNeurons[n])->cloneNeuron(
				original->hiddenNeurons[n]);
	}
	// Then output node
	((GeneticSurvivalNeuron *) outputNeuron)->cloneNeuron(
			original->outputNeuron);
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
	// Init random number stuff
	boost::mt19937 eng; // a core engine class
	eng.seed(time(NULL));
	// Geometric distribution for selecting parents
	boost::geometric_distribution<int, double> geo_dist(0.95);
	boost::variate_generator<boost::mt19937&,
			boost::geometric_distribution<int, double> > geometric(eng,
			geo_dist);
	// Normal distribution for weight mutation, 0 mean and 1 stddev
	// We can then get any normal distribution with y = mean + stddev * x
	boost::normal_distribution<double> gauss_dist(0, 1);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > gaussian(
			eng, gauss_dist);
	// Uniform distribution 0 to 1 (inclusive)
	boost::uniform_int<> uni_dist(0, 1);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform(
			eng, uni_dist);

	// Create a population of networks
	printf("Creating population\n");
	vector<GeneticSurvivalNetwork*> sortedPopulation;
	vector<double> sortedErrors;
	// Have one throw away network used to create next child
	sortedPopulation.reserve(populationSize + 1);
	sortedErrors.reserve(populationSize + 1);
	// Rank and insert them in a sorted order
	double error;
	unsigned int i;
	vector<GeneticSurvivalNetwork*>::iterator netIt;
	vector<double>::iterator errorIt;
	for (i = 0; i < populationSize + 1; i++) {
		printf("Creating individual\n");
		GeneticSurvivalNetwork *net = getGeneticSurvivalNetwork(numOfInputs,
				numOfHidden, &uniform);
		// TODO evaluate error here
		error = net->output(X[0]);

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
			selectParents(&geometric, populationSize, &mother, &father);
			//printf("Mother: %d, Father: %d\n", mother, father);

			// Create new child through crossover
			child->crossover(&uniform, sortedPopulation[mother],
					sortedPopulation[father]);
			// Mutate child
			child->mutateWeights(&gaussian, &uniform, weightMutationChance,
					weightMutationStdDev, weightMutationHalfPoint, curGen);

			// TODO Evaluate child
			error = child->output(X[0]);
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
	printf("best error: %f\n", best->output(X[0]));

	// When done, make this network into the best network
	this->cloneNetwork(best);

	// And destroy population
	// do this last of all!
	best = NULL;
	for (netIt = sortedPopulation.begin(); netIt < sortedPopulation.end();
			netIt++) {
		//printf("deleting population\n");
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

void GeneticSurvivalNeuron::cloneNeuron(Neuron* original) {
	unsigned int i = 0;
	// First hidden connections
	for (i = 0; i < neuronConnections->size(); i++) {
		neuronConnections->at(i).second =
				original->neuronConnections->at(i).second;
	}

	// Then input connections
	for (i = 0; i < inputConnections->size(); i++) {
		inputConnections->at(i).second =
				original->inputConnections->at(i).second;
	}
}

void GeneticSurvivalNeuron::mutateWeights(
		boost::variate_generator<boost::mt19937&,
				boost::normal_distribution<double> >* gaussian,
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
		double mutationChance, double stdDev) {
	unsigned int n;
	for (n = 0; n < neuronConnections->size(); n++) {
		if ((*uniform)() <= mutationChance)
			neuronConnections->at(n).second += (*gaussian)() * stdDev;
	}
	for (n = 0; n < inputConnections->size(); n++) {
		if ((*uniform)() <= mutationChance)
			inputConnections->at(n).second += (*gaussian)() * stdDev;
	}
}

