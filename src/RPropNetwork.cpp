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

/*
 * Returns a single layer FeedForward Network that is trained by the RProp algorithm.
 * It is up to the user to free this network from the heap once done.
 */
RPropNetwork* getRPropNetwork(unsigned int numOfInputs, unsigned int numOfHidden) {
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

RPropNetwork::RPropNetwork(unsigned int numOfInputs, unsigned int numOfHidden) :
		FFNetwork(numOfInputs, numOfHidden) {

}

void RPropNetwork::learn(double **data) {

}
