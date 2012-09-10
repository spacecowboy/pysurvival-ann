/*
 * RPropNetwork.h
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#ifndef RPROPNETWORK_H_
#define RPROPNETWORK_H_

#include "FFNetwork.h"

class RPropNetwork: public FFNetwork {
public:
	RPropNetwork(unsigned int numOfInputs, unsigned int numOfHidden);
	virtual void learn(double **data);
};

/*
 * Returns a FeedForward network that is trained by the RProp algorithm. The network is
 * composed of one layer of hidden units fully connected to inputs. The single output
 * unit is fully connected to the hidden layer. All neurons are connected to the bias.
 */
RPropNetwork* getRPropNetwork(unsigned int numOfInputs, unsigned int numOfHidden);

#endif /* RPROPNETWORK_H_ */
