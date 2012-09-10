/*
 * simple_test.cpp
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#include <stdio.h>
#include "FFNeuron.h"
#include "FFNetwork.h"

int main(int argc, char* argv[]) {

	Bias b;
	printf("Bias output is %f\n", b.output());
	double x[2] = { -1.0, 1.0 };
	Neuron *n = new Neuron;
	n->connectToInput(0, 0.5);
	n->connectToInput(1, 0.7);

	Neuron *o = new Neuron;
	o->connectToNeuron(n, -1);
	printf("Neuron output is %f\n", o->output(x));
	printf("Neuron outputDeriv is %f\n", o->outputDeriv());

	delete o;
	delete n;

	FFNetwork *ann = new FFNetwork(2, 0);



	delete ann;
}
