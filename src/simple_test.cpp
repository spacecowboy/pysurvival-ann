/*
 * simple_test.cpp
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#include <stdio.h>
#include "FFNeuron.h"
#include "RPropNetwork.h"

int main(int argc, char* argv[]) {

	Bias b;
	printf("Bias output is %f\n", b.output());
	double x[2] = { 0.5, 0.5 };
	Neuron *n = new Neuron;
	n->connectToInput(0, 0.5);
	n->connectToInput(1, 0.7);

	Neuron *o = new Neuron;
	o->connectToNeuron(n, -1);
	printf("Neuron output is %f\n", o->output(x));
	printf("Neuron outputDeriv is %f\n", o->outputDeriv());

	delete o;
	delete n;

	FFNetwork *ann = new RPropNetwork(2, 0);

	printf("Before connect, ann out = %f\n", ann->output(x));
	ann->connectOToI(0, 1.0);
	printf("Connect to I0, ann out = %f\n", ann->output(x));
	ann->connectOToI(1, 1.0);
	printf("Connect to I1, ann out = %f\n", ann->output(x));
	ann->connectOToB(1.0);
	printf("Connect to Bias, ann out = %f\n", ann->output(x));

	delete ann;

	ann = getRPropNetwork(2, 2);

	printf("Factory network out: %f\n", ann->output(x));

	delete ann;
}
