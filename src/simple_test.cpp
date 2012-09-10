/*
 * simple_test.cpp
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#include <stdio.h>
#include "FFNeuron.h"
#include "RPropNetwork.h"
#include <exception>

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

	RPropNetwork *rprop = getRPropNetwork(2, 3);

	printf("Factory network out: %f\n", rprop->output(x));

	//double xorIn[4][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
	double **xorIn = new double*[4];
	for (int i = 0; i < 4; i++) {
		xorIn[i] = new double[2];
	}
	xorIn[0][0] = 0;
	xorIn[0][1] = 0;

	xorIn[1][0] = 0;
	xorIn[1][1] = 1;

	xorIn[2][0] = 1;
	xorIn[2][1] = 0;

	xorIn[3][0] = 1;
	xorIn[3][1] = 1;

	double* xorOut = new double[4];
	xorOut[0] = 0;
	xorOut[1] = 1;
	xorOut[2] = 1;
	xorOut[3] = 0;

	printf("Before training of XOR\n");
	printf("%f %f : %f\n", xorIn[0][0], xorIn[0][1], rprop->output(xorIn[0]));
	printf("%f %f : %f\n", xorIn[1][0], xorIn[1][1], rprop->output(xorIn[1]));
	printf("%f %f : %f\n", xorIn[2][0], xorIn[2][1], rprop->output(xorIn[2]));
	printf("%f %f : %f\n", xorIn[3][0], xorIn[3][1], rprop->output(xorIn[3]));

	printf("Learning...\n");

	try {
		rprop->learn(xorIn, xorOut, 4);

		printf("After training of XOR\n");
		printf("%f %f : %f\n", xorIn[0][0], xorIn[0][1],
				rprop->output(xorIn[0]));
		printf("%f %f : %f\n", xorIn[1][0], xorIn[1][1],
				rprop->output(xorIn[1]));
		printf("%f %f : %f\n", xorIn[2][0], xorIn[2][1],
				rprop->output(xorIn[2]));
		printf("%f %f : %f\n", xorIn[3][0], xorIn[3][1],
				rprop->output(xorIn[3]));

	} catch (std::exception& e) {
		printf("Exception cast ");
	}

	delete rprop;
	delete[] xorOut;
	delete[] xorIn[0];
	delete[] xorIn[1];
	delete[] xorIn[2];
	delete[] xorIn[3];
	delete[] xorIn;
}
