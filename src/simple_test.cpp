/*
 * simple_test.cpp
 *
 *  Created on: 10 sep 2012
 *      Author: jonas
 */

#include <stdio.h>
#include "FFNeuron.h"
#include "RPropNetwork.h"
#include "GeneticSurvivalNetwork.h"
#include <exception>
#include <time.h>
//#include <random>
#include <math.h>
#include <iostream>
#include "boost/random.hpp"

void randomTest() {
	boost::mt19937 eng; // a core engine class
	eng.seed(time(NULL));
	//boost::normal_distribution<double> dist;
	boost::uniform_int<> six(1, 6);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(eng,
			six);
	for (int i = 0; i < 10; ++i)
		std::cout << die() << std::endl;

	boost::normal_distribution<double> normal;
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > gauss(
			eng, normal);
	for (int i = 0; i < 10; ++i)
		std::cout << gauss() << std::endl;

	// Uses 1-p compared to numpy
	boost::geometric_distribution<int, double> geom(0.95);
	boost::variate_generator<boost::mt19937&,
			boost::geometric_distribution<int, double> > geo(eng, geom);
	double mean = 0;
	for (int i = 0; i < 1000; ++i) {
		mean += geo();
		//std::cout << geo() << std::endl;
	}
	mean /= 1000;
	std::cout << mean << std::endl;
	mean = (0.95) / 0.05;
	std::cout << mean << std::endl;
	double stddev = sqrt((0.95) / (0.05 * 0.05));
	std::cout << stddev << std::endl;

	double lambda = 10.0;
	boost::exponential_distribution<double> expo(lambda);
	boost::variate_generator<boost::mt19937&,
			boost::exponential_distribution<double> > expon(eng, expo);
	mean = 0;
	for (int i = 0; i < 1000; ++i) {
		mean += expon();
		std::cout << expon() << std::endl;
	}
	mean /= 1000;
	std::cout << "experimental mean: " << mean << std::endl;
	mean = 1 / lambda;
	std::cout << "calculated mean: " << mean << std::endl;
}

void baseTest() {
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

	FFNetwork *ann = new RPropNetwork(2, 0, 1);
	double *outputs = new double[1];

	printf("Before connect, ann out = %f\n", ann->output(x, outputs)[0]);
	ann->connectOToI(0, 0, 1.0);
	printf("Connect to I0, ann out = %f\n", ann->output(x, outputs)[0]);
	ann->connectOToI(0, 1, 1.0);
	printf("Connect to I1, ann out = %f\n", ann->output(x, outputs)[0]);
	ann->connectOToB(0, 1.0);
	printf("Connect to Bias, ann out = %f\n", ann->output(x, outputs)[0]);

	delete ann;
	delete[] outputs;
}

void rPropTest() {
	RPropNetwork *rprop = getRPropNetwork(2, 3);
	double x[2] = { 0.5, 0.5 };
	double *outputs = new double[1];

	printf("Factory network out: %f\n", rprop->output(x, outputs)[0]);

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

	double** xorOut = new double*[4];
	for (int i = 0; i < 4; i++) {
		xorOut[i] = new double[1];
	}
	xorOut[0][0] = 0;
	xorOut[1][0] = 1;
	xorOut[2][0] = 1;
	xorOut[3][0] = 0;

	printf("Before training of XOR\n");
	printf("%f %f : %f\n", xorIn[0][0], xorIn[0][1], rprop->output(xorIn[0], outputs)[0]);
	printf("%f %f : %f\n", xorIn[1][0], xorIn[1][1], rprop->output(xorIn[1], outputs)[0]);
	printf("%f %f : %f\n", xorIn[2][0], xorIn[2][1], rprop->output(xorIn[2], outputs)[0]);
	printf("%f %f : %f\n", xorIn[3][0], xorIn[3][1], rprop->output(xorIn[3], outputs)[0]);

	rprop->setMaxError(0);
	rprop->setMaxEpochs(100);
	printf("Learning...\n");

	try {
		rprop->learn(xorIn, xorOut, 4);

		printf("After training of XOR\n");
		printf("%f %f : %f\n", xorIn[0][0], xorIn[0][1],
				rprop->output(xorIn[0], outputs)[0]);
		printf("%f %f : %f\n", xorIn[1][0], xorIn[1][1],
				rprop->output(xorIn[1], outputs)[0]);
		printf("%f %f : %f\n", xorIn[2][0], xorIn[2][1],
				rprop->output(xorIn[2], outputs)[0]);
		printf("%f %f : %f\n", xorIn[3][0], xorIn[3][1],
				rprop->output(xorIn[3], outputs)[0]);

	} catch (std::exception& e) {
		printf("Exception cast ");
	}

	printf("Train a 100 epochs 10 times\n");
	int t, i;
	rprop->setMaxError(0);
	rprop->setMaxEpochs(100);
	rprop->setPrintEpoch(-1);
	t = clock();
	for (i = 0; i < 10; i++) {
		rprop->learn(xorIn, xorOut, 4);
	}
	float dt = 10 * ((float) clock() - t) / CLOCKS_PER_SEC;
	printf("Training takes %f milliseconds per 100 epochs\n", dt);

	delete rprop;
	delete[] xorOut[0];
	delete[] xorOut[1];
	delete[] xorOut[2];
	delete[] xorOut[3];
	delete[] xorOut;
	delete[] xorIn[0];
	delete[] xorIn[1];
	delete[] xorIn[2];
	delete[] xorIn[3];
	delete[] xorIn;

	delete[] outputs;
}

void geneticSurvivalTest() {
	GeneticSurvivalNetwork* net = getGeneticSurvivalNetwork(2, 3);

	net->setGenerations(100);
	net->setPopulationSize(50);
	//net->setWeightMutationHalfPoint(100);

	double x[2] = { 0.5, 0.5 };

	double *px = new double[2];
	px[0] = 0.5;
	px[1] = 0.5;
	double **ppx = new double*[1];
	ppx[0] = px;

	double *outputs = new double[1];

	printf("Factory gene-network out: %f\n", net->output(x, outputs)[0]);

	net->learn(ppx, NULL, 1);

	printf("After breeding out: %f\n", net->output(x, outputs)[0]);

	delete net;
	delete[] ppx;
	delete[] px;

	delete[] outputs;
}

int main(int argc, char* argv[]) {
	//randomTest();
	baseTest();
	rPropTest();
	geneticSurvivalTest();
}
