#include "MatrixNetwork.hpp"
#include "activationfunctions.hpp"
#include <stdexcept>
#include <stdio.h>
#include <algorithm>
//#include <iostream>
//#include <fstream>
//#include <vector>
using namespace std;

MatrixNetwork::MatrixNetwork(unsigned int numOfInput,
                             unsigned int numOfHidden,
                             unsigned int numOfOutput) :
  aLogPerf(nullptr),
  logPerfLength(0),
  // network structure
  INPUT_START(0),
  INPUT_END(INPUT_START + numOfInput),
  BIAS_INDEX(INPUT_END),
  BIAS_START(BIAS_INDEX),
  BIAS_END(BIAS_START + 1),
  HIDDEN_START(BIAS_END),
  HIDDEN_END(HIDDEN_START + numOfHidden),
  OUTPUT_START(HIDDEN_END),
  OUTPUT_END(OUTPUT_START + numOfOutput),
  // counts
  // plus 1 for bias
  LENGTH(1 + numOfInput + numOfHidden + numOfOutput),
  INPUT_COUNT(numOfInput),
  HIDDEN_COUNT(numOfHidden),
  OUTPUT_COUNT(numOfOutput),
// matrices
  actFuncs(new ActivationFuncEnum[LENGTH]()),
  conns(new unsigned int[LENGTH*LENGTH]()),
  weights(new double[LENGTH*LENGTH]()),
  outputs(new double[LENGTH]())
{
}

MatrixNetwork::~MatrixNetwork() {
  delete[] actFuncs;
  delete[] conns;
  delete[] weights;
  delete[] outputs;
  delete[] aLogPerf;
}

double *MatrixNetwork::getLogPerf() {
  return aLogPerf;
}

unsigned int MatrixNetwork::getLogPerfLength() {
  return logPerfLength;
}

/**
 * Sets the activation function of the output layer
 */
void MatrixNetwork::setOutputActivationFunction(ActivationFuncEnum func) {
  for (unsigned int i = OUTPUT_START; i < OUTPUT_END; i++) {
    actFuncs[i] = func;
  }
}

ActivationFuncEnum MatrixNetwork::getOutputActivationFunction() {
  return actFuncs[OUTPUT_START];
}

/**
 * Sets the activation function of the hidden layer
 */
void MatrixNetwork::setHiddenActivationFunction(ActivationFuncEnum func) {
  for (unsigned int i = HIDDEN_START; i < HIDDEN_END; i++) {
    actFuncs[i] = func;
  }
}

ActivationFuncEnum MatrixNetwork::getHiddenActivationFunction() {
  return actFuncs[HIDDEN_START];
}


double *MatrixNetwork::output(const double * const inputs,
               double * const ret_outputs) {
  unsigned int i, j, target;
  double sum;
  // First set input values
  for (i = INPUT_START; i < INPUT_END; i++) {
    outputs[i] = inputs[i];
  }
  // Make sure bias is 1
  outputs[BIAS_INDEX] = 1;

  // Calculate neurons
  for (i = HIDDEN_START; i < OUTPUT_END; i++) {
    sum = 0;
    // No recursive connections allowed
    for (j = INPUT_START; j < i; j++) {
      target = LENGTH * i + j;
      if (conns[target] != 0)
        sum += weights[target] * outputs[j];
    }
    //printf("\nSum = %f", sum);
    outputs[i] = evaluateActFunction(actFuncs[i], sum);
    //printf("\nOut = %f", outputs[i]);
  }

  // Copy values to output array
  std::copy(outputs + OUTPUT_START,
            outputs + OUTPUT_END,
            ret_outputs);

  return ret_outputs;
}
