#include "MatrixNetwork.hpp"
#include "activationfunctions.hpp"
#include <stdexcept>
#include <stdio.h>
#include <algorithm>
#include <math.h>
//#include <iostream>
//#include <fstream>
#include <vector>
using namespace std;

MatrixNetwork::MatrixNetwork(unsigned int numOfInput,
                             unsigned int numOfHidden,
                             unsigned int numOfOutput) :
  //aLogPerf(std::vector<double>),
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
  actFuncs(LENGTH, LOGSIG),
  conns(LENGTH*LENGTH, 0),
  weights(LENGTH*LENGTH, 0.0),
  outputs(LENGTH, 0),
  // By default, no dropout prob, e.g. probability to remain is 1.0
  dropoutProbs(LENGTH*LENGTH, 1.0),
  // 1.0, not dropped
  dropoutStatus(LENGTH*LENGTH, 1.0)
{
}

MatrixNetwork::~MatrixNetwork() {
}

//double *MatrixNetwork::getLogPerf() {
//  return aLogPerf;
//}

//unsigned int MatrixNetwork::getLogPerfLength() {
//  return logPerfLength;
//}

void MatrixNetwork::initLog(const unsigned int length) {
  aLogPerf.clear();
  aLogPerf.resize(length);
}

/**
 * Sets the activation function of the output layer
 */
void MatrixNetwork::setOutputActivationFunction(ActivationFuncEnum func) {
  for (unsigned int i = OUTPUT_START; i < OUTPUT_END; i++) {
    actFuncs.at(i) = func;
  }
}

ActivationFuncEnum MatrixNetwork::getOutputActivationFunction() {
  return actFuncs.at(OUTPUT_START);
}

/**
 * Sets the activation function of the hidden layer
 */
void MatrixNetwork::setHiddenActivationFunction(ActivationFuncEnum func) {
  for (unsigned int i = HIDDEN_START; i < HIDDEN_END; i++) {
    actFuncs.at(i) = func;
  }
}

ActivationFuncEnum MatrixNetwork::getHiddenActivationFunction() {
  return actFuncs.at(HIDDEN_START);
}

void MatrixNetwork::output(const std::vector<double>::const_iterator inputStart,
                           std::vector<double>::iterator outputStart) {
  output(inputStart, true, outputStart);
}

void MatrixNetwork::output(const std::vector<double>::const_iterator inputStart,
                           const bool doScaling,
                           std::vector<double>::iterator outputStart) {
  output(inputStart, doScaling, !doScaling, outputStart);
}

void MatrixNetwork::output(const std::vector<double>::const_iterator inputStart,
                           const bool doScaling,
                           const bool doDropout,
                           std::vector<double>::iterator outputStart) {
  unsigned int i, j, target;
  double sum, outputSum, outputMax=0, scale=1.0;
  bool first = true;

  // scaling and dropout are mutually exclusive. scaling is used after
  // training, and dropout is used during training.
  if (doScaling && doDropout) {
    throw "Scaling and Dropout can't both be active at once.";
  }

  // First set input values
  for (i = 0; i < INPUT_COUNT; i++) {
    j = INPUT_START + i;
    // A connection to self means neuron is active
    if (1 == conns.at(LENGTH * j + j)) {
      outputs.at(i) = *(inputStart + i);
    } else {
      outputs.at(i) = 0;
    }
  }
  // Make sure bias is 1
  outputs.at(BIAS_INDEX) = 1;

  // Calculate neurons
  for (i = HIDDEN_START; i < OUTPUT_END; i++) {
    // A connection to self means neuron is active
    if (1 == conns.at(LENGTH * i + i)) {
      sum = 0;
      // No recursive connections allowed, so j < i
      for (j = INPUT_START; j < i; j++) {
        target = LENGTH * i + j;

        if (doScaling) {
          scale = dropoutProbs.at(target);
        }
        // Only allow valid probabilities
        if (scale < 0 || scale > 1) {
          scale = 1.0;
        }

        // Take dropout into account
        if (1 == conns.at(target) &&
            (!doDropout || 1 == dropoutStatus.at(target))) {
          sum += scale * weights.at(target) * outputs.at(j);
        }
      }

      outputs.at(i) = evaluateActFunction(actFuncs.at(i), sum);

      // Keep track of largest output neuron value for normalization
      if (i >= OUTPUT_START && SOFTMAX == actFuncs.at(i)
          && (first || outputs.at(i) > outputMax)) {
        outputMax = outputs.at(i);
        first = false;
      }
    } else {
      // Neuron is not active
      outputs.at(i) = 0;
    }
  }

  // Need to do Softmax calculation
  if (SOFTMAX == actFuncs.at(OUTPUT_START)) {
    // Set outputSum to zero
    outputSum = 0;

    // First calculate exponential outputs of normalized values
    for (i = OUTPUT_START; i < OUTPUT_END; i++) {
      if (1 == conns.at(LENGTH * i + i)) {
        // Only active neurons are included, other should have been set to 0
        // Use outputmax to protect against overflow
        outputs.at(i) = exp(outputs.at(i) - outputMax);
        // Remember sum of all outputs
        outputSum += outputs.at(i);
      }
    }
    // Then divide by the sum
    for (i = OUTPUT_START; i < OUTPUT_END; i++) {
      if (1 == conns.at(LENGTH * i + i)) {
        outputs.at(i) /= outputSum;
      }
    }
  }

  // Copy values to output array
  std::copy(outputs.begin() + OUTPUT_START,
            outputs.end(),
            outputStart);
}


void MatrixNetwork::dropoutConns() {
  unsigned int i, j, target;
  for (i = HIDDEN_START; i < OUTPUT_END; i++) {
    // Can only be connected "back", so j < i
    for (j = INPUT_START; j < i; j++) {
      target = LENGTH * i + j;

      // Only sample for active connections
      if (1 == conns.at(target)) {
        dropoutStatus.at(target) =
          (rand.uniform() <= dropoutProbs.at(target));
      }
    }
  }
}

void MatrixNetwork::dropoutReset() {
  std::fill(dropoutStatus.begin(), dropoutStatus.end(), 1.0);
}
