/*
 * RPropNetwork.cpp
 *
 *  Created on: 29 oct 2013
 *      Author: Jonas Kalderstam
 */

#include "RPropNetwork.hpp"
#include "ErrorFunctions.hpp"
#include "activationfunctions.hpp"
#include "drand.h"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <exception>      // std::exception

using namespace std;

signed int sign(double x)
{
  if (x >= 0)
    return 1;
  else
    return -1;
}

RPropNetwork::RPropNetwork(const unsigned int numOfInputs,
                           const unsigned int numOfHidden,
                           const unsigned int numOfOutputs) :
  MatrixNetwork(numOfInputs, numOfHidden, numOfOutputs),
  maxEpochs(10000),
  maxError(0.0001),
  errorFunction(ErrorFunction::ERROR_MSE)
{

}

/**
 * Check that all elements of an array is less
 * than the specified value.
 */
bool allLessThan(const double * const array,
                 const unsigned int length,
                 const double value)
{
  for (int i = 0; i < length; i++) {
    if (array[i] >= value) {
      return false;
    }
  }
  return true;
}


/**
 * Trains the Neural Network using the RProp algorithm.
 * To be specific, it is the iRProp+ algorithm as explained in:
 *
 * "Improving the Rprop Learning Algorithm"
 * Christian Igel and Michael Hüsken
 * Proceedings of the Second International Symposium on Neural Computation,
 *  NC’2000, pp. 115–121, ICSC Academic Press, 2000
 *
 * Regarding error functions, make sure their derivatives return proper values.
 * As an example for Mean Square Error:
 * E = (T - Y)^2
 * dE = -(T - Y) = (Y - T)
 *
 * The minus sign is important.
 */
int RPropNetwork::learn(const double * const X,
                        const double * const Y,
                        const unsigned int length)
{
  int retval = 0;
  // Reset log
  initLog(maxEpochs);

  // Used in training
  double dMax = 50, dMin = 0.00001, dPos = 1.2, dNeg = 0.5;
  // Cache used for some error functions
  ErrorCache *cache = getErrorCache(errorFunction);
  if (cache != NULL) {
    cache->clear();
  }

  // Local variables. () sets all to zero
  double meanError = 1 + maxError;
  double prevError = 1 + meanError;
  double *preds = new double[length * OUTPUT_COUNT]();
  double *backPropValues = new double[LENGTH * LENGTH]();
  double *derivs = new double[LENGTH];
  double *prevBackPropValues = new double[LENGTH * LENGTH]();
  std::fill(prevBackPropValues, prevBackPropValues + LENGTH * LENGTH, 1.0);
  double *weightUpdates = new double[LENGTH * LENGTH]();
  double *prevUpdates = new double[LENGTH * LENGTH]();
  std::fill(prevUpdates, prevUpdates + LENGTH * LENGTH, 0.1);

  unsigned int epoch = 0;

  try {

  // Train
  do {
    // Reset arrays
    std::fill(backPropValues, backPropValues + LENGTH * LENGTH, 0);

    // Evaluate for each value in input vector
    for (int i = 0; i < length; i++) {
      std::fill(derivs, derivs + LENGTH, 0);

      // First let all neurons evaluate
      output(X + i * INPUT_COUNT, preds + i * OUTPUT_COUNT);

      // Calc output derivative: dE/dY
      getDerivative(errorFunction,
                    Y,
                    length,
                    OUTPUT_COUNT,
                    preds,
                    i * OUTPUT_COUNT,
                    cache,
                    derivs + OUTPUT_START);

      // Iterate backwards over the network
      for (int n = OUTPUT_END - 1; n >= HIDDEN_START; n--) {
        // Multiply with derivative to neuron input: dY/dI
        derivs[n] *= evaluateActFuncDerivative(actFuncs[n], outputs[n]);
        // Iterate over the connections of this neuron
        for (int i = 0; i < n; i++) {
          if (conns[n * LENGTH + i] != 1) {
            continue;
          }
          // Propagate error backwards
          derivs[i] += derivs[n] * weights[n * LENGTH + i];
          // Calc update for this connection: dI/dWij
          backPropValues[n * LENGTH + i] += -derivs[n] * outputs[i];
        }
      }
    }

    // Apply updates
    for (int n = HIDDEN_START * LENGTH; n < LENGTH * LENGTH; n++) {
      if (prevBackPropValues[n] * backPropValues[n] > 0) {
        // On the right track, increase speed!
        weightUpdates[n] = abs(prevUpdates[n]) * dPos;
        if (weightUpdates[n] > dMax)
          weightUpdates[n] = dMax;
        weightUpdates[n] *= sign(backPropValues[n]);
        // remember for next round
        prevUpdates[n] = weightUpdates[n];
      }
      else if (prevBackPropValues[n] * backPropValues[n] < 0) {
        // Overshot the target, go back
        // This if turns the algorithm into iRprop+ instead of Rprop+
        // Normal Rprop+ did this every time
        if (meanError > prevError) {
          weightUpdates[n] = -prevUpdates[n];
        }
        // remember for next round
        prevUpdates[n] = abs(prevUpdates[n]) * dNeg;
        if (prevUpdates[n] < dMin)
          prevUpdates[n] = dMin;
        // Next time forget this step
        backPropValues[n] = 0;
      }
      else {
        // Previous round we overshot, go forward here
        weightUpdates[n] = sign(backPropValues[n]) * prevUpdates[n];
        // remember for next round
        prevUpdates[n] = weightUpdates[n];
      }
      // Add actual weights
      weights[n] += weightUpdates[n];
      // Remember derivative
      prevBackPropValues[n] = backPropValues[n];
    }

    // Evaluate again to calculate new error
    for (int i = 0; i < length; i++) {
      // First let all neurons evaluate
      output(X + i * INPUT_COUNT, preds + i * OUTPUT_COUNT);
    }
    prevError = meanError;
    meanError = getError(errorFunction, Y, length, OUTPUT_COUNT, preds, cache);
    epoch += 1;
    // And log performance
    this->aLogPerf[epoch - 1] = meanError;
  } while (epoch < maxEpochs && meanError > maxError);

  } catch (std::exception& e) {
    std::cerr << "\nException thrown: " << e.what() << "\n\n";
    retval = 1;
  }

  // Clean memory
  delete[] preds;
  delete[] backPropValues;
  delete[] derivs;
  delete[] prevBackPropValues;
  delete[] weightUpdates;
  delete[] prevUpdates;
  if (NULL != cache)
    delete cache;

  return retval;
}


// Getters and Setters
unsigned int RPropNetwork::getMaxEpochs() const
{
  return maxEpochs;
}

void RPropNetwork::setMaxEpochs(unsigned int maxEpochs)
{
  this->maxEpochs = maxEpochs;
}

double RPropNetwork::getMaxError() const
{
  return maxError;
}

void RPropNetwork::setMaxError(double maxError)
{
  this->maxError = maxError;
}

ErrorFunction RPropNetwork::getErrorFunction() const
{
  return errorFunction;
}

void RPropNetwork::setErrorFunction(ErrorFunction val)
{
  errorFunction = val;
}
