/*
 * RPropNetwork.cpp
 *
 *  Created on: 29 oct 2013
 *      Author: Jonas Kalderstam
 */

#include "RPropNetwork.hpp"
#include "ErrorFunctions.hpp"
#include "activationfunctions.hpp"
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
  maxEpochs(1000),
  maxError(0.0001),
  minErrorFrac(0.01),
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
  for (unsigned int i = 0; i < length; i++) {
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
  initLog(maxEpochs * OUTPUT_COUNT);

  // Used in training
  double dMax = 50, dMin = 0.00001, dPos = 1.2, dNeg = 0.5;
  // Cache used for some error functions
  ErrorCache *cache = getErrorCache(errorFunction);
  if (cache != NULL) {
    // Initialize it
    cache->clear();
    cache->verifyInit(Y, length);
  }

  // Used for early stopping
  const unsigned int relDiffCount = 100;
  double relDiffLimit, relDiff;

  // Used to calculate error
  double *errors = new double[OUTPUT_COUNT * length];
  double avgErrors[OUTPUT_COUNT];

  // Local variables. () sets all to zero
  double meanError = 1 + maxError;
  double prevError = 1 + meanError;
  double *preds2 = new double[length * OUTPUT_COUNT];
  double *backPropValues = new double[LENGTH * LENGTH];
  //  double *derivs = new double[LENGTH];
  double *prevBackPropValues = new double[LENGTH * LENGTH]();
  double *weightUpdates = new double[LENGTH * LENGTH]();
  double *prevUpdates = new double[LENGTH * LENGTH]();

  for (unsigned int i = 0; i < LENGTH*LENGTH; i++) {
    prevBackPropValues[i] = 1.0;
    prevUpdates[i] = 0.1;
  }

  unsigned int epoch = 0;

  try {
  // Train
  do {
    // Reset arrays
    for (unsigned int i = 0; i < LENGTH*LENGTH; i++) {
      backPropValues[i] = 0.0;
    }

# pragma omp parallel default(none) shared(backPropValues, cache)
    {
      double *outValues = new double[LENGTH];
      double *preds = new double[length * OUTPUT_COUNT]();
      double *derivs = new double[LENGTH];
      // Evaluate for each value in input vector
# pragma omp for
      for (unsigned int i = 0; i < length; i++) {
        for (unsigned int j = 0; j < LENGTH; j++) {
          derivs[j] = 0;
        }
        // The class member "outputs" must be protected from
        // concurrent modifications, hence the critical region.
        // This should not be a fairly short operation
# pragma omp critical
        {
          // First let all neurons evaluate
          output(X + i * INPUT_COUNT, preds + i * OUTPUT_COUNT);
          // Copy to local array
          for (unsigned int nc = 0; nc < LENGTH; nc++) {
            outValues[nc] = outputs[nc];
          }
        }
        // End parallel critical

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
        // Backwards operation so sign is very important
        for (int n = OUTPUT_END - 1; n >= static_cast<int>(HIDDEN_START); n--) {
          // Multiply with derivative to neuron input: dY/dI
          derivs[n] *= evaluateActFuncDerivative(actFuncs[n], outValues[n]);
          // Iterate over the connections of this neuron
          for (int i = 0; i < n; i++) {
            if (conns[n * LENGTH + i] != 1) {
              continue;
            }
            // Propagate error backwards
            derivs[i] += derivs[n] * weights[n * LENGTH + i];
            // Calc update for this connection: dI/dWij
#pragma omp atomic
            backPropValues[n * LENGTH + i] += -derivs[n] * outValues[i];
          }
        }
      }
      // End parallel for

      delete[] outValues;
      delete[] preds;
      delete[] derivs;
    }
    // End parallel

    // Apply updates
    for (unsigned int n = HIDDEN_START * LENGTH; n < LENGTH * LENGTH; n++) {
      if (prevBackPropValues[n] * backPropValues[n] > 0) {
        // On the right track, increase speed!
        weightUpdates[n] = abs(prevUpdates[n]) * dPos;
        if (weightUpdates[n] > dMax)
          weightUpdates[n] = dMax;
        weightUpdates[n] *= sign(backPropValues[n]);
        // remember for next round
        prevUpdates[n] = weightUpdates[n];
      } else if (prevBackPropValues[n] * backPropValues[n] < 0) {
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
      } else {
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
    for (unsigned int i = 0; i < length; i++) {
      // First let all neurons evaluate
      output(X + i * INPUT_COUNT, preds2 + i * OUTPUT_COUNT);
    }

    epoch += 1;

    // Calculate current error
    getAllErrors(errorFunction, Y, length, OUTPUT_COUNT,
                 preds2, cache, errors);
    averagePatternError(errors, length, OUTPUT_COUNT, avgErrors);

    prevError = meanError;

    // Calculate mean and log errors
    meanError = 0;
    for (unsigned int i = 0; i < OUTPUT_COUNT; i++) {
      meanError += avgErrors[i];
      this->aLogPerf[OUTPUT_COUNT * (epoch - 1) + i] = avgErrors[i];
    }
    meanError /= static_cast<double>(OUTPUT_COUNT);

    // Calculate relative error change
    // If the error did not change enough during the last
    // 100 epochs, then terminate.
    if (epoch - 1 > relDiffCount && this->minErrorFrac > 0) {
      relDiffLimit = this->minErrorFrac * avgErrors[0];
      relDiff = abs(this->aLogPerf[OUTPUT_COUNT * (epoch - 1)] -
                    this->aLogPerf[OUTPUT_COUNT * (epoch - 1 - relDiffCount)]);
      if (relDiff < relDiffLimit) {
        // It's small, break the loop.
        // printf("\nTermination time epoch %d", epoch);
        break;
      }
    }
  } while (epoch < maxEpochs && meanError > maxError);
  } catch (std::exception& e) {
    std::cerr << "\nException thrown: " << e.what() << "\n\n";
    retval = 1;
  }

  // Clean memory
  delete[] errors;
  delete[] preds2;
  delete[] backPropValues;
  //  delete[] derivs;
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

double RPropNetwork::getMinErrorFrac() const {
  return minErrorFrac;
}

void RPropNetwork::setMinErrorFrac(double minErrorFrac) {
  this->minErrorFrac = minErrorFrac;
}

ErrorFunction RPropNetwork::getErrorFunction() const
{
  return errorFunction;
}

void RPropNetwork::setErrorFunction(ErrorFunction val)
{
  errorFunction = val;
}
