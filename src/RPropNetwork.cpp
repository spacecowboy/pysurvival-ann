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
 */
void RPropNetwork::learn(const double * const X,
                         const double * const Y,
                         const unsigned int length)
{
  // Reset log
  initLog(maxEpochs);

  // Used in training
  double dMax = 50, dMin = 0.00001, dPos = 1.2, dNeg = 0.5;

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
                    Y + i * OUTPUT_COUNT,
                    preds + i * OUTPUT_COUNT,
                    OUTPUT_COUNT,
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
          backPropValues[n * LENGTH + i] += derivs[n] * outputs[i];
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
    meanError = getError(errorFunction, Y, length, OUTPUT_COUNT, preds);
    epoch += 1;
    // And log performance
    this->aLogPerf[epoch - 1] = meanError;
  } while (epoch < maxEpochs && meanError > maxError);


  // Clean memory
  delete[] preds;
  delete[] backPropValues;
  delete[] derivs;
  delete[] prevBackPropValues;
  delete[] weightUpdates;
  delete[] prevUpdates;
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

////////////////////////////////////////////
///////////////// OLD /////////////////////
///////////////////////////////////////////

// void RPropNetwork::learn(double *X, double *Y, unsigned int rows) {
// 	double error[numOfOutput];
// 	for (unsigned int i = 0; i < numOfOutput; i++)
// 		error[i] = 0;

// 	double *outputs = new double[numOfOutput];
// 	double deriv = 0;
// 	unsigned int epoch = 0;

// 	int i, n;

// 	do {
//         if (printEpoch > 0 && epoch % (int) printEpoch == 0) {
// 			printf("epoch: %d, error: %f\n", epoch, error[0] / rows);
//         }
//         for (n = 0; n < (int) numOfOutput; n++) {
//             error[n] = 0;
//         }
// 		// Evaluate for each value in input vector
//       for (i = 0; i < (int) rows; i++) {
// 			// First let all neurons evaluate
// 			output(X + i*numOfInputs, outputs);
// 			for (n = 0; n < (int) numOfOutput; n++) {
// 				deriv = SSEDeriv(Y[i*numOfOutput + n], outputs[n]);
// 				error[n] += SSE(Y[i*numOfOutput + n], outputs[n]);
// 				// set error deriv on output node
// 				static_cast<RPropNeuron*>(outputNeurons[n])->addLocalError(
// 						deriv);
// 				// Calculate local derivatives at output and propagate
// 				static_cast<RPropNeuron*>(outputNeurons[n])->calcLocalDerivative(
// 						X + i*numOfInputs);
// 			}

// 			// Calculate local derivatives at all neurons
// 			// and propagate
// 			for (n = numOfHidden - 1; n >= 0; n--) {
// 				static_cast<RPropNeuron*>(hiddenNeurons[n])->calcLocalDerivative(
// 						X + i*numOfInputs);
// 			}
//       }
// 		// Apply weight updates
// 		for (n = 0; n < (int) numOfOutput; n++) {
// 			static_cast<RPropNeuron*>(outputNeurons[n])->applyWeightUpdates();
// 		}
// 		for (n = numOfHidden - 1; n >= 0; n--) {
// 			static_cast<RPropNeuron*>(hiddenNeurons[n])->applyWeightUpdates();
// 		}
// 		epoch += 1;
// 	} while (epoch < maxEpochs && error[0] > maxError);

// 	delete[] outputs;
// }

// /*
//  * -----------------------
//  * RPropNeuron definitions
//  * -----------------------
//  */

// RPropNeuron::RPropNeuron(int id) :
// 		Neuron(id) {
// 	localError = 0;

// 	prevNeuronUpdates = new std::vector<double>;
// 	prevInputUpdates = new std::vector<double>;
// 	prevNeuronDerivs = new std::vector<double>;
// 	prevInputDerivs = new std::vector<double>;

// 	neuronUpdates = new std::vector<double>;
// 	inputUpdates = new std::vector<double>;
// }

// RPropNeuron::RPropNeuron(int id, double (*activationFunction)(double),
// 		double (*activationDerivative)(double)) :
//   Neuron(id, activationFunction, activationDerivative) {
// 	localError = 0;

// 	prevNeuronUpdates = new std::vector<double>;
// 	prevInputUpdates = new std::vector<double>;
// 	prevNeuronDerivs = new std::vector<double>;
// 	prevInputDerivs = new std::vector<double>;

// 	neuronUpdates = new std::vector<double>;
// 	inputUpdates = new std::vector<double>;
// }

// RPropNeuron::~RPropNeuron() {
// 	delete prevNeuronUpdates;
// 	delete prevNeuronDerivs;
// 	delete prevInputDerivs;
// 	delete prevInputUpdates;
// 	delete neuronUpdates;
// 	delete inputUpdates;
// }

// void RPropNeuron::connectToInput(unsigned int index, double weight) {
// 	this->Neuron::connectToInput(index, weight);

// 	prevInputUpdates->push_back(0.1);
// 	prevInputDerivs->push_back(1);
// 	inputUpdates->push_back(0);
// }

// void RPropNeuron::connectToNeuron(Neuron *neuron, double weight) {
// 	this->Neuron::connectToNeuron(neuron, weight);

// 	prevNeuronUpdates->push_back(0.1);
// 	prevNeuronDerivs->push_back(1);
// 	neuronUpdates->push_back(0);
// }

// void RPropNeuron::addLocalError(double error) {
// 	localError += error;
// }

// void RPropNeuron::calcLocalDerivative(double *inputs) {
// 	unsigned int inputIndex, i;
// 	localError *= outputDeriv();

// 	//Propagate the error backwards
// 	//And calculate weight updates
// 	for (i = 0; i < neuronConnections->size(); i++) {
// 		// Propagate backwards
// 		// I know it's an RPropNeuron
// 		static_cast<RPropNeuron*>(neuronConnections->at(i).first)->addLocalError(
// 				localError * neuronConnections->at(i).second);

// 		// Calculate and add weight update
// 		(neuronUpdates->begin())[i] += (localError
// 				* (neuronConnections->at(i).first->output()));
// 	}
// 	// Calculate weight updates to inputs also
// 	for (i = 0; i < inputConnections->size(); i++) {
// 		inputIndex = inputConnections->at(i).first;
// 		(inputUpdates->begin()[i]) += (localError * inputs[inputIndex]);
// 	}

// 	// Set to zero for next iteration
// 	localError = 0;
// }

// void RPropNeuron::applyWeightUpdates() {
// 	unsigned int i;
// 	double prevUpdate, prevDeriv, weightUpdate, deriv;
// 	// Maximum and minimm weight changes
// 	double dMax = 50, dMin = 0.00001;
// 	// How much to adjust updates
// 	double dPos = 1.2, dNeg = 0.5;

// 	for (i = 0; i < neuronConnections->size(); i++) {
// 		prevUpdate = prevNeuronUpdates->at(i);
// 		prevDeriv = prevNeuronDerivs->at(i);
// 		deriv = neuronUpdates->at(i);

// 		if (prevDeriv * deriv > 0) {
// 			// We are on the right track, increase speed!
// 			weightUpdate = std::abs(prevUpdate) * dPos;
// 			// But not too fast!
// 			if (weightUpdate > dMax)
// 				weightUpdate = dMax;
// 			weightUpdate *= sign(deriv);
// 			neuronConnections->at(i).second += weightUpdate;
// 		} else if (prevDeriv * deriv < 0) {
// 			// Shit, we overshot the target!
// 			weightUpdate = std::abs(prevUpdate) * dNeg;
// 			if (weightUpdate < dMin)
// 				weightUpdate = dMin;
// 			weightUpdate *= sign(deriv);
// 			// Go back
// 			neuronConnections->at(i).second -= prevUpdate;
// 			// Next time forget about this disastrous direction
// 			deriv = 0;
// 		} else {
// 			// Previous round we overshot, go forward
// 			weightUpdate = std::abs(prevUpdate) * sign(deriv);
// 			neuronConnections->at(i).second += weightUpdate;
// 		}

// 		prevNeuronDerivs->begin()[i] = deriv;
// 		prevNeuronUpdates->begin()[i] = weightUpdate;
// 		neuronUpdates->begin()[i] = 0;
// 	}
// 	// Calculate weight updates to inputs also
// 	for (i = 0; i < inputConnections->size(); i++) {
// 		prevUpdate = prevInputUpdates->at(i);
// 		prevDeriv = prevInputDerivs->at(i);
// 		deriv = inputUpdates->at(i);

// 		if (prevDeriv * deriv > 0) {
// 			// We are on the right track, increase speed!
// 			weightUpdate = std::abs(prevUpdate) * dPos;
// 			// But not too fast!
// 			if (weightUpdate > dMax)
// 				weightUpdate = dMax;
// 			weightUpdate *= sign(deriv);
// 			inputConnections->at(i).second += weightUpdate;
// 		} else if (prevDeriv * deriv < 0) {
// 			// Shit, we overshot the target!
// 			weightUpdate = std::abs(prevUpdate) * dNeg;
// 			if (weightUpdate < dMin)
// 				weightUpdate = dMin;
// 			weightUpdate *= sign(deriv);
// 			// Go back
// 			inputConnections->at(i).second -= prevUpdate;
// 			// Next time forget about this disastrous direction
// 			deriv = 0;
// 		} else {
// 			// Previous round we overshot, go forward
// 			weightUpdate = std::abs(prevUpdate) * sign(deriv);
// 			inputConnections->at(i).second += weightUpdate;
// 		}

// 		prevInputDerivs->begin()[i] = deriv;
// 		prevInputUpdates->begin()[i] = weightUpdate;
// 		inputUpdates->begin()[i] = 0;
// 	}
// }
