/*
  A network which trains by use of the cascade correlation algorithm.
  RProp is used to train both output layer and hidden layers.
*/

#include "CascadeNetwork.h"
#include "RPropNetwork.h"
#include "FFNeuron.h"
#include "activationfunctions.h"
#include "boost/random.hpp"
#include <vector>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
using namespace std;

CascadeNetwork::CascadeNetwork(unsigned int numOfInputs,
                                 unsigned int numOfOutput) :
  RPropNetwork(numOfInputs, 0, numOfOutput) {
  maxHidden = 30;
  maxHiddenEpochs = 1000;
}

void CascadeNetwork::initNodes() {
  hiddenRCascadeNeurons = new std::vector<RCascadeNeuron*>;

  this->hiddenNeurons = new Neuron*[0];
  unsigned int i;

  this->outputNeurons = new Neuron*[this->numOfOutput];
  for (i = 0; i < this->numOfOutput; i++) {
    this->outputNeurons[i] = new RPropNeuron(i, &sigmoid, &sigmoidDeriv);
  }

  this->bias = new RPropBias;
}

unsigned int CascadeNetwork::getNumOfHidden() const {
  unsigned int retval = 0;
  if (this->hiddenRCascadeNeurons != NULL) {
    retval = this->hiddenRCascadeNeurons->size();
  }
  return retval;
}


double *CascadeNetwork::output(double *inputs, double *output) {
	// Iterate over the neurons in order and calculate their outputs.
	unsigned int i;
	for (i = 0; i < this->hiddenRCascadeNeurons->size(); i++) {
      this->hiddenRCascadeNeurons->at(i)->output(inputs);
	}
    return this->FFNetwork::output(inputs, output);
	// Finally the output neurons
    //	for (i = 0; i < numOfOutput; i++) {
	//	output[i] = outputNeurons[i]->output(inputs);
	//}

    //	return output;
}

void CascadeNetwork::learn(double *X, double *Y, unsigned int rows) {
  // Init random number stuff
  boost::mt19937 eng; // a core engine class
  eng.seed(time(NULL));

  // Uniform distribution 0 to 1 (inclusive)
  boost::uniform_int<> uni_dist(0, 1);
  boost::variate_generator<boost::mt19937&,
                           boost::uniform_int<> > uniform(eng, uni_dist);


  // Init error[numOfOutput]
  double *error = new double[numOfOutput];
  // Init pError[numOfOutput * rows]
  double *patError = new double[numOfOutput * rows];
  // Init outputs
  double *outputs = new double[numOfOutput * rows];

  RCascadeNeuron *candidate;

  unsigned int i;
  // Step 1, train the output neuron (has its own virtual error function)
  trainOutputs(X, Y, rows);

  // Calculate C-index and p-specific C-sums (must be virtual function)
  // Also saves outputs
  calcErrors(X, Y, rows, patError, error, outputs);
  printf("Average error so far: %f\n", error[0]);
  printf("maxError: %d\n", maxError);
  // While some condition
  unsigned int neuronCount = 0;
  while (neuronCount < maxHidden && error[0] > maxError) {
    printf("Creating candidate\n");
    // Create candidate
    candidate = new RCascadeNeuron(neuronCount, &hyperbole, &hyperboleDeriv);
    // connect candidate
    candidate->connectToNeuron(bias,
                               (uniform() - 0.5));
    for (i = 0; i < numOfInputs; i++) {
      // random weight
      candidate->connectToInput(i,
                                (uniform() - 0.5));
    }
    for (i = 0; i < hiddenRCascadeNeurons->size(); i++) {
      candidate->connectToNeuron(hiddenRCascadeNeurons->at(i),
                                 (uniform() - 0.5));
    }
    printf("Training candidate\n");
    // Train candidate
    candidate->learn(patError, error, X, outputs, rows, numOfInputs);
    printf("Installing candidate\n");
    // Install candidate
    hiddenRCascadeNeurons->push_back(candidate);
    // Connect to output neurons and train them
    for (i = 0; i < numOfOutput; i++) {
      static_cast<RPropNeuron*>(outputNeurons[i])->connectToNeuron(candidate,
                                        (uniform() - 0.5));
    }
    printf("Training outputs\n");
    trainOutputs(X, Y, rows);

    // Calculate C-index and p-specific C-sums (must be virtual function)
    // Also saves outputs
    calcErrors(X, Y, rows, patError, error, outputs);
    printf("Average error so far: %f\n", error[0]);

    // Remember to increment
    neuronCount++;
  }
  // END While
  delete[] outputs;
  delete[] patError;
  delete[] error;
}

/*
 * Simple call normal RProp training procedure.
 * Because hidden neurons are not stored in normal array,
 * they can not be touched by the training procedure anymore.
 */
void CascadeNetwork::trainOutputs(double *X, double *Y, unsigned int rows) {
  this->RPropNetwork::learn(X, Y, rows);
}

/*
 * Calculates the sum square error. Both the individual values,
 * and average value. Also stores the output values
 */
void CascadeNetwork::calcErrors(double *X, double *Y, unsigned int rows,
                                double *patError, double *error,
                                double *outputs) {
  unsigned int poIndex;
  // Zero error array first
  memset(error, 0, numOfOutput * sizeof(double));

  for (unsigned int i = 0; i < rows; i++)
    {
      // Save outputs for covariance calculation
      output(X + i*numOfInputs, outputs + i*numOfOutput);

      for (unsigned int j = 0; j < numOfOutput; ++j)
        {
          poIndex = i*numOfOutput + j;
          patError[poIndex] = SSE(Y[poIndex], outputs[poIndex]);
          // Calc average
          error[j] += (patError[poIndex] / rows);
        }
    }
}

bool CascadeNetwork::getNeuronWeightFromHidden(unsigned int fromId,
                                               int toId, double *weight) {
  printf("getNW\n");
  if (fromId >= getNumOfHidden() ||
      (toId > -1 && (unsigned int) toId >= getNumOfHidden())) {
    throw invalid_argument("Id was larger than number of nodes");
  }

  return hiddenRCascadeNeurons->at(fromId)->getNeuronWeight(toId, weight);
}

bool CascadeNetwork::getInputWeightFromHidden(unsigned int fromId,
                                              unsigned int toIndex, double *weight) {
  printf("getIW\n");
  if (fromId >= getNumOfHidden() || toIndex >= numOfInputs) {
    throw invalid_argument("Id was larger than number of nodes or \
index was greater than number of inputs");
  }

  return hiddenRCascadeNeurons->at(fromId)->getInputWeight(toIndex, weight);
}

Neuron* CascadeNetwork::getHiddenNeuron(unsigned int id) const {
  return hiddenRCascadeNeurons->at(id);
}

unsigned int CascadeNetwork::getMaxHidden() const {
  return maxHidden;
}
void CascadeNetwork::setMaxHidden(unsigned int num)  {
  maxHidden = num;
}
unsigned int CascadeNetwork::getMaxHiddenEpochs() const {
  return maxHiddenEpochs;
}
void CascadeNetwork::setMaxHiddenEpochs(unsigned int num) {
  maxHiddenEpochs = num;
}


/*
 * -----------------------
 * RCascadeNeuron definitions
 * -----------------------
 */

RCascadeNeuron::RCascadeNeuron(int id) :
  RPropNeuron(id) {
}

RCascadeNeuron::RCascadeNeuron(int id, double (*activationFunction)(double),
                               double (*activationDerivative)(double)) :
  RPropNeuron(id, activationFunction, activationDerivative) {
}

RCascadeNeuron::~RCascadeNeuron() {
}

/*
 * Assuming only one output to make it faster to implement.
 */
void RCascadeNeuron::learn(double *patError, double *error,
                      double *X, double *outputs,
                      unsigned int rows, unsigned int numOfInputs) {
  unsigned int epoch, p, i, inputIndex;
  int covariance = 1;
  // temporary values for calculating covariance
  // covariance = sign( len(x)*sum(x*y) - sum(x)*sum(y) )
  // mSum = sum(x*y), hSum = sum(x), oSum = sum(y), len(x) == rows
  double hiddenOut = 0, mSum = 0, hSum = 0, oSum = 0;
  // While covariance is still increasing
  for (epoch = 0; epoch < 500; ++epoch)
    {
      // Evaluate all patterns
      for (p = 0; p < rows; ++p)
        {
          localError = 0;
          // Let all neurons evaluate
          for (i = 0; i < neuronConnections->size(); ++i)
            {
              neuronConnections->at(i).first->output(X + p*numOfInputs);
            }
          // Evaluate yourself
          hiddenOut = output(X + p*numOfInputs);
          // Calc covariance parts
          mSum += (hiddenOut * outputs[p]);
          hSum += hiddenOut;
          oSum += outputs[p];

          // Calc localerror
          // localerror = (E_op - avg(E_o))* f'_p
          localError = (patError[p] - error[0]) * outputDeriv();
          // Calc weight updates for all connections
          for (i = 0; i < inputConnections->size(); ++i)
            {
              inputIndex = inputConnections->at(i).first;
              (inputUpdates->begin()[i]) += (localError * X[inputIndex]);
            }
          for (i = 0; i < neuronConnections->size(); ++i)
            {
              (neuronUpdates->begin())[i] +=
                (localError * (neuronConnections->at(i).first->output()));
            }
        }
      // Calc sign of covariance
      // covariance = sign( len(x)*sum(x*y) - sum(x)*sum(y) )
      covariance = sign(rows * mSum - hSum*oSum);

      // Apply weight update
      applyWeightUpdates(covariance);
    }
}

void RCascadeNeuron::addLocalError(double error) {
  // Not used for this type
}
void RCascadeNeuron::calcLocalDerivative(double *inputs) {
  // Not used for this type
}

void RCascadeNeuron::applyWeightUpdates(int covariance) {
	unsigned int i;
	double prevUpdate, prevDeriv, weightUpdate, deriv;
	// Maximum and minimm weight changes
	double dMax = 50, dMin = 0.00001;
	// How much to adjust updates
	double dPos = 1.2, dNeg = 0.5;

	for (i = 0; i < neuronConnections->size(); i++) {
		prevUpdate = prevNeuronUpdates->at(i);
		prevDeriv = prevNeuronDerivs->at(i);
		deriv = covariance * neuronUpdates->at(i);

		if (prevDeriv * deriv > 0) {
			// We are on the right track, increase speed!
			weightUpdate = std::abs(prevUpdate) * dPos;
			// But not too fast!
			if (weightUpdate > dMax)
				weightUpdate = dMax;
			weightUpdate *= sign(deriv);
			neuronConnections->at(i).second += weightUpdate;
		} else if (prevDeriv * deriv < 0) {
			// Shit, we overshot the target!
			weightUpdate = std::abs(prevUpdate) * dNeg;
			if (weightUpdate < dMin)
				weightUpdate = dMin;
			weightUpdate *= sign(deriv);
			// Go back
			neuronConnections->at(i).second -= prevUpdate;
			// Next time forget about this disastrous direction
			deriv = 0;
		} else {
			// Previous round we overshot, go forward
			weightUpdate = std::abs(prevUpdate) * sign(deriv);
			neuronConnections->at(i).second += weightUpdate;
		}

		prevNeuronDerivs->begin()[i] = deriv;
		prevNeuronUpdates->begin()[i] = weightUpdate;
		neuronUpdates->begin()[i] = 0;
	}
	// Calculate weight updates to inputs also
	for (i = 0; i < inputConnections->size(); i++) {
		prevUpdate = prevInputUpdates->at(i);
		prevDeriv = prevInputDerivs->at(i);
		deriv = covariance * inputUpdates->at(i);

		if (prevDeriv * deriv > 0) {
			// We are on the right track, increase speed!
			weightUpdate = std::abs(prevUpdate) * dPos;
			// But not too fast!
			if (weightUpdate > dMax)
				weightUpdate = dMax;
			weightUpdate *= sign(deriv);
			inputConnections->at(i).second += weightUpdate;
		} else if (prevDeriv * deriv < 0) {
			// Shit, we overshot the target!
			weightUpdate = std::abs(prevUpdate) * dNeg;
			if (weightUpdate < dMin)
				weightUpdate = dMin;
			weightUpdate *= sign(deriv);
			// Go back
			inputConnections->at(i).second -= prevUpdate;
			// Next time forget about this disastrous direction
			deriv = 0;
		} else {
			// Previous round we overshot, go forward
			weightUpdate = std::abs(prevUpdate) * sign(deriv);
			inputConnections->at(i).second += weightUpdate;
		}

		prevInputDerivs->begin()[i] = deriv;
		prevInputUpdates->begin()[i] = weightUpdate;
		inputUpdates->begin()[i] = 0;
	}
}
