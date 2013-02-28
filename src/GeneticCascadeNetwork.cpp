/*
Trains hidden neurons with RProp, while output neuron is trained genetically.
*/

#include "GeneticCascadeNetwork.h"
#include "CascadeNetwork.h"
#include "RPropNetwork.h" // for bias neuron
#include "FFNeuron.h"
#include "activationfunctions.h"
#include <vector>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <time.h>
#include "boost/random.hpp"
#include "c_index.h"
using namespace std;

GeneticCascadeNetwork::GeneticCascadeNetwork(unsigned int numOfInputs) :
  CascadeNetwork(numOfInputs, 1)
{

}

GeneticCascadeNetwork::~GeneticCascadeNetwork() {
  // Delete subclass specific allocations here
}

void GeneticCascadeNetwork::initNodes() {
  hiddenRCascadeNeurons = new vector<RCascadeNeuron*>;

  this->hiddenNeurons = new Neuron*[0];
  unsigned int i;

  this->outputNeurons = new Neuron*[this->numOfOutput];
  for (i = 0; i < this->numOfOutput; i++) {
    this->outputNeurons[i] = new GeneticNeuron(i);
  }

  this->bias = new RPropBias;
}

void GeneticCascadeNetwork::trainOutputs(double *X, double *Y, unsigned int rows) {
  unsigned int i;
  for (i = 0; i < numOfOutput; i++)
    ((GeneticNeuron*) outputNeurons[i])->learn(X, Y, rows);
}

void GeneticCascadeNetwork::calcErrors(double *X, double *Y, unsigned int rows,
                                double *patError, double *error,
                                double *outputs) {
  // Zero error array first
  memset(error, 0, numOfOutput * sizeof(double));

  for (unsigned int i = 0; i < rows; i++)
    {
      // Save outputs for covariance calculation
      output(X + i*numOfInputs, outputs + i*numOfOutput);
    }

  // Calc C-index
  // There's only one output neuron
  error[0] = getPatError(outputs, Y, rows, patError);
}


/*
 * -----------------------
 * GeneticNeuron definitions
 * -----------------------
 */

GeneticNeuron::GeneticNeuron(int id) : Neuron(id),
                                       rng(boost::mt19937(time(NULL))),
                                       dist_normal(boost::normal_distribution<double>(0,1)),
                                       gaussian(boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(rng, dist_normal)),
                                       uniform(boost::variate_generator<boost::mt19937&, boost::uniform_int<> >(rng, dist_uniform))
{
  setup();
}

GeneticNeuron::GeneticNeuron(int id, double (*activationFunction)(double),
                               double (*activationDerivative)(double)) :
  Neuron(id, activationFunction, activationDerivative),
  rng(boost::mt19937(time(NULL))),
  dist_normal(boost::normal_distribution<double>(0,1)),
  gaussian(boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >(rng, dist_normal)),
  uniform(boost::variate_generator<boost::mt19937&, boost::uniform_int<> >(rng, dist_uniform))
{
  setup();
}

void GeneticNeuron::setup() {
  populationSize = 30;
  generations = 20;
  weightMutationChance = 0.8;
  weightMutationStdDev = 0.3;
  weightMutationHalfPoint = 0;
  /*
  // Random stuff
   boost::mt19937 eng; // a core engine class
  eng.seed(time(NULL));

  // Geometric distribution for selecting parents
  //boost::geometric_distribution<int, double> geo_dist(0.95);
  //geometric = boost::variate_generator<boost::mt19937&,
  //                                     boost::geometric_distribution<int,
  //                                                                   double> >(eng, geo_dist);


  // Normal distribution for weight mutation, 0 mean and 1 stddev
  // We can then get any normal distribution with y = mean + stddev * x
  boost::normal_distribution<double> gauss_dist(0, 1);
  gaussian = new boost::variate_generator<boost::mt19937&,
                                      boost::normal_distribution<double> >(eng, gauss_dist);

  // Uniform distribution 0 to 1 (inclusive)
  boost::uniform_int<> uni_dist(0, 1);
  uniform = boost::variate_generator<boost::mt19937&,
                                     boost::uniform_int<> >(eng, uni_dist);
  */
}

GeneticNeuron::~GeneticNeuron() {
}

/*
    Returns a joint vector of weights. Ordered as:
    Input weights
    Neuron weights

    Clears the vector before filling it
  */
void GeneticNeuron::getAllWeights(vector<double> *weights) {
  // Clear first
  weights->clear();

  // Add input weights
  unsigned int i;
  for (i = 0; i < inputConnections->size(); i++) {
    weights->push_back(inputConnections->at(i).second);
  }

  // Add neuron weights
  for (i = 0; i < neuronConnections->size(); i++) {
    weights->push_back(neuronConnections->at(i).second);
  }
}

/*
  Sets the weights in the connections. Expects the same order as in
  getAllWeights
*/
void GeneticNeuron::setAllWeights(vector<double> *weights) {
  unsigned int i, j;
  for (i = 0; i < weights->size(); i++) {
    // Add input weights
    if (i < inputConnections->size())
      inputConnections->at(i).second = weights->at(i);
    else {
      printf("Setting neuron weights\n");
      // Add neuron weights
      j = i - inputConnections->size();
      neuronConnections->at(j).second = weights->at(i);
    }
  }
}

/*
  mutateVector(weights, &gaussian, &uniform, mutationChance, stdDev)
 */
void mutateVector(vector<double> *weights,
                  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >* gaussian,
                  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > *uniform,
                  double mutationChance, double stdDev) {
  unsigned int i;
  for (i = 0; i < weights->size(); i++) {
    if ((*uniform)() <= mutationChance)
      weights->at(i) += (*gaussian)() * stdDev;
  }
}

double GeneticNeuron::outputWithVector(vector<double> *weights,
                                       double *X) {
  double inputSum = 0;
  unsigned int i, j;
  // First evalute the connected hidden neurons
  // This relies on the fact that the neuron is connected to all
  // hidden neurons, and that they are connected in the order that
  // they themselves connect to each other.
  for (i = 0; i < neuronConnections->size(); i++) {
    neuronConnections->at(i).first->output(X);
  }

  // Now evaluate this using the weights vector
  for (i = 0; i < weights->size(); i++) {
    if (i < inputConnections->size()) {
      // Add inputs to inputSum
      inputSum += X[i] * weights->at(i);
	}
    else {
      j = i - inputConnections->size();
      // Add neuron outputs to inputSum
      inputSum += weights->at(i) * neuronConnections->at(j).first->output();
    }
  }

  //printf("output using inputsum: %f and result: %f\n", inputSum, activationFunction(inputSum));
  // return activationFunction result
  return activationFunction(inputSum);
}

double GeneticNeuron::evaluateWithVector(vector<double> *weights,
                                         double *X, double *Y,
                                         unsigned int length,
                                         double *outputs) {
  // Evaluate each input set
  for (unsigned int i = 0; i < length; i++) {
    // Place output in correct position here
    outputs[i] = outputWithVector(weights, X + i*inputConnections->size());
  }
  // Now calculate c-index
  double ci = get_C_index(outputs, Y, length);

  printf("ci = %f\n", ci);

  // Return the inverse since this returns the error of the network
  // If less than 0.001, return 1000 instead to avoid dividing by zero
  if (ci < 0.0000001)
    return 1;
  else
    return 1 - ci;
}

void insertSorted(vector<vector<double>*> * const sortedPopulation,
                  vector<double> * const sortedErrors, const double error,
                  vector<double> * const vec) {
  vector<vector<double>*>::iterator netIt;
  vector<double>::iterator errorIt;
  bool inserted = false;
  unsigned int j;

  netIt = sortedPopulation->begin();
  errorIt = sortedErrors->begin();
  // Insert in sorted position
  for (j = 0; j < sortedPopulation->size(); j++) {
    if (error < errorIt[j]) {
      //printf("Inserting at %d, error = %f\n", j, error);
      sortedPopulation->insert(netIt + j, vec);
      sortedErrors->insert(errorIt + j, error);
      inserted = true;
      break;
    }
  }
  // If empty, or should be placed last in list
  if (!inserted) {
    //printf("Inserting last, error = %f\n", error);
    sortedPopulation->push_back(vec);
    sortedErrors->push_back(error);
    inserted = true;
  }
}



/*
  Skipping cross over
 */
void GeneticNeuron::learn(double *X, double *Y,
                                   unsigned int length) {
  printf("Neuron Learn\n");
  // Create necessary vectors
  // population
  vector<vector<double>*> sortedPopulation;
  vector<double> sortedErrors;
  // Have one throw away used to create next child
  sortedPopulation.reserve(populationSize + 1);
  sortedErrors.reserve(populationSize + 1);

  double error;
  double *outputs = new double[length];
  unsigned int i;

  // Create population
  // Sort on performance
  printf("Length: %d\n", length);

  for (i = 0; i < populationSize + 1; i++) {
    // Create a vector
    vector<double> *vec = new vector<double>;
    getAllWeights(vec);
    mutateVector(vec, &gaussian, &uniform,
                 weightMutationChance, weightMutationStdDev);
    error = evaluateWithVector(vec, X, Y, length, outputs);
    insertSorted(&sortedPopulation, &sortedErrors, error, vec);
  }

  // Save best vector
  vector<double> *best = sortedPopulation.front();
  vector<double> *child;

  //vector<int>::iterator it;
  // Loop over generations
  unsigned int curGen, genChild;
  for (curGen = 0; curGen < generations; curGen++) {
    // For child in range(populationSize)
    for (genChild = 0; genChild < populationSize; genChild++) {
      printf("gen: %d, child: %d\n", curGen, genChild);
      // Recycle worst vector
      child = sortedPopulation.back();
      // Remove it from the list
      sortedPopulation.pop_back();
      sortedErrors.pop_back();
      // Copy best vector
      //it = best->begin();
      child->assign(best->begin(), best->end());
      // Create a new mutation
      mutateVector(child, &gaussian, &uniform,
                 weightMutationChance, weightMutationStdDev);
      // Evaluate and Insert sorted
      error = evaluateWithVector(child, X, Y, length, outputs);
      //printf("new child error: %f\n", error);
      insertSorted(&sortedPopulation, &sortedErrors, error, child);

      // Update best
      best = sortedPopulation.front();
      printf("best error: %f\n", sortedErrors.front());
    }
    // End for
  }
  // End loop

  // Set neuron weights to best vector
  setAllWeights(best);

  // Destroy local stuff
  delete outputs;
  // Loop over population and delete each vector
  while (0 < sortedPopulation.size()) {
    delete sortedPopulation.back();
    sortedPopulation.pop_back();
  }
}


/*
------------------
GeneticLadder definitions
------------------
*/

GeneticLadderNetwork::GeneticLadderNetwork(unsigned int numOfInputs):
  GeneticCascadeNetwork(numOfInputs) {

}

GeneticLadderNetwork::~GeneticLadderNetwork() {
  while (hiddenGeneticNeurons->size() > 0) {
    delete hiddenGeneticNeurons->back();
    hiddenGeneticNeurons->pop_back();
  }
  delete hiddenGeneticNeurons;
}

void GeneticLadderNetwork::initNodes() {
  hiddenRCascadeNeurons = new vector<RCascadeNeuron*>;
  hiddenGeneticNeurons = new vector<GeneticNeuron*>;

  this->hiddenNeurons = new Neuron*[0];
  unsigned int i;

  this->outputNeurons = new Neuron*[this->numOfOutput];
  for (i = 0; i < this->numOfOutput; i++) {
    this->outputNeurons[i] = new GeneticNeuron(i);
  }

  this->bias = new RPropBias;
}

void GeneticLadderNetwork::learn(double *X, double *Y, unsigned int rows) {

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

  unsigned int i;
  // Step 1, train the output neuron (has its own virtual error function)
  trainOutputs(X, Y, rows);

  // Calculate C-index and p-specific C-sums (must be virtual function)
  // Also saves outputs
  calcErrors(X, Y, rows, patError, error, outputs);
  printf("Error so far: %f\n", error[0]);
  printf("maxError: %f\n", maxError);
  // While some condition
  unsigned int neuronCount = 0;
  while (neuronCount < maxHidden && error[0] > maxError) {
    printf("Moving output to hidden layer\n");
    // Set id
    outputNeurons[0]->setId((int) neuronCount);
    // Move from outputs to hiddenGeneticNeurons
    hiddenGeneticNeurons->push_back(static_cast<GeneticNeuron*>(outputNeurons[0]));

    printf("Creating new output\n");
    // Create new output
    outputNeurons[0] = new GeneticNeuron(0);
    setOutputActivationFunction(outputActivationFunction);
    // Connect it to previous layers
    outputNeurons[0]->connectToNeuron(bias,
                               (uniform() - 0.5));
    for (i = 0; i < numOfInputs; i++) {
      // random weight
      outputNeurons[0]->connectToInput(i,
                                (uniform() - 0.5));
    }
    for (i = 0; i < hiddenGeneticNeurons->size(); i++) {
      outputNeurons[0]->connectToNeuron(hiddenGeneticNeurons->at(i),
                                 (uniform() - 0.5));
    }
    // Train output
    trainOutputs(X, Y, rows);

    // Error calculation
    // Calculate C-index and p-specific C-sums (must be virtual function)
    // Also saves outputs
    calcErrors(X, Y, rows, patError, error, outputs);
    printf("Error so far: %f\n", error[0]);

    // Remember to increment
    neuronCount++;
  }
  // END While
  delete[] outputs;
  delete[] patError;
  delete[] error;
}

void GeneticLadderNetwork::calcErrors(double *X, double *Y, unsigned int rows,
                                double *patError, double *error,
                                double *outputs) {
  // Zero error array first
  memset(error, 0, numOfOutput * sizeof(double));
  // Dont use paterror here
  // memset(patError, 0, numOfOutput * sizeof(double));

  for (unsigned int i = 0; i < rows; i++)
    {
      output(X + i*numOfInputs, outputs + i*numOfOutput);
    }

  // Calc C-index
  // There's only one output neuron
  double ci = get_C_index(outputs, Y, rows);
  error[0] = 1.0 - ci;
}


unsigned int GeneticLadderNetwork::getNumOfHidden() const {
  unsigned int retval = 0;
  if (this->hiddenGeneticNeurons != NULL) {
    retval = this->hiddenGeneticNeurons->size();
  }
  return retval;
}

Neuron* GeneticLadderNetwork::getHiddenNeuron(unsigned int id) const {
  return hiddenGeneticNeurons->at(id);
}

bool GeneticLadderNetwork::getNeuronWeightFromHidden(unsigned int fromId, int toId, double *weight) {
  if (fromId >= getNumOfHidden() ||
      (toId > -1 && (unsigned int) toId >= getNumOfHidden())) {
    throw invalid_argument("Id was larger than number of nodes");
  }

  return hiddenGeneticNeurons->at(fromId)->getNeuronWeight(toId, weight);
}

bool GeneticLadderNetwork::getInputWeightFromHidden(unsigned int fromId, unsigned int toIndex, double *weight) {
  if (fromId >= getNumOfHidden() || toIndex >= numOfInputs) {
    throw invalid_argument("Id was larger than number of nodes or \
index was greater than number of inputs");
  }

  return hiddenGeneticNeurons->at(fromId)->getInputWeight(toIndex, weight);
}
