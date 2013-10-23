#ifndef MATRIXNETWORK_H_
#define MATRIXNETWORK_H_

#include "activationfunctions.hpp"

/*
 * The network is not connected upon construction. After
 * construction, the connection matrix is zero. As argument
 * the constructor takes the number of neurons to include
 * in the matrix. A convenience constructor exists where
 * number of inputs, hidden and outputs are separated.
 * Note that a bias unit is always ADDED to the number
 * of neurons you specify. E.g. specifying a network
 * with 4 inputs, and 2 hidden and 1 output will result
 * in a network with (4 + 2 + 1 +1) = 8 neurons,
 * meaning matrices (arrays) of 8*8 elements. The bias
 * is always placed directly AFTER THE INPUTS.
 *
 * In addition to weights and connections, the activation
 * functions of neurons are also stored in a matrix.
 * While the available functions set the values for all
 * hidden and output units respectively, there is no reason
 * a genetic algorithm couldn't evolve the activation functions
 * over time.
 */
class MatrixNetwork {
 protected:
  // Log to keep over training performance
  // up to training algorithm to make use of it
  double *aLogPerf;
  unsigned int logPerfLength;

  void initLog(const unsigned int length);

 public:
  const unsigned int INPUT_START;
  const unsigned int INPUT_END;
  const unsigned int BIAS_INDEX;
  const unsigned int BIAS_START;
  const unsigned int BIAS_END;
  const unsigned int HIDDEN_START;
  const unsigned int HIDDEN_END;
  const unsigned int OUTPUT_START;
  const unsigned int OUTPUT_END;
  const unsigned int LENGTH;

  const unsigned int INPUT_COUNT;
  const unsigned int HIDDEN_COUNT;
  const unsigned int OUTPUT_COUNT;

  // Network structure
  ActivationFuncEnum * const actFuncs;
  unsigned int * const conns;
  double * const weights;

  // Outputs are stored here
  double * const outputs;

  MatrixNetwork(unsigned int numOfInput, unsigned int numOfHidden,
                unsigned int numOfOutput);
  virtual ~MatrixNetwork();

  // Null if not trained
  double *getLogPerf();
  unsigned int getLogPerfLength();

  //  double *getWeights();
  //  unsigned int *getConns();
  //  ActivationFuncEnum *getActFuncs();

  /**
   * Returns the pointer given as output, so pay no attention to return object
   * if not wanted.
   */
  virtual double *output(const double * const inputs,
                         double * const output);


  // Please note that these are convenience methods and
  // the getters return values from ONE neuron each, and
  // might not be representative!

  /**
   * Sets the activation function of the output layer
   */
  void setOutputActivationFunction(ActivationFuncEnum func);
  ActivationFuncEnum getOutputActivationFunction();

  /**
   * Sets the activation function of the hidden layer
   */
  void setHiddenActivationFunction(ActivationFuncEnum func);
  ActivationFuncEnum getHiddenActivationFunction();

};

#endif //MATRIXNETWORK_H_
