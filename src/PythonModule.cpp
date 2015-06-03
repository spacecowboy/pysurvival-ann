/*
 * PythonModule.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "Python.h"
#include "PythonModule.h"
#include "structmember.h" // used to declare member list
// Do not want any deprecation warnings
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "ModuleHeader.h" // Must include this before arrayobject
#include <numpy/arrayobject.h> // Numpy seen from C
#include "RPropNetworkWrapper.hpp"
#include "activationfunctions.hpp"
#include "CIndexWrapper.h"
#include "MatrixNetworkWrapper.hpp"
//#include "CoxCascadeNetworkWrapper.h"
//#include "GeneticCascadeNetworkWrapper.h"
#include "GeneticNetwork.hpp"
#include "GeneticNetworkWrapper.hpp"
#include "ErrorFunctionsWrapper.hpp"
#include "ErrorFunctions.hpp"

// Matrix network
// ==============


// Public Python methods
static PyMethodDef MatrixNetworkMethods[] =
{
  {"output", (PyCFunction) MatrixNetwork_output, METH_O,
     "net.output(X)\n\n\
Computes the network's output, given the input vector X.\n\n\
Parameters\n\
----------\n\
X : Sequence of floats or ints\n\
    Input values to feed to the network. Note that the list/array must\n\
    have a length equal to the number of input neurons in the network.\n\
\nReturns\n\
-------\n\
out : ndarray\n\
    Array of output values. The size of this array is equal to the number\n\
    of output neurons in the array.\n\
\nExamples\n\
--------\n\
>>> net = matrixnetwork(2, 0, 1)\n\
>>> net.weights = net.weights + 1\n\
>>> net.connections = net.connections + 1\n\
>>> net.output([1, -1])\n\
array([ 0.73105858])\n"},

  {NULL}, // So that we can iterate safely below
};


//Public Python members
static PyMemberDef MatrixNetworkMembers[] = {
  {NULL} // for safe iteration
};


//Public Python members with get/setters
static PyGetSetDef MatrixNetworkGetSetters[] = {
  {(char*)"input_count", (getter)MatrixNetwork_getNumOfInputs, NULL,    \
   (char*)"Number of input neurons", NULL},
  {(char*)"hidden_count", (getter)MatrixNetwork_getNumOfHidden, NULL,    \
   (char*)"Number of hidden neurons", NULL},
  {(char*)"output_count", (getter)MatrixNetwork_getNumOfOutput, NULL,  \
   (char*)"Number of output neurons", NULL},

  {(char*)"log", (getter)MatrixNetwork_getLogPerf, NULL,       \
   (char*)"Get a log of the training performance, [epochs, outputs]", NULL},
  /*
  {(char*)"output_activationunction",               \
   (getter)MatrixNetwork_getOutputActivationFunction,   \
   (setter)MatrixNetwork_setOutputActivationFunction,        \
   (char*)"The activation function used by output neurons. \
For example network.LOGSIG.", NULL},
  {(char*)"hiddenActivationFunction",               \
   (getter)MatrixNetwork_getHiddenActivationFunction,   \
   (setter)MatrixNetwork_setHiddenActivationFunction,        \
   (char*)"The activation function used by hidden neurons. \
For example network.TANH", NULL},
  */
  {(char*)"weights",               \
   (getter)MatrixNetwork_getWeights,   \
   (setter)MatrixNetwork_setWeights,        \
   (char*)"All weights of networks as a LxL matrix. Ordered by \
[neuron * neuron_count + target_weight]", NULL},

  {(char*)"connections",                   \
   (getter)MatrixNetwork_getConns,                           \
   (setter)MatrixNetwork_setConns,        \
   (char*)"Connections of neurons, LxL matrix.", NULL},

  {(char*)"activation_functions",                   \
   (getter)MatrixNetwork_getActFuncs,   \
   (setter)MatrixNetwork_setActFuncs,        \
   (char*)"The activation functions of the neurons.", NULL},

  {(char*)"dropout_input_probability",       \
   (getter)MatrixNetwork_getInputDropoutProb,   \
   (setter)MatrixNetwork_setInputDropoutProb,        \
   (char*)"Dropout probability for input neurons. Set negative to disable.", NULL},

  {(char*)"dropout_hidden_probability",          \
   (getter)MatrixNetwork_getHiddenDropoutProb,   \
   (setter)MatrixNetwork_setHiddenDropoutProb,        \
   (char*)"Dropout probability for hidden neurons. Set negative to disable.", NULL},

    {NULL} // Sentinel
};


// Python type declaration
static PyTypeObject MatrixNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "_ann.matrixnetwork",		// tp_name VITAL PACKAGE NAME FOR PICKLING!
  sizeof(PyMatrixNetwork),					// tp_basicsize
  0,						// tp_itemsize
  (destructor)MatrixNetwork_dealloc,			// tp_dealloc
  0,						//* tp_print
  0,						//* tp_getattr
  0,						//* tp_setattr
  0,						//* tp_compare
  0,						//* tp_repr
  0,						//* tp_as_number
  0,						//* tp_as_sequence
  0,						//* tp_as_mapping
  0,						//* tp_hash
  0,						//* tp_call
  0,						//* tp_str
  0,						//* tp_getattro
  0,						//* tp_setattro
  0,						//* tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 	//* tp_flags
  "matrixnetwork(inputcount, hiddencount, outputcount)\n\n\
A feed forward neural network. This class serves as a base class for all\n\
other types of networks.\n\n\
Parameters\n\
----------\n\
inputcount : integer\n\
    The number of input neurons.\n\
hiddencount : integer\n\
    The number of hidden neurons. Note that this is the total number of\n\
    neurons in all hidden layers.\n\
outputcount : integer\n\
    The number of output neurons.\n\n\
Returns\n\
-------\n\
out : matrixnetwork\n\
    A matrix network with the specified number of neurons.\n\n\
See Also\n\
--------\n\
rpropnetwork, geneticnetwork\n\n\
Examples\n\
--------\n\
>>> matrixnetwork(2, 3, 1)\n\
matrixnetwork(2, 3, 1)\n",			//* tp_doc
  0,						//* tp_traverse
  0,			 			//* tp_clear
  0,			 			//* tp_richcompare
  0,			 			//* tp_weaklistoffset
  0,			 			//* tp_iter
  0,			 			//* tp_iternext
  MatrixNetworkMethods,					//* tp_methods
  MatrixNetworkMembers,					//* tp_members
  MatrixNetworkGetSetters,			 			//* tp_getset
  0,			 			//* tp_base
  0,			 			//* tp_dict
  0,			 			//* tp_descr_get
  0,			 			//* tp_descr_set
  0,			 			//* tp_dictoffset
  (initproc)MatrixNetwork_init,				//* tp_init
  0,			 			//* tp_alloc
  MatrixNetwork_new,			 		//* tp_new
};



/*
 * RPropNetwork
 * ============
 */

/*
 * Public Python methods
 * ---------------------
 */
static PyMethodDef RPropNetworkMethods[] =
{
    {"learn", (PyCFunction) RPropNetwork_learn,                         \
     METH_VARARGS | METH_KEYWORDS,
     "net.learn(X, Y)\n\n\
Trains the network using iRProp+ as described in:\n\
'Improving the Rprop Learning Algorithm' by\n\
Christian Igel and Michael Hüsken.\n\n\
The training parameters are set directly as properties on the network.\n\
\n\
Parameters\n\
----------\n\
X : 2-dimensional ndarray of floats\n\
    Input values to feed to the network. The second dimension (columns) must\n\
    be equal to the number of input neurons in the network. The first\n\
    dimension (rows) must equal the number of rows in Y.\n\
Y : 2-dimensional ndarray of floats\n\
    Target values to train against. The second dimension (columns) must\n\
    be equal to the number of output neurons in the network. The first\n\
    dimension (rows) must equal the number of rows in X.\n\
\n\
Returns\n\
-------\n\
Nothing\n\
\n\
Examples\n\
--------\n\
>>> X = array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\n\
>>> Y = array([[0], [1], [1], [0]], dtype=float)\n\
>>> net = rpropnetwork(X.shape[1], 5, Y.shape[1])\n\
>>> net.connections = net.connections + 1\n\
>>> net.learn(X, Y)\n\
>>> [net.output(x) for x in X]\n\
[array([ 0.]),\n\
 array([ 0.98790896]),\n\
 array([ 0.98790896]),\n\
 array([ 0.02218867])]\n"},
    {NULL}, // So that we can iterate safely below
};


/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef RPropNetworkGetSetters[] = {
  {(char*)"max_error",
   (getter)RPropNetwork_getMaxError,                            \
   (setter)RPropNetwork_setMaxError,                            \
   (char*)"Maximum error allowed for early stopping", NULL},

  {(char*)"min_errorfrac", \
   (getter)RPropNetwork_getMinErrorFrac, \
   (setter)RPropNetwork_setMinErrorFrac, \
   (char*)"Minimum relative change in error over 100 epochs to allow before stopping", \
   NULL},

  {(char*)"max_epochs",
   (getter)RPropNetwork_getMaxEpochs,                               \
   (setter)RPropNetwork_setMaxEpochs,                               \
   (char*)"Maximum number of epochs allowed for training", NULL},

  {(char*)"error_function",
   (getter)RPropNetwork_getErrorFunction,                               \
   (setter)RPropNetwork_setErrorFunction,                                  \
   (char*)"Error function to use during training. Default is Mean Square Error",
   NULL},

  {NULL} // Sentinel
};



/*
 * Python type declaration
 * -----------------------
 */
static PyTypeObject RPropNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_ann.rpropnetwork",  // tp_name VITAL PACKAGE NAME FOR PICKLING!
    sizeof(PyRPropNetwork),                   /* tp_basicsize */
    0,                                              /* tp_itemsize */
    0,                  /* tp_dealloc */
    0,                                              /* tp_print */
    0,                                              /* tp_getattr */
    0,                                              /* tp_setattr */
    0,                                              /* tp_compare */
    0,                                              /* tp_repr */
    0,                                              /* tp_as_number */
    0,                                              /* tp_as_sequence */
    0,                                              /* tp_as_mapping */
    0,                                              /* tp_hash */
    0,                                              /* tp_call */
    0,                                              /* tp_str */
    0,                                              /* tp_getattro */
    0,                                              /* tp_setattro */
    0,                                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,       /* tp_flags*/
    "rpropnetwork(inputcount, hiddencount, outputcount)\n\n\
A feed forward neural network which can be trained with iRProp+ as described in:\n\
'Improving the Rprop Learning Algorithm' by \n\
Christian Igel and Michael Hüsken.\n\
\n\
Parameters\n\
----------\n\
inputcount : integer\n\
    The number of input neurons.\n\
hiddencount : integer\n\
    The number of hidden neurons. Note that this is the total number of\n\
    neurons in all hidden layers.\n\
outputcount : integer\n\
    The number of output neurons.\n\
\n\
Returns\n\
-------\n\
out : rpropnetwork\n\
    An Rprop matrix network with the specified number of neurons.\n\
\n\
The training parameters are set as properties directly on the network:\n\
\n\
Properties\n\
----------\n\
net.max_error : float\n\
    Maximum error allowed for early stopping. Default 0.001\n\
net.min_errorfrac : float\n\
    Minimum relative change in error over 100 epochs required to prevent\n\
    early stopping. Default 0.01.\n\
net.max_epochs : integer\n\
    Maximum number of epochs to train for. Default 1000.\n\
net.error_function : error function identifier (integer)\n\
    Error function to us during training. Default ann.ERROR_MSE.\n\
\n\
See Also\n\
--------\n\
geneticnetwork\n\
\n\
Examples\n\
--------\n\
>>> rpropnetwork(2, 3, 1)\n\
rpropnetwork(2, 3, 1)\n", // tp_doc
    0,                                              /* tp_traverse */
    0,                                              /* tp_clear */
    0,                                              /* tp_richcompare */
    0,                                              /* tp_weaklistoffset */
    0,                                              /* tp_iter */
    0,                                              /* tp_iternext */
    RPropNetworkMethods,                                       /* tp_methods */
    0,                                       /* tp_members */
    RPropNetworkGetSetters,                       /* tp_getset */
    0,                                              /* tp_base */
    0,                                              /* tp_dict */
    0,                                              /* tp_descr_get */
    0,                                              /* tp_descr_set */
    0,                                              /* tp_dictoffset */
    (initproc)RPropNetwork_init,                               /* tp_init */
    0,                                              /* tp_alloc */
    0,                                  /* tp_new */
};



/*
 * Genetic network
 * ========================
 */


// Public Python methods
static PyMethodDef GenNetworkMethods[] =
{
    {"learn", (PyCFunction) GenNetwork_learn,                           \
     METH_VARARGS | METH_KEYWORDS, \
     "net.learn(X, Y)\n\n\
Trains the network using a genetic algorithm.\n\
\n\
Parameters\n\
----------\n\
X : 2-dimensional ndarray of floats\n\
    Input values to feed to the network. The second dimension (columns) must\n\
    be equal to the number of input neurons in the network. The first\n\
    dimension (rows) must equal the number of rows in Y.\n\
Y : 2-dimensional ndarray of floats\n\
    Target values to train against. The second dimension (columns) must\n\
    be equal to the number of output neurons in the network, or the number\n\
    specified by the current fitness function (survival functions can\n\
    require 2 columns, while only having 1 output neuron). The first\n\
    dimension (rows) must equal the number of rows in X.\n\
\n\
Returns\n\
-------\n\
Nothing\n\
\n\
Examples\n\
--------\n\
>>> X = array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)\n\
>>> Y = array([[0], [1], [1], [0]], dtype=float)\n\
>>> net = geneticnetwork(X.shape[1], 5, Y.shape[1])\n\
>>> net.connections = net.connections + 1\n\
>>> net.learn(X, Y)\n\
>>> [net.output(x) for x in X]\n\
[array([  3.39518590e-37]),\n\
 array([ 1.]),\n\
 array([ 1.]),\n\
 array([  5.21302310e-38])]\n\
\n\
The training parameters are set as properties directly on the network, \n\
where some properties take integer constants defined on this class.\n\
\n\
Properties\n\
----------\n\
net.selection_method : method id (integer)\n\
    The way in which networks are chosen as parents for\n\
    crossover. Possible values are:\n\
    - Geometric picks the parents from a geometric distribution\n\
      which benefits the fittest network the most.\n\
    - Roulette gives each network a probability proportional to their score\n\
      divided by the sum of all scores.\n\
    - Tournament picks, for each parent, two networks and selects the most\n\
      fit. This is the least elitist, and least predictable.\n\
    Default is SELECTION_GEOMETRIC.\n\
net.crossover_method : method id (integer)\n\
    The crossover operation can be done in several ways. Given two parents\n\
    with genomes XXXXX and YYYYY for example:\n\
    - Uniform, assembles a child by selecting neurons and associated\n\
      weights at random from each parent with equal probability.\n\
      Example children: XYXYX, YXYXY.\n\
    - Two point, selects two places along the genome and exchanges everything\n\
      in-between for the other parent's genetic material. This produces two\n\
      children, with 'opposite' genomes.\n\
      Example children: XYYXX, YXXYY.\n\
    - One point, selects a single pivot point along the genome and exchanges\n\
      everything before/after the point for the other parent's genome.\n\
      Produces two children, with 'opposite' genomes.\n\
       Example children: XXYYY, YYXXX.\n\
    Default is CROSSOVER_ONEPOINT.\n\
net.fitness_function : function id (integer)\n\
    The function that will judge the performance of the networks.\n\
    Default is FITNESS_MSE.\n\
net.generations : integer\n\
    Number of generations to train for, where the number of 'iterations'\n\
    or networks 'born' in each generation is equal to population_size.\n\
    Default 100.\n\
net.population_size : integer\n\
    Size of population to create. This indirectly makes generations longer.\n\
    Default 50.\n\
net.crossover_chance : float\n\
    Probability to perform cross-over between parents. Default 1.0.\n\
net.connection_mutation_chance : float\n\
    Probability to flip each bit in the connection matrix. Default 0.\n\
net.activation_mutation_chance : float\n\
    Probability for each neuron to switch activation function. Default 0.\n\
net.weight_mutation_chance : float\n\
    Probability of a single weight being selected for mutation. Default 0.15.\n\
net.weight_mutation_factor: float\n\
    Standard deviation of gaussian distribution used for mutation. A random\n\
    number R is picked from this distribution and then weight w = w + R.\n\
    Default 1.0.\n\
net.weight_mutation_halfpoint : integer\n\
    Generation at which the weight_mutation_factor should have decreased to\n\
    half of its original value. Default 0 (off).\n\
net.mingroup : integer\n\
    Used by some fitness functions (SURV_KAPLAN_*) to determine minimum\n\
    valid group size. Default 1.\n"},
    {"getPredictionFitness", (PyCFunction) GenNetwork_getPredictionFitness,
     METH_VARARGS | METH_KEYWORDS, \
     "getPredictionFitness(X, Y)\n\n\
Returns the fitness of the network based on the predictions of the data,\n\
and the currently specified fitness function.\n\
\n\
Parameters\n\
----------\n\
X : 2-dimensional ndarray of floats\n\
    Input values to feed to the network. The second dimension (columns) must\n\
    be equal to the number of input neurons in the network. The first\n\
    dimension (rows) must equal the number of rows in Y.\n\
Y : 2-dimensional ndarray of floats\n\
    Target values to train against. The second dimension (columns) must\n\
    be equal to the number of output neurons in the network, or the number\n\
    specified by the current fitness function (survival functions can\n\
    require 2 columns, while only having 1 output neuron). The first\n\
    dimension (rows) must equal the number of rows in X.\n\
\n\
Returns\n\
-------\n\
fitness : float\n\
    The current fitness of the network, judged on the given data.\n\
\n\
Examples\n\
--------\n\
>>> net.getPredictionFitness(X, Y)\n\
-1.4748804266032915e-74\n"},
    {NULL}, // So that we can iterate safely below
};



//Public Python members with get/setters
static PyGetSetDef GenNetworkGetSetters[] = {
  {(char*)"generations", (getter)GenNetwork_getGenerations, \
   (setter)GenNetwork_setGenerations,                       \
   (char*)"Time to train", NULL},
  {(char*)"population_size", (getter)GenNetwork_getPopulationSize,   \
   (setter)GenNetwork_setPopulationSize,                            \
   (char*)"Number of networks created each generation", NULL},
  {(char*)"weight_mutation_chance", (getter)GenNetwork_getWeightMutationChance, \
   (setter)GenNetwork_setWeightMutationChance,                          \
   (char*)"The chance of a single weight being changed during cloning", NULL},
  {(char*)"weight_mutation_halfpoint",              \
   (getter)GenNetwork_getWeightMutationHalfPoint,                   \
   (setter)GenNetwork_setWeightMutationHalfPoint,                       \
   (char*)"If time dependant mutation is desired, set this to a non-zero value.\
 StdDev will decrease linearly and reach half at specified generation.", NULL},
  {(char*)"weight_mutation_factor",              \
   (getter)GenNetwork_getWeightMutationFactor, \
   (setter)GenNetwork_setWeightMutationFactor,                      \
   (char*)"Mutations are gaussians with this stddev and added to current\
 weight.", NULL},

  {(char*)"weight_decayL1",                  \
   (getter)GenNetwork_getDecayL1, \
   (setter)GenNetwork_setDecayL1,                      \
   (char*)"Coefficient for L1 weight decay. Zero by default.", NULL},
  {(char*)"weight_decayL2",                  \
   (getter)GenNetwork_getDecayL2, \
   (setter)GenNetwork_setDecayL2,                      \
   (char*)"Coefficient for L2 weight decay. Zero by default.", NULL},
  {(char*)"weight_elimination",                  \
   (getter)GenNetwork_getWeightElimination, \
   (setter)GenNetwork_setWeightElimination,                      \
   (char*)"Coefficient (g) for soft weight elimination: P = g * sum(). \
Zero by default.", NULL},
  {(char*)"weight_eliminationLambda",                  \
   (getter)GenNetwork_getWeightEliminationLambda, \
   (setter)GenNetwork_setWeightEliminationLambda,                      \
   (char*)"Coefficient (l) for soft weight elimination: \
P = sum( w^2 / [l^2 + w^2] ). Zero by default.", NULL},
  {(char*)"taroneware_statistic",                  \
   (getter)GenNetwork_getTaroneWareStatistic, \
   (setter)GenNetwork_setTaroneWareStatistic,                      \
   (char*)"What weighting to use with TaroneWare fitness function. \
LogRank (weights=1) used by default.", NULL},
  {(char*)"mingroup",                  \
   (getter)GenNetwork_getMinGroup, \
   (setter)GenNetwork_setMinGroup,                      \
   (char*)"Minimum group size. Default is 1.", NULL},

  /*
  {(char*)"resume",              \
   (getter)GenNetwork_getResume, \
   (setter)GenNetwork_setResume,                      \
   (char*)"If the network should use the existing weighs as a base \
for the population. Default False.", NULL},
  */

  {(char*)"crossover_chance",              \
   (getter)GenNetwork_getCrossoverChance, \
   (setter)GenNetwork_setCrossoverChance,                      \
   (char*)"Probability to perform crossover before mutation.", NULL},

  {(char*)"connection_mutation_chance",              \
   (getter)GenNetwork_getConnsMutationChance, \
   (setter)GenNetwork_setConnsMutationChance,                      \
   (char*)"Probability to mutate each connection (shift the bit).", NULL},

  {(char*)"activation_mutation_chance",              \
   (getter)GenNetwork_getActFuncMutationChance, \
   (setter)GenNetwork_setActFuncMutationChance,                      \
   (char*)"Probability to mutate each activation function.", NULL},


  {(char*)"selection_method",              \
   (getter)GenNetwork_getSelectionMethod, \
   (setter)GenNetwork_setSelectionMethod,                      \
   (char*)"Method to select parents.", NULL},

  {(char*)"crossover_method",              \
   (getter)GenNetwork_getCrossoverMethod, \
   (setter)GenNetwork_setCrossoverMethod,                      \
   (char*)"Method to pair parents.", NULL},

  {(char*)"fitness_function",              \
   (getter)GenNetwork_getFitnessFunction, \
   (setter)GenNetwork_setFitnessFunction,                      \
   (char*)"Fitness function to use during evolution.", NULL},

  {NULL} // Sentinel
};

// Python type declaration
static PyTypeObject GenNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "_ann.gennetwork", // tp_name  VITAL CORRECT PACKAGE NAME FOR PICKLING!
  sizeof(PyGenNetwork),  // tp_basicsize
  0,                                              // tp_itemsize
  0,                  // tp_dealloc
  0,                                              // tp_print
  0,                                              //* tp_getattr
  0,                                              //* tp_setattr
  0,                                              //* tp_compare
  0,                                              //* tp_repr
  0,                                              //* tp_as_number
  0,                                              //* tp_as_sequence
  0,                                              //* tp_as_mapping
  0,                                              //* tp_hash
  0,                                              //* tp_call
  0,                                              //* tp_str
  0,      //* tp_getattro
  0,                                              //* tp_setattro
  0,                                              //* tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,       //* tp_flags
  "geneticnetwork(inputcount, hiddencount, outputcount)\n\n\
A feed forward neural network which can be trained with a\n\
genetic algorithm.\n\
\n\
Parameters\n\
----------\n\
inputcount : integer\n\
    The number of input neurons.\n\
hiddencount : integer\n\
    The number of hidden neurons. Note that this is the total number of\n\
    neurons in all hidden layers.\n\
outputcount : integer\n\
    The number of output neurons.\n\
\n\
Returns\n\
-------\n\
out : geneticnetwork\n\
    A genetic matrix network with the specified number of neurons.\n\
\n\
The training parameters are set as properties directly on the network, \n\
where some properties take integer constants defined on this class.\n\
\n\
Properties\n\
----------\n\
net.selection_method : method id (integer)\n\
    The way in which networks are chosen as parents for\n\
    crossover. Possible values are:\n\
    - Geometric picks the parents from a geometric distribution\n\
      which benefits the fittest network the most.\n\
    - Roulette gives each network a probability proportional to their score\n\
      divided by the sum of all scores.\n\
    - Tournament picks, for each parent, two networks and selects the most\n\
      fit. This is the least elitist, and least predictable.\n\
    Default is SELECTION_GEOMETRIC.\n\
net.crossover_method : method id (integer)\n\
    The crossover operation can be done in several ways. Given two parents\n\
    with genomes XXXXX and YYYYY for example:\n\
    - Uniform, assembles a child by selecting neurons and associated\n\
      weights at random from each parent with equal probability.\n\
      Example children: XYXYX, YXYXY.\n\
    - Two point, selects two places along the genome and exchanges everything\n\
      in-between for the other parent's genetic material. This produces two\n\
      children, with 'opposite' genomes.\n\
      Example children: XYYXX, YXXYY.\n\
    - One point, selects a single pivot point along the genome and exchanges\n\
      everything before/after the point for the other parent's genome.\n\
      Produces two children, with 'opposite' genomes.\n\
       Example children: XXYYY, YYXXX.\n\
    Default is CROSSOVER_ONEPOINT.\n\
net.fitness_function : function id (integer)\n\
    The function that will judge the performance of the networks.\n\
    Default is FITNESS_MSE.\n\
net.generations : integer\n\
    Number of generations to train for, where the number of 'iterations'\n\
    or networks 'born' in each generation is equal to population_size.\n\
    Default 100.\n\
net.population_size : integer\n\
    Size of population to create. This indirectly makes generations longer.\n\
    Default 50.\n\
net.crossover_chance : float\n\
    Probability to perform cross-over between parents. Default 1.0.\n\
net.connection_mutation_chance : float\n\
    Probability to flip each bit in the connection matrix. Default 0.\n\
net.activation_mutation_chance : float\n\
    Probability for each neuron to switch activation function. Default 0.\n\
net.weight_mutation_chance : float\n\
    Probability of a single weight being selected for mutation. Default 0.15.\n\
net.weight_mutation_factor: float\n\
    Standard deviation of gaussian distribution used for mutation. A random\n\
    number R is picked from this distribution and then weight w = w + R.\n\
    Default 1.0.\n\
net.weight_mutation_halfpoint : integer\n\
    Generation at which the weight_mutation_factor should have decreased to\n\
    half of its original value. Default 0 (off).\n\
net.mingroup : integer\n\
    Used by some fitness functions (SURV_KAPLAN_*) to determine minimum\n\
    valid group size. Default 1.\n\
\n\
See Also\n\
--------\n\
rpropnetwork\n\
\n\
Examples\n\
--------\n\
>>> geneticnetwork(2, 3, 1)\n\
geneticnetwork(2, 3, 1)\n", // tp_doc
  0,                                              //* tp_traverse
  0,                                              //* tp_clear
  0,                                              //* tp_richcompare
  0,                                              //* tp_weaklistoffset
  0,                                              //* tp_iter
  0,                                              //* tp_iternext
  GenNetworkMethods,                                //* tp_methods
  0,                                       //* tp_members
  GenNetworkGetSetters,              //* tp_getset
  0,                                              //* tp_base
  0,                                              //* tp_dict
  0,                                              //* tp_descr_get
  0,                                              //* tp_descr_set
  0,                                              //* tp_dictoffset
  (initproc)GenNetwork_init,                      //* tp_init
  0,                                              //* tp_alloc
  0,                                  //* tp_new
};


/*
 * Python module declaration
 * =========================
 */
/*
Module methods
*/
static PyMethodDef annMethods[] = {
    {"cindex", (PyCFunction) CIndex_getCindex,                     \
     METH_VARARGS | METH_KEYWORDS,
     "cindex(targets, outputs)\n\
\n\
Calculates the concordance index. This is an O(n^2) implementation.\n\
\n\
Parameters\n\
----------\n\
targets : ndarray of floats\n\
    2-dimension array as (survival time, event flag).\n\
outputs : ndarray of floats\n\
    1-dimensional array of predictions. Must have the same number of\n\
    rows as targets. If not 1-dimensional, the array is flattened first.\n\
\n\
Returns\n\
-------\n\
cindex : float\n\
    A value between 0 and 1, where 0.5 is expected for random sorting.\n\
    1 is perfect concordance, and 0 is perfect anti-concordance.\n"},
    {"get_error", (PyCFunction) ErrorFuncs_getError,
     METH_VARARGS | METH_KEYWORDS, \
     "get_error(errorfunc, targets, outputs)\n\
\n\
Calculates the error using the specified error function on the given\n\
target/output arrays. The result is averaged over the first axis.\n\
\n\
Parameters\n\
----------\n\
errorfunc : function id (integer)\n\
    The error function to calculate error with.\n\
targets : ndarray of floats\n\
    2-dimension array as (survival time, event flag).\n\
outputs : ndarray of floats\n\
    2-dimensional array of predictions. Must have the same number of\n\
    rows as targets. If the specified error function only functions with\n\
    one prediction value, then the second value is ignored.\n\
\n\
Returns\n\
-------\n\
error : float\n\
    The calculated error for the predictions.\n"},
    {"get_deriv", (PyCFunction) ErrorFuncs_getDeriv,
     METH_VARARGS | METH_KEYWORDS, \
     "get_deriv(error_func, targets, outputs)\n\n \
\n\
Calculates the derivative using the specified error function on the given\n\
target/output arrays. The result is averaged over the first axis.\n\
\n\
Parameters\n\
----------\n\
errorfunc : function id (integer)\n\
    The error function to calculate error with.\n\
targets : ndarray of floats\n\
    2-dimension array as (survival time, event flag).\n\
outputs : ndarray of floats\n\
    2-dimensional array of predictions. Must have the same number of\n\
    rows as targets. If the specified error function only functions with\n\
    one prediction value, then the second value is ignored.\n\
\n\
Returns\n\
-------\n\
deriv : float\n\
    The calculated error derivative for the predictions.\n"},

    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* PyMODINIT_FUNC automatically does extern C in c++
   However it is not automatic in python2, hence we must
   include it in that case.
 */
extern "C" {
  void setModuleConstants(PyObject *dict) {
    // Error function
    PyDict_SetItemString(dict, "ERROR_MSE",
                         Py_BuildValue("i", ErrorFunction::ERROR_MSE));
    PyDict_SetItemString(dict, "ERROR_SURV_MSE",
                         Py_BuildValue("i", ErrorFunction::ERROR_SURV_MSE));
    PyDict_SetItemString(dict, "ERROR_SURV_LIKELIHOOD",
                         Py_BuildValue("i",
                                       ErrorFunction::ERROR_SURV_LIKELIHOOD));
  }

  MOD_INIT(_ann) {
    PyObject* mod;

    // Need to import numpy arrays before anything is done
    import_array();

    // Create the module
    MOD_DEF(mod, "_ann", "C++ implementation of the neural network.",
            annMethods)

    if (mod == NULL) {
      return MOD_ERROR_VAL;
    }

    // Module constants
    setModuleConstants(PyModule_GetDict(mod));

    // MatrixNetwork

    // Make it ready
    if (PyType_Ready(&MatrixNetworkType) < 0) {
      //Py_DECREF(&FFNetworkType);
      //Py_DECREF(&RPropNetworkType);
      //Py_DECREF(&CascadeNetworkType);
      return MOD_ERROR_VAL;
    }

    // Add static class variables
    PyDict_SetItemString(MatrixNetworkType.tp_dict, "LINEAR",
                         Py_BuildValue("i", LINEAR));
    PyDict_SetItemString(MatrixNetworkType.tp_dict, "LOGSIG",
                         Py_BuildValue("i", LOGSIG));
    PyDict_SetItemString(MatrixNetworkType.tp_dict, "TANH",
                         Py_BuildValue("i", TANH));
    PyDict_SetItemString(MatrixNetworkType.tp_dict, "SOFTMAX",
                         Py_BuildValue("i", SOFTMAX));


    // Add the type to the module.
    Py_INCREF(&MatrixNetworkType);
    PyModule_AddObject(mod, "matrixnetwork", (PyObject*)&MatrixNetworkType);


    /*
     * RPropNetwork
     */
    RPropNetworkType.tp_base = &MatrixNetworkType;
    if (PyType_Ready(&RPropNetworkType) < 0) {
      Py_DECREF(&MatrixNetworkType);
      return MOD_ERROR_VAL;
    }

    // add static variables
    setRPropConstants(RPropNetworkType.tp_dict);

    Py_INCREF(&RPropNetworkType);
    PyModule_AddObject(mod, "rpropnetwork", (PyObject*)&RPropNetworkType);


    // GenNetwork
    GenNetworkType.tp_base = &MatrixNetworkType;
    if (PyType_Ready(&GenNetworkType) < 0) {
      Py_DECREF(&MatrixNetworkType);
      Py_DECREF(&RPropNetworkType);
      return MOD_ERROR_VAL;
    }

    // Add static class variables
    setGeneticNetworkConstants(GenNetworkType.tp_dict);

    Py_INCREF(&GenNetworkType);
    PyModule_AddObject(mod, "gennetwork", (PyObject*)&GenNetworkType);

    return MOD_SUCCESS_VAL(mod);
  }
}
