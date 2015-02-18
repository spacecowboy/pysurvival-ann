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
   "Computes the network's output."},

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
  "A feed forward neural network.",			//* tp_doc
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
     "Trains the network using iRProp+ as described in:\n\
'Improving the Rprop Learning Algorithm' by \n\
Christian Igel and Michael Hüsken."},
    {NULL}, // So that we can iterate safely below
};


/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef RPropNetworkGetSetters[] = {
  {(char*)"maxError",
   (getter)RPropNetwork_getMaxError,                            \
   (setter)RPropNetwork_setMaxError,                            \
   (char*)"Maximum error allowed for early stopping", NULL},

  {(char*)"minErrorFrac", \
   (getter)RPropNetwork_getMinErrorFrac, \
   (setter)RPropNetwork_setMinErrorFrac, \
   (char*)"Minimum relative change in error over 100 epochs to allow before stopping", \
   NULL},

  {(char*)"maxEpochs",
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
    "A feed forward neural network which can be trained with iRProp+ \
 as described in:\n \
'Improving the Rprop Learning Algorithm' by \n \
Christian Igel and Michael Hüsken.\n \
\n \
You can change the behaviour of the algorithm by setting a few values:\n \
**error_function** - Choose the appropriate error function for your data\n \
\n \
**max_error** - Used for early stopping. Set to zero to train until limit\n \
\n \
**max_epochs** - Maximum number of epochs to train for.\n", // tp_doc
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
     "Trains the network using a genetic algorithm."},
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
  "A neural network that trains using a genetic algorithm.\n\
A few properties influence the algorithm's behaviour:\n\
\n                                                                      \
**selection_method** - The way in which networks are chosen as parents for\n \
crossover. *Geometric* picks the parents from a geometric distribution\n \
to benifit the fittest network the most.\n                              \
*Roulette* gives each network a probability proportional to their score\n \
divided by the sum of all scores. \n                                    \
*Tournament* picks, for each parent, two networks and uses the fittest.\n \
This is the least elitist, and least predictable.\n                     \
\n                                                                      \
**crossover_method** - The crossover operation itself can be done in several\n \
ways. *Random neuron* assembles a child by selecting neurons and associated\n \
weight vector at random from each parent with equal probability. \n     \
*Two point* selects two places along the genome and exchanges everything\n \
in between for the other parent's genetic material. This produces two\n\
children, with opposite genomes.\n                                     \
\n                                                                      \
**insertion_method** - Once a crossover, and possible mutation, is performed\n \
the children must be inserted into the population somehow.\n            \
*Insert all* simply inserts the children into the sorted population.\n  \
Nothing is done with the parents. \n                                    \
*Insert fittest* on the other hand makes a choice between the children\n \
and the parents. The fittest of the two is (re)inserted in the\n        \
population. The exception is if the parent is the best current member,\n \
then it is always kept in the population.\n                             \
\n                                                                      \
**fitness_function** - The function that will judge the performance \n  \
of the networks.\n", // tp_doc
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
    {"get_C_index", (PyCFunction) CIndex_getCindex,                     \
     METH_VARARGS | METH_KEYWORDS, "Calculates the C-index. Note that outputs \
are converted to one dimension. Targets should be (survival time,\
 event)\n\nInput: Targets, Predictions\nReturns: 0 if no concordance \
could be found."},

    {"get_error", (PyCFunction) ErrorFuncs_getError,
     METH_VARARGS | METH_KEYWORDS, \
     "get_error(errorfunc, targets, outputs)\n\n \
\nCalculates the error using the specified error function on the given \
target/output arrays. The result is averaged over the first axis."},

    {"get_deriv", (PyCFunction) ErrorFuncs_getDeriv,
     METH_VARARGS | METH_KEYWORDS, \
     "get_deriv(error_func, targets, outputs)\n\n \
Calculates the derivative using the specified error function on the given \
target/output arrays. The result is averaged over the first axis."},


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
