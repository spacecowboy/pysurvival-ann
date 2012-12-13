/*
 * PythonModule.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "Python.h"
#include "PythonModule.h"
#include "structmember.h" // used to declare member list
#include "ModuleHeader.h" // Must include this before arrayobject
#include <numpy/arrayobject.h> // Numpy seen from C
#include "FFNetworkWrapper.h"
#include "RPropNetworkWrapper.h"
#include "GeneticSurvivalNetworkWrapper.h"
#include "activationfunctions.h"
#include "CIndexWrapper.h"

/*
 * FFNetwork
 * =========
 */

/*
 * Public Python methods
 * ---------------------
 */
static PyMethodDef FFNetworkMethods[] =
{
	{"output", (PyCFunction) FFNetwork_output, METH_O, "Computes the network's output."},
    {"connectHToH", (PyCFunction) FFNetwork_connectHToH, METH_VARARGS | METH_KEYWORDS, "Connect hidden neuron i to hidden neuron j with the specified weight."},
    {"connectHToI", (PyCFunction) FFNetwork_connectHToI, METH_VARARGS | METH_KEYWORDS, "Connect hidden neuron i to input neuron j with the specified weight."},
    {"connectHToB", (PyCFunction) FFNetwork_connectHToB, METH_VARARGS | METH_KEYWORDS, "Connect hidden neuron i to bias neuron with specified weight"},
    {"connectOToH", (PyCFunction) FFNetwork_connectOToH, METH_VARARGS | METH_KEYWORDS, "Connect output neuron i to hidden neuron j with specified weight"},
    {"connectOToI", (PyCFunction) FFNetwork_connectOToI, METH_VARARGS | METH_KEYWORDS, "Connect output neuron i to input neuron j with specified weight"},
    {"connectOToB", (PyCFunction) FFNetwork_connectOToB, METH_VARARGS | METH_KEYWORDS, "Connect output neuron i to bias neuron with specified weight"},

    {"getInputWeightsOfOutput", (PyCFunction) FFNetwork_getInputWeightsOfOutput, METH_VARARGS | METH_KEYWORDS, "Returns a dictionary of the input weights for specified output neuron."},

    {"getNeuronWeightsOfOutput", (PyCFunction) FFNetwork_getNeuronWeightsOfOutput, METH_VARARGS | METH_KEYWORDS, "Returns a dictionary of the neuron weights for specified output neuron."},

    {"getInputWeightsOfHidden", (PyCFunction) FFNetwork_getInputWeightsOfHidden, METH_VARARGS | METH_KEYWORDS, "Returns a dictionary of the input weights for specified hidden neuron."},

    {"getNeuronWeightsOfHidden", (PyCFunction) FFNetwork_getNeuronWeightsOfHidden, METH_VARARGS | METH_KEYWORDS, "Returns a dictionary of the neuron weights for specified hidden neuron."},


	{NULL}, // So that we can iterate safely below
};

/*
 * Public Python members
 * ---------------------
 */
static PyMemberDef FFNetworkMembers[] = {
		{NULL} // for safe iteration
};

/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef FFNetworkGetSetters[] = {
  {(char*)"numOfInputs", (getter)FFNetwork_getNumOfInputs, NULL,    \
   (char*)"Number of input neurons", NULL},
  {(char*)"numOfHidden", (getter)FFNetwork_getNumOfHidden, NULL,    \
   (char*)"Number of hidden neurons", NULL},
  {(char*)"numOfOutputs", (getter)FFNetwork_getNumOfOutputs, NULL,  \
   (char*)"Number of output neurons", NULL},

  {(char*)"outputActivationFunction", (getter)FFNetwork_getOutputActivationFunction, (setter)FFNetwork_setOutputActivationFunction, (char*)"The activation function used by output neurons. For example network.LOGSIG", NULL},
  {(char*)"hiddenActivationFunction", (getter)FFNetwork_getHiddenActivationFunction, (setter)FFNetwork_setHiddenActivationFunction, (char*)"The activation function used by hidden neurons. For example network.TANH", NULL},

  {NULL} // Sentinel
};

/*
 * Python type declaration
 * -----------------------
 */
static PyTypeObject FFNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
	"_ann.ffnetwork",		/* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
	sizeof(PyFFNetwork),					/* tp_basicsize */
	0,						/* tp_itemsize */
	(destructor)FFNetwork_dealloc,			/* tp_dealloc */
	0,						/* tp_print */
	0,						/* tp_getattr */
	0,						/* tp_setattr */
	0,						/* tp_compare */
	0,						/* tp_repr */
	0,						/* tp_as_number */
	0,						/* tp_as_sequence */
	0,						/* tp_as_mapping */
	0,						/* tp_hash */
	0,						/* tp_call */
	0,						/* tp_str */
	0,						/* tp_getattro */
	0,						/* tp_setattro */
	0,						/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 	/* tp_flags*/
	"A feed forward neural network.",			/* tp_doc */
	0,						/* tp_traverse */
	0,			 			/* tp_clear */
	0,			 			/* tp_richcompare */
	0,			 			/* tp_weaklistoffset */
	0,			 			/* tp_iter */
	0,			 			/* tp_iternext */
	FFNetworkMethods,					/* tp_methods */
	FFNetworkMembers,					/* tp_members */
	FFNetworkGetSetters,			 			/* tp_getset */
	0,			 			/* tp_base */
	0,			 			/* tp_dict */
	0,			 			/* tp_descr_get */
	0,			 			/* tp_descr_set */
	0,			 			/* tp_dictoffset */
	(initproc)FFNetwork_init,				/* tp_init */
	0,			 			/* tp_alloc */
	FFNetwork_new,			 		/* tp_new */
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
  {"learn", (PyCFunction) RPropNetwork_learn, METH_VARARGS | METH_KEYWORDS, "Trains the network using RProp."},
                        {NULL}, // So that we can iterate safely below
};


/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef RPropNetworkGetSetters[] = {
  {(char*)"maxError", (getter)RPropNetwork_getMaxError,         \
   (setter)RPropNetwork_setMaxError,                            \
   (char*)"Maximum error allowed for early stopping", NULL},
  {(char*)"maxEpochs", (getter)RPropNetwork_getMaxEpochs,           \
   (setter)RPropNetwork_setMaxEpochs,                               \
   (char*)"Maximum number of epochs allowed for training", NULL},
  {(char*)"printEpoch", (getter)RPropNetwork_getPrintEpoch,             \
   (setter)RPropNetwork_setPrintEpoch,                                  \
   (char*)"How often stats are printed during training. 0 to disable.", NULL},
  {NULL} // Sentinel
};



/*
 *  * Python type declaration
 *   * -----------------------
 *    */
static PyTypeObject RPropNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
        "_ann.rpropnetwork",                /* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
        sizeof(PyRPropNetwork),                                    /* tp_basicsize */
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
        "A feed forward neural network which can be trained with RProp.",                       /* tp_doc */
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        RPropNetworkMethods,                                       /* tp_methods */
        0,                                       /* tp_members */
        RPropNetworkGetSetters,                                            /* tp_getset */
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
 * Genetic survival network
 * ========================
 */

/*
 * Public Python methods
 * ---------------------
 */
static PyMethodDef GenSurvNetworkMethods[] =
{
  {"learn", (PyCFunction) GenSurvNetwork_learn, METH_VARARGS | METH_KEYWORDS, "Trains the network using a genetic algorithm."},
  {NULL}, // So that we can iterate safely below
};


/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef GenSurvNetworkGetSetters[] = {
  {(char*)"generations", (getter)GenSurvNetwork_getGenerations, \
   (setter)GenSurvNetwork_setGenerations,                       \
   (char*)"Time to train", NULL},
  {(char*)"populationSize", (getter)GenSurvNetwork_getPopulationSize,   \
   (setter)GenSurvNetwork_setPopulationSize,                            \
   (char*)"Number of networks created each generation", NULL},
  {(char*)"weightMutationChance", (getter)GenSurvNetwork_getWeightMutationChance, \
   (setter)GenSurvNetwork_setWeightMutationChance,                      \
   (char*)"The chance of a single weight being changed during cloning", NULL},
  {(char*)"weightMutationHalfPoint",                  \
   (getter)GenSurvNetwork_getWeightMutationHalfPoint, \
   (setter)GenSurvNetwork_setWeightMutationHalfPoint,                   \
   (char*)"If time dependant mutation is desired, set this to a non-zero value.\
 StdDev will decrease linearly and reach half at specified generation.", NULL},
  {(char*)"weightMutationStdDev",                  \
   (getter)GenSurvNetwork_getWeightMutationStdDev, \
   (setter)GenSurvNetwork_setWeightMutationStdDev,                      \
   (char*)"Mutations are gaussians with this stddev and added to current\
 weight.", NULL},
  {NULL} // Sentinel
};



/*
 *  * Python type declaration
 *   * -----------------------
 *    */
static PyTypeObject GenSurvNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
        "_ann.gensurvnetwork",                /* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
        sizeof(PyGenSurvNetwork),                                    /* tp_basicsize */
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
        "A feed forward neural network for survival diagnosis. Trains on the C-index using a genetic algoritm.",                       /* tp_doc */
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        GenSurvNetworkMethods,                                       /* tp_methods */
        0,                                       /* tp_members */
        GenSurvNetworkGetSetters,                                            /* tp_getset */
        0,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
       (initproc)GenSurvNetwork_init,                               /* tp_init */
        0,                                              /* tp_alloc */
        0,                                  /* tp_new */
};


/*
 * Python module declaration
 * =========================
 */
/*
Module methods
*/
static PyMethodDef annMethods[] = {
  {"get_C_index", (PyCFunction) CIndex_getCindex, METH_VARARGS | METH_KEYWORDS, "Calculates the C-index. Note that outputs converted to one dimension. Targets should be (survival time, event)\n\nInput: Targets, Predictions\nReturns: 0 if no concordance could be found."},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

/* PyMODINIT_FUNC automatically does extern C in c++
   However it is not automatic in python2, hence we must
   include it in that case.
 */
extern "C" {
  MOD_INIT(_ann) {
    PyObject* mod;

    // Need to import numpy arrays before anything is done
    import_array();

    // Create the module
    MOD_DEF(mod, "_ann", "C++ implementation of the neural network.", annMethods)

      if (mod == NULL) {
        return MOD_ERROR_VAL;
      }

    /*
     * FFNetwork
     * ---------
     */

    // Make it ready
    if (PyType_Ready(&FFNetworkType) < 0) {
      return MOD_ERROR_VAL;
    }

    // Add static class variables
    PyDict_SetItemString(FFNetworkType.tp_dict, "LINEAR", Py_BuildValue("i", LINEAR));
    PyDict_SetItemString(FFNetworkType.tp_dict, "LOGSIG", Py_BuildValue("i", LOGSIG));
    PyDict_SetItemString(FFNetworkType.tp_dict, "TANH", Py_BuildValue("i", TANH));


    // Add the type to the module.
    Py_INCREF(&FFNetworkType);
    PyModule_AddObject(mod, "ffnetwork", (PyObject*)&FFNetworkType);

    /*
     * RPropNetwork
     */
    RPropNetworkType.tp_base = &FFNetworkType;
    if (PyType_Ready(&RPropNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      return MOD_ERROR_VAL;
    }

    Py_INCREF(&RPropNetworkType);
    PyModule_AddObject(mod, "rpropnetwork", (PyObject*)&RPropNetworkType);

    /*
     * GenSurvNetwork
     */
    GenSurvNetworkType.tp_base = &FFNetworkType;
    if (PyType_Ready(&GenSurvNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      Py_DECREF(&RPropNetworkType);
      return MOD_ERROR_VAL;
    }

    Py_INCREF(&GenSurvNetworkType);
    PyModule_AddObject(mod, "gensurvnetwork", (PyObject*)&GenSurvNetworkType);

    return MOD_SUCCESS_VAL(mod);
  }
}
