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
#include "GeneticNetworkWrapper.h"
#include "GeneticSurvivalNetworkWrapper.h"
#include "activationfunctions.h"
#include "CIndexWrapper.h"
#include "CascadeNetworkWrapper.h"
//#include "CoxCascadeNetworkWrapper.h"
#include "GeneticCascadeNetworkWrapper.h"
#include "GeneticNetwork.h"

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
    {(char*)"numOfInputs", (getter)FFNetwork_getNumOfInputs, NULL,  \
     (char*)"Number of input neurons", NULL},
    {(char*)"numOfHidden", (getter)FFNetwork_getNumOfHidden, NULL,  \
     (char*)"Number of hidden neurons", NULL},
    {(char*)"numOfOutputs", (getter)FFNetwork_getNumOfOutputs, NULL,    \
     (char*)"Number of output neurons", NULL},

    {(char*)"outputActivationFunction", \
     (getter)FFNetwork_getOutputActivationFunction, \
     (setter)FFNetwork_setOutputActivationFunction, \
     (char*)"The activation function used by output neurons. \
For example network.LOGSIG", NULL},
    {(char*)"hiddenActivationFunction", \
     (getter)FFNetwork_getHiddenActivationFunction, \
     (setter)FFNetwork_setHiddenActivationFunction, \
     (char*)"The activation function used by hidden neurons. \
For example network.TANH", NULL},

    {NULL} // Sentinel
};

/*
 * Python type declaration
 * -----------------------
 */
static PyTypeObject FFNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_ann.ffnetwork",		/* tp_name */ // VITAL PACKAGE NAME FOR PICKLING!
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
    {"learn", (PyCFunction) RPropNetwork_learn,                         \
     METH_VARARGS | METH_KEYWORDS, "Trains the network using RProp."},
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
    "A feed forward neural network which can be trained with RProp.",// tp_doc
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

/*
 * Public Python methods
 * ---------------------
 */
static PyMethodDef GenNetworkMethods[] =
{
    {"learn", (PyCFunction) GenNetwork_learn,                           \
     METH_VARARGS | METH_KEYWORDS, \
     "Trains the network using a genetic algorithm."},
    {NULL}, // So that we can iterate safely below
};


/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef GenNetworkGetSetters[] = {
  {(char*)"generations", (getter)GenNetwork_getGenerations, \
   (setter)GenNetwork_setGenerations,                       \
   (char*)"Time to train", NULL},
  {(char*)"populationSize", (getter)GenNetwork_getPopulationSize,   \
   (setter)GenNetwork_setPopulationSize,                            \
   (char*)"Number of networks created each generation", NULL},
  {(char*)"weightMutationChance", (getter)GenNetwork_getWeightMutationChance, \
   (setter)GenNetwork_setWeightMutationChance,                      \
   (char*)"The chance of a single weight being changed during cloning", NULL},
  {(char*)"weightMutationHalfPoint",                  \
   (getter)GenNetwork_getWeightMutationHalfPoint, \
   (setter)GenNetwork_setWeightMutationHalfPoint,                   \
   (char*)"If time dependant mutation is desired, set this to a non-zero value.\
 StdDev will decrease linearly and reach half at specified generation.", NULL},
  {(char*)"weightMutationFactor",                  \
   (getter)GenNetwork_getWeightMutationFactor, \
   (setter)GenNetwork_setWeightMutationFactor,                      \
   (char*)"Mutations are gaussians with this stddev and added to current\
 weight.", NULL},

  {(char*)"weightDecayL1",                  \
   (getter)GenNetwork_getDecayL1, \
   (setter)GenNetwork_setDecayL1,                      \
   (char*)"Coefficient for L1 weight decay. Zero by default.", NULL},
  {(char*)"weightDecayL2",                  \
   (getter)GenNetwork_getDecayL2, \
   (setter)GenNetwork_setDecayL2,                      \
   (char*)"Coefficient for L2 weight decay. Zero by default.", NULL},
  {(char*)"weightElimination",                  \
   (getter)GenNetwork_getWeightElimination, \
   (setter)GenNetwork_setWeightElimination,                      \
   (char*)"Coefficient (g) for soft weight elimination: P = g * sum(). \
Zero by default.", NULL},
  {(char*)"weightEliminationLambda",                  \
   (getter)GenNetwork_getWeightEliminationLambda, \
   (setter)GenNetwork_setWeightEliminationLambda,                      \
   (char*)"Coefficient (l) for soft weight elimination: \
P = sum( w^2 / [l^2 + w^2] ). Zero by default.", NULL},

  {(char*)"resume",              \
   (getter)GenNetwork_getResume, \
   (setter)GenNetwork_setResume,                      \
   (char*)"If the network should use the existing weighs as a base \
for the population. Default False.", NULL},

  {(char*)"crossoverchance",              \
   (getter)GenNetwork_getCrossoverChance, \
   (setter)GenNetwork_setCrossoverChance,                      \
   (char*)"Probability to perform crossover before mutation.", NULL},

  {(char*)"selection_method",              \
   (getter)GenNetwork_getSelectionMethod, \
   (setter)GenNetwork_setSelectionMethod,                      \
   (char*)"Method to select parents.", NULL},

  {(char*)"crossover_method",              \
   (getter)GenNetwork_getCrossoverMethod, \
   (setter)GenNetwork_setCrossoverMethod,                      \
   (char*)"Way to perform crossover.", NULL},

  {(char*)"insert_method",              \
   (getter)GenNetwork_getInsertMethod, \
   (setter)GenNetwork_setInsertMethod,                      \
   (char*)"Way to insert parent and children into population after crossover.",\
   NULL},



  {NULL} // Sentinel
};



/*
 *  * Python type declaration
 *   * -----------------------
 *    */
static PyTypeObject GenNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_ann.gennetwork", // tp_name // VITAL CORRECT PACKAGE NAME FOR PICKLING!
    sizeof(PyGenNetwork),                                    /* tp_basicsize */
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
    "A neural network that trains using a genetic algorithm.\n\
A few properties influence the algorithm's behaviour:\n\
\n\
**selection_method** - The way in which networks are chosen as parents for\n\
crossover. *Geometric* picks the parents from a geometric distribution\n\
to benifit the fittest network the most.\n\
*Roulette* gives each network a probability proportional to their score\n\
divided by the sum of all scores. \n\
*Tournament* picks, for each parent, two networks and uses the fittest.\n\
This is the least elitist, and least predictable.\n\
\n\
**crossover_method** - The crossover operation itself can be done in several\n\
ways. *Random neuron* assembles a child by selecting neurons and associated\n\
weight vector at random from each parent with equal probability. \n\
*Two point* selects two places along the genome and exchanges everything\n\
in between for the other parent's genetic material. This produces two\n\
children, with opposite genomes.\n\
\n\
**insertion_method** - Once a crossover, and possible mutation, is performed\n\
the children must be inserted into the population somehow.\n\
*Insert all* simply inserts the children into the sorted population.\n\
Nothing is done with the parents. \n\
*Insert fittest* on the other hand makes a choice between the children\n\
and the parents. The fittest of the two is (re)inserted in the\n\
population. The exception is if the parent is the best current member,\n\
then it is always kept in the population.\n", /* tp_doc */
    0,                                              /* tp_traverse */
    0,                                              /* tp_clear */
    0,                                              /* tp_richcompare */
    0,                                              /* tp_weaklistoffset */
    0,                                              /* tp_iter */
    0,                                              /* tp_iternext */
    GenNetworkMethods,                                       /* tp_methods */
    0,                                       /* tp_members */
    GenNetworkGetSetters,              /* tp_getset */
    0,                                              /* tp_base */
    0,                                              /* tp_dict */
    0,                                              /* tp_descr_get */
    0,                                              /* tp_descr_set */
    0,                                              /* tp_dictoffset */
    (initproc)GenNetwork_init,                      /* tp_init */
    0,                                              /* tp_alloc */
    0,                                  /* tp_new */
};


/*
 * Genetic Survival Network
 * ========================
 */
static PyMethodDef GenSurvNetworkMethods[] =
{
    {"learn", (PyCFunction) GenSurvNetwork_learn, \
     METH_VARARGS | METH_KEYWORDS,                      \
     "Trains the network using a genetic algorithm."},
    {NULL}, // So that we can iterate safely below
};


static PyGetSetDef GenSurvNetworkGetSetters[] = {
    {NULL}, // Sentinel
};

/*
 *   Python type declaration
 *   -----------------------
 */
static PyTypeObject GenSurvNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_ann.gensurvnetwork", // tp_name VITAL PACKAGE NAME FOR PICKLING!
    sizeof(PyGenSurvNetwork),               /* tp_basicsize */
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
    "A feed forward neural network for survival analysis that trains \
using a genetic algoritm.",                       /* tp_doc */
    0,                                              /* tp_traverse */
    0,                                              /* tp_clear */
    0,                                              /* tp_richcompare */
    0,                                              /* tp_weaklistoffset */
    0,                                              /* tp_iter */
    0,                                              /* tp_iternext */
    GenSurvNetworkMethods,                  /* tp_methods */
    0,                                       /* tp_members */
    GenSurvNetworkGetSetters,                          /* tp_getset */
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
 * Cascade network
 * ========================
 */

/*
 * Public Python methods
 * ---------------------
 */
static PyMethodDef CascadeNetworkMethods[] =
{
  {NULL}, // So that we can iterate safely below
};

/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef CascadeNetworkGetSetters[] = {
  {(char*)"maxHidden", (getter)CascadeNetwork_getMaxHidden, \
   (setter)CascadeNetwork_setMaxHidden,                       \
   (char*)"Maximum allowed number of hidden neurons to create", NULL},
  {(char*)"maxHiddenEpochs", (getter)CascadeNetwork_getMaxHiddenEpochs,   \
   (setter)CascadeNetwork_setMaxHiddenEpochs,                            \
   (char*)"Maximum allowed epochs to train each hidden neuron with", NULL},
  {NULL} // Sentinel
};

/*
 *  * Python type declaration
 *   * -----------------------
 *    */
static PyTypeObject CascadeNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
        "_ann.cascadenetwork",                /* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
        sizeof(PyCascadeNetwork),                                    /* tp_basicsize */
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
        "An implementation of the Cascade correlation algorithm. Layers are trained with RProp.",                       /* tp_doc */
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        CascadeNetworkMethods,                                       /* tp_methods */
        0,                                       /* tp_members */
        CascadeNetworkGetSetters,                                            /* tp_getset */
        0,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
       (initproc)CascadeNetwork_init,                               /* tp_init */
        0,                                              /* tp_alloc */
        0,                                  /* tp_new */
};


/*
 * Cox Cascade network
 * ========================
 */

/*
 * Public Python methods
 * ---------------------
 *
static PyMethodDef CoxCascadeNetworkMethods[] =
{
  {"learn", (PyCFunction) CoxCascadeNetwork_learn, METH_VARARGS | METH_KEYWORDS, "Trains the network using the Cascade algorithm. \
Takes arguments: X (inputs), Y (time, event)."},

  {NULL}, // So that we can iterate safely below
  };*/

/*
 * Public Python members with get/setters
 * --------------------------------------
 *
static PyGetSetDef CoxCascadeNetworkGetSetters[] = {
  {NULL} // Sentinel
  };*/

/*
 *   Python type declaration
 *   -----------------------
 *
 *
static PyTypeObject CoxCascadeNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
        "_ann.coxcascadenetwork",
        sizeof(PyCoxCascadeNetwork),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "An implementation of the Cascade correlation algorithm. Hidden Layers are trained with RProp, while the output neuron is replaced by a Cox model.",
        0,
        0,
        0,
        0,
        0,
        0,
        CoxCascadeNetworkMethods,
        0,
        CoxCascadeNetworkGetSetters,
        0,
        0,
        0,
        0,
        0,
       (initproc)CoxCascadeNetwork_init,
        0,
        0,
};*/


/*
 * Genetic Cascade network
 * ========================
 */

/*
 * Public Python methods
 * ---------------------
 */
static PyMethodDef GeneticCascadeNetworkMethods[] =
{
  {"learn", (PyCFunction) GeneticCascadeNetwork_learn, METH_VARARGS | METH_KEYWORDS, "Trains the network using the Cascade algorithm. \
Takes arguments: X (inputs), Y (time, event)."},

  {NULL}, // So that we can iterate safely below
};

/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef GeneticCascadeNetworkGetSetters[] = {
  {NULL} // Sentinel
};

/*
 *  * Python type declaration
 *   * -----------------------
 *    */
static PyTypeObject GeneticCascadeNetworkType = {
  PyVarObject_HEAD_INIT(NULL, 0)
        "_ann.geneticcascadenetwork",                /* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
        sizeof(PyGeneticCascadeNetwork),                                    /* tp_basicsize */
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
        "An implementation of the Cascade correlation algorithm. Hidden Layers are trained with RProp, while the output neuron is trained genetically.",                       /* tp_doc */
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        GeneticCascadeNetworkMethods,                                       /* tp_methods */
        0,                                       /* tp_members */
        GeneticCascadeNetworkGetSetters,                                            /* tp_getset */
        0,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
       (initproc)GeneticCascadeNetwork_init,                               /* tp_init */
        0,                                              /* tp_alloc */
        0,                                  /* tp_new */
};


/*
 * Genetic Ladder network
 * ========================
 */

/*
 * Public Python methods
 * ---------------------
 */
static PyMethodDef GeneticLadderNetworkMethods[] =
{
  {"learn", (PyCFunction) GeneticLadderNetwork_learn, METH_VARARGS | METH_KEYWORDS, "Trains the network using a modified genetic Cascade algorithm. \
Takes arguments: X (inputs), Y (time, event)."},

  {NULL}, // So that we can iterate safely below
};

/*
 * Public Python members with get/setters
 * --------------------------------------
 */
static PyGetSetDef GeneticLadderNetworkGetSetters[] = {
  {NULL} // Sentinel
};

/*
 *  * Python type declaration
 *   * -----------------------
 *    */
static PyTypeObject GeneticLadderNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_ann.geneticladdernetwork",                /* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
    sizeof(PyGeneticLadderNetwork),                                    /* tp_basicsize */
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
    "A modified cascade algorithm. Neurons are trained as output neurons and then moved to the hidden layer as new output neurons are added. The final structure is fully shortcut connected as for cascade, but neurons are trained as output neurons on the full error function genetically.",                       /* tp_doc */
    0,                                              /* tp_traverse */
    0,                                              /* tp_clear */
    0,                                              /* tp_richcompare */
    0,                                              /* tp_weaklistoffset */
    0,                                              /* tp_iter */
    0,                                              /* tp_iternext */
    GeneticLadderNetworkMethods,                                       /* tp_methods */
    0,                                       /* tp_members */
    GeneticLadderNetworkGetSetters,                     /* tp_getset */
    0,                                              /* tp_base */
    0,                                              /* tp_dict */
    0,                                              /* tp_descr_get */
    0,                                              /* tp_descr_set */
    0,                                              /* tp_dictoffset */
    (initproc)GeneticLadderNetwork_init,            // tp_init
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
    {"get_C_index", (PyCFunction) CIndex_getCindex,                     \
     METH_VARARGS | METH_KEYWORDS, "Calculates the C-index. Note that outputs \
are converted to one dimension. Targets should be                       \
(survival time, event)\n\nInput: Targets, Predictions\nReturns: 0 if no \
concordance could be found."},
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
    PyDict_SetItemString(FFNetworkType.tp_dict, "LINEAR",
                         Py_BuildValue("i", LINEAR));
    PyDict_SetItemString(FFNetworkType.tp_dict, "LOGSIG",
                         Py_BuildValue("i", LOGSIG));
    PyDict_SetItemString(FFNetworkType.tp_dict, "TANH",
                         Py_BuildValue("i", TANH));


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
     * GenNetwork
     */
    GenNetworkType.tp_base = &FFNetworkType;
    if (PyType_Ready(&GenNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      Py_DECREF(&RPropNetworkType);
      return MOD_ERROR_VAL;
    }

    // Add static class variables

    PyDict_SetItemString(GenNetworkType.tp_dict, "SELECTION_GEOMETRIC",
                         Py_BuildValue("i", SELECTION_GEOMETRIC));
    PyDict_SetItemString(GenNetworkType.tp_dict, "SELECTION_ROULETTE",
                         Py_BuildValue("i", SELECTION_ROULETTE));
    PyDict_SetItemString(GenNetworkType.tp_dict, "SELECTION_TOURNAMENT",
                         Py_BuildValue("i", SELECTION_TOURNAMENT));



    PyDict_SetItemString(GenNetworkType.tp_dict, "CROSSOVER_NEURON",
                         Py_BuildValue("i", CROSSOVER_NEURON));
    PyDict_SetItemString(GenNetworkType.tp_dict, "CROSSOVER_TWOPOINT",
                         Py_BuildValue("i", CROSSOVER_TWOPOINT));



    PyDict_SetItemString(GenNetworkType.tp_dict, "INSERT_ALL",
                         Py_BuildValue("i", INSERT_ALL));
    PyDict_SetItemString(GenNetworkType.tp_dict, "INSERT_FITTEST",
                         Py_BuildValue("i", INSERT_FITTEST));



    Py_INCREF(&GenNetworkType);
    PyModule_AddObject(mod, "gennetwork", (PyObject*)&GenNetworkType);

    /*
     * GenSurvNetwork
     */
    GenSurvNetworkType.tp_base = &GenNetworkType;
    if (PyType_Ready(&GenSurvNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      Py_DECREF(&RPropNetworkType);
      Py_DECREF(&GenNetworkType);
      return MOD_ERROR_VAL;
    }

    Py_INCREF(&GenSurvNetworkType);
    PyModule_AddObject(mod, "gensurvnetwork", (PyObject*)&GenSurvNetworkType);


    /*
     * CascadeNetwork
     */
    CascadeNetworkType.tp_base = &RPropNetworkType;
    if (PyType_Ready(&CascadeNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      Py_DECREF(&RPropNetworkType);
      Py_DECREF(&GenNetworkType);
      Py_DECREF(&GenSurvNetworkType);
      return MOD_ERROR_VAL;
    }

    Py_INCREF(&CascadeNetworkType);
    PyModule_AddObject(mod, "cascadenetwork", (PyObject*)&CascadeNetworkType);

    /*
     * CoxCascadeNetwork
     *
    CoxCascadeNetworkType.tp_base = &CascadeNetworkType;
    if (PyType_Ready(&CoxCascadeNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      Py_DECREF(&RPropNetworkType);
      Py_DECREF(&GenNetworkType);
      Py_DECREF(&GenSurvNetworkType);
      Py_DECREF(&CascadeNetworkType);
      return MOD_ERROR_VAL;
    }

    Py_INCREF(&CoxCascadeNetworkType);
    PyModule_AddObject(mod, "coxcascadenetwork", (PyObject*)&CoxCascadeNetworkType);
    */

    /*
     * GeneticCascadeNetwork
     */
    GeneticCascadeNetworkType.tp_base = &CascadeNetworkType;
    if (PyType_Ready(&GeneticCascadeNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      Py_DECREF(&RPropNetworkType);
      Py_DECREF(&GenNetworkType);
      Py_DECREF(&GenSurvNetworkType);
      Py_DECREF(&CascadeNetworkType);
      //Py_DECREF(&CoxCascadeNetworkType);
      return MOD_ERROR_VAL;
    }

    Py_INCREF(&GeneticCascadeNetworkType);
    PyModule_AddObject(mod, "geneticcascadenetwork",
                       (PyObject*)&GeneticCascadeNetworkType);

    /*
     * GeneticLadderNetwork
     */
    GeneticLadderNetworkType.tp_base = &GeneticCascadeNetworkType;
    if (PyType_Ready(&GeneticLadderNetworkType) < 0) {
      Py_DECREF(&FFNetworkType);
      Py_DECREF(&RPropNetworkType);
      Py_DECREF(&GenNetworkType);
      Py_DECREF(&GenSurvNetworkType);
      Py_DECREF(&CascadeNetworkType);
      //Py_DECREF(&CoxCascadeNetworkType);
      Py_DECREF(&GeneticCascadeNetworkType);
      return MOD_ERROR_VAL;
    }

    Py_INCREF(&GeneticLadderNetworkType);
    PyModule_AddObject(mod, "geneticladdernetwork",
                       (PyObject*)&GeneticLadderNetworkType);


    return MOD_SUCCESS_VAL(mod);
  }
}
