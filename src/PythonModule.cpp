/*
 * PythonModule.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "ModuleHeader.h" // Must include this before arrayobject
#include <numpy/arrayobject.h> // Numpy seen from C
#include "FFNetworkWrapper.h"
#include "RPropNetworkWrapper.h"

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
	//{"__reduce__", (PyCFunction) Node_reduce, METH_NOARGS, "Needed for pickling. Specifices how to reconstruct the object."},
	//{"__getnewargs__", (PyCFunction) Node_getnewargs, METH_NOARGS, "Needed for pickling. Specifices what args to give new()."},
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
	{"numOfInputs", (getter)FFNetwork_getNumOfInputs, NULL, "Number of input neurons", NULL},
	{"numOfHidden", (getter)FFNetwork_getNumOfHidden, NULL, "Number of hidden neurons", NULL},
	{"numOfOutputs", (getter)FFNetwork_getNumOfOutputs, NULL, "Number of output neurons", NULL},
	{NULL} // Sentinel
};

/*
 * Python type declaration
 * -----------------------
 */
static PyTypeObject FFNetworkType = {
	PyObject_HEAD_INIT(NULL)
	0,						/* ob_size */
	"ann.ffnetwork",		/* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
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
 *  * Python type declaration
 *   * -----------------------
 *    */
static PyTypeObject RPropNetworkType = {
        PyObject_HEAD_INIT(NULL)
        0,                                              /* ob_size */
        "ann.rpropnetwork",                /* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
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
        0,                                            /* tp_getset */
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
 * Python module declaration
 * =========================
 */
extern "C" {

void
initann(void)
{
	PyObject* mod;

	// Need to import numpy arrays
	import_array();

	// Create the module
	mod = Py_InitModule3("ann", NULL, "C++ implementation of the neural network.");
	if (mod == NULL) {
		return;
	}

	/*
	 * FFNetwork
	 * ---------
	 */

	// Make it ready
	if (PyType_Ready(&FFNetworkType) < 0) {
		return;
	}

	// Add the type to the module.
	Py_INCREF(&FFNetworkType);
	PyModule_AddObject(mod, "ffnetwork", (PyObject*)&FFNetworkType);

	/*
 	 * RPropNetwork
 	 */
	RPropNetworkType.tp_base = &FFNetworkType;
	if (PyType_Ready(&RPropNetworkType) < 0) {
		Py_DECREF(&FFNetworkType);
		return;
	}

	Py_INCREF(&RPropNetworkType);
	PyModule_AddObject(mod, "rpropnetwork", (PyObject*)&RPropNetworkType);

}

}
