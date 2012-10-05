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
	{"numOfInputs", T_INT, offsetof(FFNetwork, numOfInputs), 0, "The number of input values required by the network."},
	{"numOfHidden", T_INT, offsetof(FFNetwork, numOfHidden), 0, "The number of hidden neurons contained in the network."},
	{"numOfOutputs", T_INT, offsetof(FFNetwork, numOfOutput), 0, "The number of output values returned by the network."},
		{NULL} // for safe iteration
};

/*
 * Python type declaration
 * -----------------------
 */
static PyTypeObject FFNetworkType = {
	PyObject_HEAD_INIT(NULL)
	0,						/* ob_size */
	"ann.ffnetwork",		/* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
	sizeof(FFNetwork),					/* tp_basicsize */
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
	0,			 			/* tp_getset */
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


}

}
