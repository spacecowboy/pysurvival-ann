/*
 * FFNetworkWrapper.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "RPropNetworkWrapper.h"
// Must include this befor arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include <stdio.h>

extern "C" {

/*
 * Python constructor
 * ------------------
 * Responsible for taking the python variables and converting them
 * to numbers the c++ constructor can understand.
 *
 * Constructor is of the form: FFNetwork(inputsize, hiddensize, outputsize)
 */
PyObject *RPropNetwork_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

	// parse the arguments
	static char *kwlist[] =
			{ "numOfInputs", "numOfHidden", "numOfOutputs", NULL };

	// Unsigned integers, all mandatory
	unsigned int numOfInputs, numOfHidden, numOfOutputs;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist, &numOfInputs,
			&numOfHidden, &numOfOutputs)) {
		PyErr_Format(PyExc_ValueError,
				"Arguments should be (all mandatory positive integers): numOfInputs, numOfHidden, numOfOutputs");
		return NULL;
	}
	printf("RPropNetwork_new: We are past the if: %d %d %d\n", numOfInputs, numOfHidden, numOfOutputs);

	// Now construct the object
	RPropNetwork *self = (RPropNetwork*)type->tp_alloc(type, 0);
	printf("RPNetwork_new: allocated\n");
	new(self) RPropNetwork(numOfInputs, numOfHidden, numOfOutputs);
	//printf("RPNetwork_new: doing init\n");
	//self->initNodes();

	printf("RPNetwork_new: Past the construction\n");
	return (PyObject *) self;
}

/*
 * Python init
 * -----------
 */
int RPropNetwork_init(RPropNetwork *self, PyObject *args, PyObject *kwds) {
	printf("RPnetwrok_init: doing init\n");
	self->initNodes();
	return 0;
}

/*
 * Python destructor
 * -----------------
 */
/*
void RPropNetwork_dealloc(RPropNetwork *self) {	
	printf("rNetwork_dealloc: called\n");
	// Yes FFNetwork because children aren't allowed their own destructor
	self->~FFNetwork();
	printf("rNetwork_dealloc: destructed, now tp_free\n");
	self->ob_type->tp_free((PyObject*) self);
}
*/

/*
 * Wrapper methods
 * ===============
 */

void RPropNetwork_learn(RPropNetwork *self, PyObject *args, PyObjectd *kwargs) {
	printf("rNetwork_learn\n");

	PyObject *inputs = NULL;
	PyObject *targets = NULL;
	// Check inputs
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
					&inputs, &targets)) {
		PyErr_Format(PyExc_ValueError, "Arguments should be: inputs (2d array), targets (2d array)");
		return;
	}

	// Make sure they conform to required structure
	PyArrayObject *inputArray = NULL;
	PyArrayObject *targetArray = NULL;

	inputArray = (PyArrayObject *) PyArray_ContiguousFromObject(inputs, PyArray_DOUBLE, 2, 2);
	if (inputArray == NULL)
		return;

	targetArray = (PyArrayObject *) PyArray_ContiguousFromObject(targets, PyArray_DOUBLE, 2, 2);
	if (targetArray == NULL)
		return;

	// Objects were converted successfully. But make sure they are the same length!
	//int inputRows = inputArray->dimensions[0];
	//int targetRows = targetArray->dimensions[0];

	if (inputArray->dimensions[0] != targetRows->dimensions[0] ||
		inputArray->dimensions[1] != self->numOfInputs ||
		targetArray->dimensions[1] != self->numOfOutput) {
		// Decrement, set error and return
		PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same number of rows. Also the columns must match number of input/output neurons respectively.");
		Py_DECREF(inputArray);
		Py_DECREF(targetArray);

		return;
	}

	// Arguments are valid!
	
	printf("rNetwork_learn: past check\n");

	self->learn((double *)inputArray->data, (double *)targetArray->data, inputArray->dimensions[0]);

	// Decrement counters for inputArray and targetArray
	Py_DECREF(inputArray);
	Py_DECREF(targetArray);
}


}

