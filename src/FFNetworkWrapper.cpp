/*
 * FFNetworkWrapper.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "FFNetworkWrapper.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
/*
 * Python constructor
 * ------------------
 * Responsible for taking the python variables and converting them
 * to numbers the c++ constructor can understand.
 *
 * Constructor is of the form: FFNetwork(inputsize, hiddensize, outputsize)
 */
PyObject *FFNetwork_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

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

	// Now construct the object
	FFNetwork *self = new FFNetwork(numOfInputs, numOfHidden, numOfOutputs);
	return (PyObject *) self;
}

/*
 * Python init
 * -----------
 */
int FFNetwork_init(FFNetwork *self, PyObject *args, PyObject *kwds) {
	return 0;
}

/*
 * Python destructor
 * -----------------
 */
void FFNetwork_dealloc(FFNetwork *self) {
	// Not sure if this is calling the destructor of network
	self->ob_type->tp_free((PyObject*) self);
}

/*
 * Wrapper methods
 * ===============
 */

PyObject *FFNetwork_output(FFNetwork *self, PyObject *inputs) {
	if (!(PyList_CheckExact(inputs)
			|| (PyArray_NDIM(inputs) == 1 && PyArray_TYPE(inputs) == NPY_DOUBLE))) {
		PyErr_Format(PyExc_ValueError,
				"The input does not seem to be of a suitable list type. This method only accepts a python list or a 1D numpy array of doubles.");
		return NULL;
	}

	// First convert to normal double array

	bool pylist = PyList_CheckExact(inputs);
	PyObject *pyval = NULL;
	double *ptr = NULL;

	double dInputs[self->getNumOfInputs()];
	for (int i = 0; i < self->getNumOfInputs(); i++) {
		if (pylist) {
			// Actual python list
			PyObject *pyval = PyList_GetItem(inputs, i);
			if (pyval == NULL) {
				PyErr_Format(PyExc_ValueError,
						"Something went wrong when iterating of input values. Possibly wrong length?");
				return NULL;
			}
			dInputs[i] = PyFloat_AsDouble(pyval);
		} else {
			// Numpy array
			double *ptr = (double *) PyArray_GETPTR1((PyArrayObject*) inputs, i);
			if (ptr == NULL) {
				PyErr_Format(PyExc_ValueError,
						"Something went wrong when iterating of input values. Possibly wrong length?");
				return NULL;
			}
			dInputs[i] = *ptr;
		}
	}

	// compute output
	double dResult[self->getNumOfOutputs()];
	self->output(dInputs, dResult);
	// Now convert to numpy array
	npy_intp dims[1] = { self->getNumOfInputs() };
	PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	for (int i = 0; i < self->getNumOfOutputs(); i++) {
		ptr = (double *) PyArray_GETPTR1((PyArrayObject*) result, i);
		*ptr = dResult[i];
	}
	return result;
}




