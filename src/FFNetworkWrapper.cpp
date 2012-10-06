/*
 * FFNetworkWrapper.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "FFNetworkWrapper.h"
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
	printf("FFNetwork_new: We are past the if: %d %d %d\n", numOfInputs, numOfHidden, numOfOutputs);

	// Now construct the object
	PyFFNetwork *self = (PyFFNetwork*)type->tp_alloc(type, 0);
	printf("FFNetwork_new: allocated\n");

	//new(self) FFNetwork(numOfInputs, numOfHidden, numOfOutputs);
	//printf("FFNetwork_new: doing init\n");
	//self->initNodes();

	//printf("FFNetwork_new: Past the construction\n");
	return (PyObject *) self;
}

/*
 * Python init
 * -----------
 */
int FFNetwork_init(PyFFNetwork *self, PyObject *args, PyObject *kwds) {
	printf("FFNetwork_init\n");
	static char *kwlist[] =
                        { "numOfInputs", "numOfHidden", "numOfOutputs", NULL };

	 unsigned int numOfInputs, numOfHidden, numOfOutputs;
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist, &numOfInputs,
                        &numOfHidden, &numOfOutputs)) {
                PyErr_Format(PyExc_ValueError,
                                "Arguments should be (all mandatory positive integers): numOfInputs, numOfHidden, numOfOutputs");
                return -1;
        }
        printf("FFNetwork_init: We are past the if: %d %d %d\n", numOfInputs, numOfHidden, numOfOutputs);

	self->net = new FFNetwork(numOfInputs, numOfHidden, numOfOutputs);

	printf("FFNetwork_init: net is: %d\n", self->net);
	if (self->net == NULL)
		return -1;

	self->net->initNodes();
	return 0;
}

/*
 * Python destructor
 * -----------------
 */
void FFNetwork_dealloc(PyFFNetwork *self) {
	printf("FFNetwork_dealloc: called\n");
	delete self->net;
	printf("FFNetwork_dealloc: destructed\n");
	self->ob_type->tp_free((PyObject*) self);
}

/*
 * Wrapper methods
 * ===============
 */

PyObject *FFNetwork_output(PyFFNetwork *self, PyObject *inputs) {
	printf("FFNetwork_output\n");
	if (!(PyList_CheckExact(inputs)
			|| (PyArray_NDIM(inputs) == 1 && PyArray_TYPE(inputs) == NPY_DOUBLE))) {
		PyErr_Format(PyExc_ValueError,
				"The input does not seem to be of a suitable list type. This method only accepts a python list or a 1D numpy array of doubles.");
		return NULL;
	}
	printf("FFNetwork_output: past check\n");

	// First convert to normal double array

	bool pylist = PyList_CheckExact(inputs);
	PyObject *pyval = NULL;
	double *ptr = NULL;

	double dInputs[self->net->getNumOfInputs()];
	printf("FFNetwork_output: before for loop\n");
	for (int i = 0; i < self->net->getNumOfInputs(); i++) {
		if (pylist) {
			printf("FFNetwork_output is list\n");
			// Actual python list
			pyval = PyList_GetItem(inputs, i);
			if (pyval == NULL) {
				PyErr_Format(PyExc_ValueError,
						"Something went wrong when iterating of input values. Possibly wrong length?");
				return NULL;
			}
			printf("FFNetwork_output before as double\n");
			dInputs[i] = PyFloat_AsDouble(pyval);
		} else {
			printf("FFNetwork_output: is numpy\n");
			// Numpy array
			ptr = (double *) PyArray_GETPTR1((PyArrayObject*) inputs, i);
			if (ptr == NULL) {
				PyErr_Format(PyExc_ValueError,
						"Something went wrong when iterating of input values. Possibly wrong length?");
				return NULL;
			}
			dInputs[i] = *ptr;
		}
	}

	printf("FFNetwork_output: compute output\n");
	// compute output
	double dResult[self->net->getNumOfOutputs()];
	printf("FFNetwork_output: actual output coming\n");
	self->net->output(dInputs, dResult);
	printf("FFNetwork_output: result is %f\n", dResult[0]);
	// Now convert to numpy array
	printf("FFNetwork_output: numpy convert\n");
	npy_intp dims[1] = { self->net->getNumOfOutputs() };
	printf("FFNetwork_output: before simple new\n");
	PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	printf("FFNetwork_output: time for loop\n");
	for (int i = 0; i < self->net->getNumOfOutputs(); i++) {
		printf("FFNetwork_output: in final loop\n");
		ptr = (double *) PyArray_GETPTR1((PyArrayObject*) result, i);
		printf("FFNetwork_output: got ptr\n");
		*ptr = dResult[i];
	}
	printf("FFNetwork_output: returning\n");
	return result;
}

  PyObject *FFNetwork_connectHToH(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
    unsigned int i, j;
    double weight;
    	// Check inputs
    static char *kwlist[] = {"i", "j", "weight", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IId", kwlist,
                                     &i, &j, &weight)) {
		PyErr_Format(PyExc_ValueError, "Expected integers i, j and double weight.");
		return NULL;
	}
    // Make sure they are valid
    if (i >= self->net->getNumOfHidden() ||
        j >= self->net->getNumOfHidden() ||
        j >= i) {
      PyErr_Format(PyExc_ValueError, "Not valid values for i and j. Valid values are: i, j < numOfHidden, j < i.");
      return NULL;
    }

    // They are valid, connect
    self->net->connectHToH(i, j, weight);

    // return None
    return Py_BuildValue("");
  }

PyObject *FFNetwork_connectHToI(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
    unsigned int i, j;
    double weight;
    	// Check inputs
    static char *kwlist[] = {"hiddenIndex", "inputIndex", "weight", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IId", kwlist,
                                     &i, &j, &weight)) {
		PyErr_Format(PyExc_ValueError, "Expected integers i, j and double weight.");
		return NULL;
	}
    // Make sure they are valid
    if (i >= self->net->getNumOfHidden() ||
        j >= self->net->getNumOfInputs() ) {
      PyErr_Format(PyExc_ValueError, "Not valid values for i and j. Valid values are: i < numOfHidden, j < numOfInput");
      return NULL;
    }

    // They are valid, connect
    self->net->connectHToI(i, j, weight);

    // return None
    return Py_BuildValue("");

}
PyObject *FFNetwork_connectHToB(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
    unsigned int i;
    double weight;
    	// Check inputs
    static char *kwlist[] = {"hiddenIndex", "weight", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Id", kwlist,
                                     &i, &weight)) {
		PyErr_Format(PyExc_ValueError, "Expected integer i and double weight.");
		return NULL;
	}
    // Make sure they are valid
    if (i >= self->net->getNumOfHidden() ) {
      PyErr_Format(PyExc_ValueError, "Not valid values for i. Valid values are: i < numOfHidden");
      return NULL;
    }

    // They are valid, connect
    self->net->connectHToB(i, weight);

    // return None
    return Py_BuildValue("");

}

PyObject *FFNetwork_connectOToH(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
    unsigned int i, j;
    double weight;
    	// Check inputs
    static char *kwlist[] = {"outputIndex", "hiddenIndex", "weight", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IId", kwlist,
                                     &i, &j, &weight)) {
		PyErr_Format(PyExc_ValueError, "Expected integers i, j and double weight.");
		return NULL;
	}
    // Make sure they are valid
    if (i >= self->net->getNumOfOutputs() ||
        j >= self->net->getNumOfHidden() ) {
      PyErr_Format(PyExc_ValueError, "Not valid values for i and j. Valid values are: i < numOfOutput, j < numOfHidden.");
      return NULL;
    }

    // They are valid, connect
    self->net->connectOToH(i, j, weight);

    // return None
    return Py_BuildValue("");

}
PyObject *FFNetwork_connectOToI(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
  unsigned int i, j;
  double weight;
  // Check inputs
  static char *kwlist[] = {"outputIndex", "inputIndex", "weight", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IId", kwlist,
                                   &i, &j, &weight)) {
    PyErr_Format(PyExc_ValueError, "Expected integers i, j and double weight.");
    return NULL;
  }
    // Make sure they are valid
    if (i >= self->net->getNumOfOutputs() ||
        j >= self->net->getNumOfInputs() ) {
      PyErr_Format(PyExc_ValueError, "Not valid values for i and j. Valid values are: i < numOfOutput, j < numOfInput");
      return NULL;
    }

    // They are valid, connect
    self->net->connectOToI(i, j, weight);

    // return None
    return Py_BuildValue("");

}
PyObject *FFNetwork_connectOToB(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
  unsigned int i;
    double weight;
    	// Check inputs
    static char *kwlist[] = {"outputIndex", "weight", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Id", kwlist,
					&i, &weight)) {
		PyErr_Format(PyExc_ValueError, "Expected integer i and double weight.");
		return NULL;
	}
    // Make sure they are valid
    if (i >= self->net->getNumOfOutputs() ) {
      PyErr_Format(PyExc_ValueError, "Not valid values for i. Valid values are: i < numOfOutputs");
      return NULL;
    }

    // They are valid, connect
    self->net->connectOToB(i, weight);

    // return None
    return Py_BuildValue("");

}

  PyObject *FFNetwork_setOutputActivationFunction(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
    int i;
    // Check inputs
    static char *kwlist[] = {"activationFunction", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist,
                                     &i)) {
      PyErr_Format(PyExc_ValueError, "Expected an integer value, check ann.ffnetwork.LINEAR etc");
      return NULL;
    }

    self->net->setOutputActivationFunction(i);

    // Return None
    return Py_BuildValue("");
  }

  PyObject *FFNetwork_setHiddenActivationFunction(PyFFNetwork *self, PyObject *args, PyObject *kwargs) {
    int i;
    // Check inputs
    static char *kwlist[] = {"activationFunction", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist,
                                     &i)) {
      PyErr_Format(PyExc_ValueError, "Expected an integer value, check ann.ffnetwork.LINEAR etc");
      return NULL;
    }

    self->net->setHiddenActivationFunction(i);

    // Return None
    return Py_BuildValue("");
  }


/*
 * Getters and setters
 */
PyObject *FFNetwork_getNumOfInputs(PyFFNetwork *self, void *closure) {
	return Py_BuildValue("I", self->net->getNumOfInputs());
}
PyObject *FFNetwork_getNumOfHidden(PyFFNetwork *self, void *closure) {
	return Py_BuildValue("I", self->net->getNumOfHidden());
}
PyObject *FFNetwork_getNumOfOutputs(PyFFNetwork *self, void *closure) {
	return Py_BuildValue("I", self->net->getNumOfOutputs());
}


}

