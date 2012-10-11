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
	PyFFNetwork *self = (PyFFNetwork*)type->tp_alloc(type, 0);

	return (PyObject *) self;
}

/*
 * Python init
 * -----------
 */
int FFNetwork_init(PyFFNetwork *self, PyObject *args, PyObject *kwds) {
	static char *kwlist[] =
                        { "numOfInputs", "numOfHidden", "numOfOutputs", NULL };

	 unsigned int numOfInputs, numOfHidden, numOfOutputs;
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist, &numOfInputs,
                        &numOfHidden, &numOfOutputs)) {
                PyErr_Format(PyExc_ValueError,
                                "Arguments should be (all mandatory positive integers): numOfInputs, numOfHidden, numOfOutputs");
                return -1;
        }

	self->net = new FFNetwork(numOfInputs, numOfHidden, numOfOutputs);

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
	delete self->net;
	self->ob_type->tp_free((PyObject*) self);
}

/*
 * Wrapper methods
 * ===============
 */

PyObject *FFNetwork_output(PyFFNetwork *self, PyObject *inputs) {
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

	double dInputs[self->net->getNumOfInputs()];
	for (int i = 0; i < self->net->getNumOfInputs(); i++) {
		if (pylist) {
			// Actual python list
			pyval = PyList_GetItem(inputs, i);
			if (pyval == NULL) {
				PyErr_Format(PyExc_ValueError,
						"Something went wrong when iterating of input values. Possibly wrong length?");
				return NULL;
			}
			dInputs[i] = PyFloat_AsDouble(pyval);
		} else {
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

	// compute output
	double dResult[self->net->getNumOfOutputs()];
	self->net->output(dInputs, dResult);

	// Now convert to numpy array
	npy_intp dims[1] = { self->net->getNumOfOutputs() };
	PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

	for (int i = 0; i < self->net->getNumOfOutputs(); i++) {
		ptr = (double *) PyArray_GETPTR1((PyArrayObject*) result, i);
		*ptr = dResult[i];
	}

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

