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
#include "RPropNetwork.h"

extern "C" {

/*
 * Python init
 * -----------
 */
int RPropNetwork_init(PyRPropNetwork *self, PyObject *args, PyObject *kwds) {
	 printf("RNetwork_init\n");
        static char *kwlist[] =
                        { "numOfInputs", "numOfHidden", "numOfOutputs", NULL };

         unsigned int numOfInputs, numOfHidden, numOfOutputs;
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist, &numOfInputs,
                        &numOfHidden, &numOfOutputs)) {
                PyErr_Format(PyExc_ValueError,
                                "Arguments should be (all mandatory positive integers): numOfInputs, numOfHidden, numOfOutputs");
                return -1;
        }
        printf("RNetwork_init: We are past the if: %d %d %d\n", numOfInputs, numOfHidden, numOfOutputs);

        self->super.net = new RPropNetwork(numOfInputs, numOfHidden, numOfOutputs);

        printf("RNetwork_init: net is: %d\n", self->super.net);
        if (self->super.net == NULL)
                return -1;

        self->super.net->initNodes();
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

PyObject *RPropNetwork_learn(PyRPropNetwork *self, PyObject *args, PyObject *kwargs) {
	printf("rNetwork_learn\n");

	PyObject *inputs = NULL;
	PyObject *targets = NULL;
	// Check inputs
    static char *kwlist[] = {"inputs", "targets", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
					&inputs, &targets)) {
		PyErr_Format(PyExc_ValueError, "Arguments should be: inputs (2d array), targets (2d array)");
		return NULL;
	}

	// Make sure they conform to required structure
	PyArrayObject *inputArray = NULL;
	PyArrayObject *targetArray = NULL;

	inputArray = (PyArrayObject *) PyArray_ContiguousFromObject(inputs, PyArray_DOUBLE, 2, 2);
	if (inputArray == NULL)
		return NULL;

	targetArray = (PyArrayObject *) PyArray_ContiguousFromObject(targets, PyArray_DOUBLE, 2, 2);
	if (targetArray == NULL)
		return NULL;

	// Objects were converted successfully. But make sure they are the same length!
	//int inputRows = inputArray->dimensions[0];
	//int targetRows = targetArray->dimensions[0];

	if (inputArray->dimensions[0] != targetArray->dimensions[0] ||
		inputArray->dimensions[1] != self->super.net->getNumOfInputs() ||
		targetArray->dimensions[1] != self->super.net->getNumOfOutputs()) {
		// Decrement, set error and return
		PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same number of rows. Also the columns must match number of input/output neurons respectively.");
		Py_DECREF(inputArray);
		Py_DECREF(targetArray);

		return NULL;
	}

	// Arguments are valid!

	printf("rNetwork_learn: past check\n");

	((RPropNetwork*)self->super.net)->learn((double *)inputArray->data, (double *)targetArray->data, inputArray->dimensions[0]);

	// Decrement counters for inputArray and targetArray
	Py_DECREF(inputArray);
	Py_DECREF(targetArray);

	// Return none
	return Py_BuildValue("");
}


/*
 * Getters and Setters
 */

  PyObject *RPropNetwork_getMaxEpochs(PyRPropNetwork *self, void *closure) {
    return Py_BuildValue("I", ((RPropNetwork*)self->super.net)->getMaxEpochs());
  }

  int RPropNetwork_setMaxEpochs(PyRPropNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((RPropNetwork*)self->super.net)->setMaxEpochs((unsigned int) val);
  }

  PyObject *RPropNetwork_getMaxError(PyRPropNetwork *self, void *closure) {
    return Py_BuildValue("d", ((RPropNetwork*)self->super.net)->getMaxError());
  }

  int RPropNetwork_setMaxError(PyRPropNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((RPropNetwork*)self->super.net)->setMaxError(val);
  }

  PyObject *RPropNetwork_getPrintEpoch(PyRPropNetwork *self, void *closure) {
    return Py_BuildValue("i", ((RPropNetwork*)self->super.net)->getPrintEpoch());
  }

  int RPropNetwork_setPrintEpoch(PyRPropNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((RPropNetwork*)self->super.net)->setPrintEpoch((int) val);
  }


}

