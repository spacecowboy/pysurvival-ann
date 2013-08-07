
#include "GeneticSurvivalMSENetworkWrapper.h"
// For convenience macros in python3
#include "PythonModule.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "GeneticSurvivalMSENetwork.h"
#include <stdio.h>

extern "C" {

  /*
   * Python init
   * -----------
   */
  int GenSurvMSENetwork_init(PyGenSurvMSENetwork *self, PyObject *args,
                          PyObject *kwds) {
    static char *kwlist[] = { (char*)"numOfInputs",  \
                              (char*)"numOfHidden", NULL };

    unsigned int numOfInputs, numOfHidden;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "II", kwlist, &numOfInputs,
                                     &numOfHidden)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers): \
numOfInputs, numOfHidden");
      return -1;
    }

    self->super.super.net = new GeneticSurvivalMSENetwork(numOfInputs, numOfHidden);

    if (self->super.super.net == NULL)
      return -1;

    self->super.super.net->initNodes();
    return 0;
  }

  /*
   * Wrapper methods
   * ===============
   */

  PyObject *GenSurvMSENetwork_learn(PyGenSurvMSENetwork *self, PyObject *args, \
                                 PyObject *kwargs) {
	PyObject *inputs = NULL;
	PyObject *targets = NULL;
	// Check inputs
    static char *kwlist[] = {(char*)"inputs", \
                             (char*)"targets", NULL};
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
	if (targetArray == NULL) {
      Py_DECREF(inputArray);
      return NULL;
    }

	// Objects were converted successfully. But make sure they are the same length!

	if (inputArray->dimensions[0] != targetArray->dimensions[0] ||
		(unsigned int)inputArray->dimensions[1] != self->super.super.net->getNumOfInputs() ||
		targetArray->dimensions[1] != 2) {
      // Decrement, set error and return
		PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same number of rows. Also the columns must match number of input neurons and target data must have 2 columns (time, event).");
		Py_DECREF(inputArray);
		Py_DECREF(targetArray);

		return NULL;
	}

	// Arguments are valid!
    // Release the GIL
    Py_BEGIN_ALLOW_THREADS;
	((GeneticSurvivalMSENetwork*)self->super.super.net)->learn((double *)inputArray->data, (double *)targetArray->data, inputArray->dimensions[0]);

    // Acquire the GIL again
    Py_END_ALLOW_THREADS;

	// Decrement counters for inputArray and targetArray
	Py_DECREF(inputArray);
	Py_DECREF(targetArray);

	// Return none
	return Py_BuildValue("");
  }
}
