/*
 * FFNetworkWrapper.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "GeneticSurvivalNetworkWrapper.h"
// Must include this befor arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "GeneticSurvivalNetwork.h"

extern "C" {

/*
 * Python init
 * -----------
 */
  int GenSurvNetwork_init(PyGenSurvNetwork *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = { "numOfInputs", "numOfHidden", NULL };

    unsigned int numOfInputs, numOfHidden;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "II", kwlist, &numOfInputs,
                                     &numOfHidden)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers): numOfInputs, numOfHidden");
      return -1;
    }

    self->super.net = new GeneticSurvivalNetwork(numOfInputs, numOfHidden);

    if (self->super.net == NULL)
      return -1;

    self->super.net->initNodes();
    return 0;
}

/*
 * Wrapper methods
 * ===============
 */

  PyObject *GenSurvNetwork_learn(PyGenSurvNetwork *self, PyObject *args, PyObject *kwargs) {

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

	if (inputArray->dimensions[0] != targetArray->dimensions[0] ||
		inputArray->dimensions[1] != self->super.net->getNumOfInputs() ||
		targetArray->dimensions[1] != 2) {
      // Decrement, set error and return
		PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same number of rows. Also the columns must match number of input neurons and target data must have 2 columns (time, event).");
		Py_DECREF(inputArray);
		Py_DECREF(targetArray);

		return NULL;
	}

	// Arguments are valid!

	((GeneticSurvivalNetwork*)self->super.net)->learn((double *)inputArray->data, (double *)targetArray->data, inputArray->dimensions[0]);

	// Decrement counters for inputArray and targetArray
	Py_DECREF(inputArray);
	Py_DECREF(targetArray);

	// Return none
	return Py_BuildValue("");
  }


  /*
   * Getters and Setters
   */
  PyObject *GenSurvNetwork_getGenerations(PyGenSurvNetwork *self, void *closure) {
    return Py_BuildValue("I", ((GeneticSurvivalNetwork*)self->super.net)->getGenerations());
  }

  int GenSurvNetwork_setGenerations(PyGenSurvNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticSurvivalNetwork*)self->super.net)->setGenerations((unsigned int) val);
    return 0;
  }

  PyObject *GenSurvNetwork_getPopulationSize(PyGenSurvNetwork *self, void *closure){
    return Py_BuildValue("I", ((GeneticSurvivalNetwork*)self->super.net)->getPopulationSize());
  }

  int GenSurvNetwork_setPopulationSize(PyGenSurvNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticSurvivalNetwork*)self->super.net)->setGenerations((unsigned int) val);
    return 0;
  }

  PyObject *GenSurvNetwork_getWeightMutationChance(PyGenSurvNetwork *self, void *closure) {
    return Py_BuildValue("d", ((GeneticSurvivalNetwork*)self->super.net)->getWeightMutationChance());
  }

  int GenSurvNetwork_setWeightMutationChance(PyGenSurvNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticSurvivalNetwork*)self->super.net)->setWeightMutationChance(val);
    return 0;
  }

  PyObject *GenSurvNetwork_getWeightMutationHalfPoint(PyGenSurvNetwork *self, void *closure) {
    return Py_BuildValue("I", ((GeneticSurvivalNetwork*)self->super.net)->getWeightMutationHalfPoint());
  }

  int GenSurvNetwork_setWeightMutationHalfPoint(PyGenSurvNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticSurvivalNetwork*)self->super.net)->setWeightMutationHalfPoint((unsigned int) val);
    return 0;
  }

  PyObject *GenSurvNetwork_getWeightMutationStdDev(PyGenSurvNetwork *self, void *closure) {
    return Py_BuildValue("d", ((GeneticSurvivalNetwork*)self->super.net)->getWeightMutationStdDev());
  }

  int GenSurvNetwork_setWeightMutationStdDev(PyGenSurvNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticSurvivalNetwork*)self->super.net)->setWeightMutationStdDev(val);
    return 0;
  }

}

