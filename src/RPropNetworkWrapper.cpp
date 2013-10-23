/*
 * FFNetworkWrapper.cpp
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#include "RPropNetworkWrapper.hpp"
// For convenience macros in python3
#include "PythonModule.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "RPropNetwork.hpp"
#include "ErrorFunctions.hpp"

extern "C" {

  /*
   * Static variables
   * Set some constants in the objects dictionary
   * dict refers to NetworkType.tp_dict
   */
  void setRPropConstants(PyObject *dict) {
    // Error function
    PyDict_SetItemString(dict, "ERROR_MSE",
                         Py_BuildValue("i", ErrorFunction::ERROR_MSE));
    PyDict_SetItemString(dict, "ERROR_SURV_MSE",
                         Py_BuildValue("i", ErrorFunction::ERROR_SURV_MSE));
    PyDict_SetItemString(dict, "ERROR_SURV_LIKELIHOOD",
                         Py_BuildValue("i",
                                       ErrorFunction::ERROR_SURV_LIKELIHOOD));
  }


  /*
   * Python init
   * -----------
   */
  int RPropNetwork_init(PyRPropNetwork *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = { (char*)"numOfInputs",
                              (char*)"numOfHidden",
                              (char*)"numOfOutputs",
                              NULL };

    unsigned int numOfInputs, numOfHidden, numOfOutputs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist, &numOfInputs,
                                     &numOfHidden, &numOfOutputs)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers):\
 numOfInputs, numOfHidden, numOfOutputs");
      return -1;
    }

    self->super.net = new RPropNetwork(numOfInputs, numOfHidden, numOfOutputs);

    if (self->super.net == NULL)
      return -1;

    return 0;
  }

  /*
   * Wrapper methods
   * ===============
   */

  PyObject *RPropNetwork_learn(PyRPropNetwork *self, PyObject *args,
                               PyObject *kwargs) {
    PyObject *inputs = NULL;
    PyObject *targets = NULL;
    // Check inputs
    static char *kwlist[] = {(char*)"inputs", (char*)"targets", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
                                     &inputs, &targets)) {
      PyErr_Format(PyExc_ValueError, "Arguments should be: inputs (2d array), \
targets (2d array)");
      return NULL;
    }

    // Make sure they conform to required structure
    PyArrayObject *inputArray = NULL;
    PyArrayObject *targetArray = NULL;

    inputArray =
      (PyArrayObject *) PyArray_ContiguousFromObject(inputs,
                                                     PyArray_DOUBLE, 2, 2);
    if (inputArray == NULL)
      return NULL;

    targetArray =
      (PyArrayObject *) PyArray_ContiguousFromObject(targets,
                                                     PyArray_DOUBLE, 2, 2);
    if (targetArray == NULL) {
      Py_DECREF(inputArray);
      return NULL;
    }

    // Objects were converted successfully.But make sure they are the
    // same length!

    if (inputArray->dimensions[0] != targetArray->dimensions[0] ||
        (unsigned int)inputArray->dimensions[1] != self->super.net->INPUT_COUNT ||
        (unsigned int)targetArray->dimensions[1] != self->super.net->OUTPUT_COUNT)
      {
        // Decrement, set error and return
        PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same \
number of rows. Also the columns must match number of input/output neurons \
respectively.");
        Py_DECREF(inputArray);
        Py_DECREF(targetArray);

        return NULL;
      }

    // Arguments are valid!

    ((RPropNetwork*)self->super.net)->learn((double *)inputArray->data,
                                            (double *)targetArray->data,
                                            inputArray->dimensions[0]);

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

  int RPropNetwork_setMaxEpochs(PyRPropNetwork *self,
                                PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    if (val < 0) {
      PyErr_SetString(PyExc_ValueError, "Epoch must be a positive number!");
      return -1;
    }

    ((RPropNetwork*)self->super.net)->setMaxEpochs((unsigned int) val);
    return 0;
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
    return 0;
  }

  PyObject *RPropNetwork_getErrorFunction(PyRPropNetwork *self, void *closure) {
    return Py_BuildValue("i", ((RPropNetwork*)self->super.net)->
                         getErrorFunction());
  }

  int RPropNetwork_setErrorFunction(PyRPropNetwork *self,
                                    PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((RPropNetwork*)self->super.net)->setErrorFunction((ErrorFunction) val);
    return 0;
  }

}
