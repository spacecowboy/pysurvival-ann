#include "GeneticCascadeNetworkWrapper.h"
// For convenience macros in python3
#include "PythonModule.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "GeneticCascadeNetwork.h"

extern "C" {

  /*
   * Python init
   * -----------
   */
  int GeneticCascadeNetwork_init(PyGeneticCascadeNetwork *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] =
      { (char*)"numOfInputs", NULL };

    unsigned int numOfInputs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I", kwlist, &numOfInputs)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers): \
numOfInputs");
      return -1;
    }

    // self->pyrprop->pyffnet.ffnet
    self->super.super.super.net = new GeneticCascadeNetwork(numOfInputs);

    if (self->super.super.super.net == NULL)
      return -1;

    self->super.super.super.net->initNodes();
    return 0;
  }


/*
 * Wrapper methods
 * ===============
 */

PyObject *GeneticCascadeNetwork_learn(PyGeneticCascadeNetwork *self, PyObject *args, PyObject *kwargs) {
  PyObject *inputs = NULL;
  PyObject *targets = NULL;
  // Check inputs
  static char *kwlist[] = {(char*)"inputs", (char*)"targets", NULL};
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
      (unsigned int)inputArray->dimensions[1] != self->super.super.super.net->getNumOfInputs() ||
      (unsigned int)targetArray->dimensions[1] != 2) {
    // Decrement, set error and return
    PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same number of rows. Also the columns must match number of input neurons. For the target data, the columns must be 2 (time, event), in that order.");
    Py_DECREF(inputArray);
    Py_DECREF(targetArray);

    return NULL;
  }

  // Arguments are valid!

  ((GeneticCascadeNetwork*)self->super.super.super.net)->learn((double *)inputArray->data, (double *)targetArray->data, inputArray->dimensions[0]);

  // Decrement counters for inputArray and targetArray
  Py_DECREF(inputArray);
  Py_DECREF(targetArray);

  // Return none
  return Py_BuildValue("");
}




  /*
 * Getters and Setters
 */



  /*
    ------------
    Ladder Network
    --------------
  */

  /*
   * Python init
   * -----------
   */
  int GeneticLadderNetwork_init(PyGeneticLadderNetwork *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] =
      { (char*)"numOfInputs", NULL };

    unsigned int numOfInputs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I", kwlist, &numOfInputs)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers): \
numOfInputs");
      return -1;
    }

    // self->pyrprop->pyffnet.ffnet
    self->super.super.super.super.net = new GeneticLadderNetwork(numOfInputs);

    if (self->super.super.super.super.net == NULL)
      return -1;

    self->super.super.super.super.net->initNodes();
    return 0;
  }


/*
 * Wrapper methods
 * ===============
 */

PyObject *GeneticLadderNetwork_learn(PyGeneticLadderNetwork *self, PyObject *args, PyObject *kwargs) {
  PyObject *inputs = NULL;
  PyObject *targets = NULL;
  // Check inputs
  static char *kwlist[] = {(char*)"inputs", (char*)"targets", NULL};
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
      (unsigned int)inputArray->dimensions[1] != self->super.super.super.super.net->getNumOfInputs() ||
      (unsigned int)targetArray->dimensions[1] != 2) {
    // Decrement, set error and return
    PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same number of rows. Also the columns must match number of input neurons. For the target data, the columns must be 2 (time, event), in that order.");
    Py_DECREF(inputArray);
    Py_DECREF(targetArray);

    return NULL;
  }

  // Arguments are valid!

  ((GeneticLadderNetwork*)self->super.super.super.super.net)->learn((double *)inputArray->data, (double *)targetArray->data, inputArray->dimensions[0]);

  // Decrement counters for inputArray and targetArray
  Py_DECREF(inputArray);
  Py_DECREF(targetArray);

  // Return none
  return Py_BuildValue("");
}




  /*
 * Getters and Setters
 */


}
