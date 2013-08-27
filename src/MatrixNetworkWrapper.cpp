#include "MatrixNetworkWrapper.hpp"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include <algorithm>

extern "C" {

  // Constructor
  PyObject *MatrixNetwork_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds)
  {
    PyMatrixNetwork *self = (PyMatrixNetwork*)type->tp_alloc(type, 0);
	return (PyObject *) self;
  }

  // Init
  int MatrixNetwork_init(PyMatrixNetwork *self,
                         PyObject *args, PyObject *kwds)
  {
    static char *kwlist[] =
      { (char*)"numOfInputs", (char*)"numOfHidden", \
        (char*)"numOfOutputs", NULL };

    unsigned int numOfInputs, numOfHidden, numOfOutputs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist,
                                     &numOfInputs,
                                     &numOfHidden, &numOfOutputs)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers):\
 numOfInputs, numOfHidden, numOfOutputs");
      return -1;
    }

	self->net = new MatrixNetwork(numOfInputs, numOfHidden, numOfOutputs);

	if (self->net == NULL)
      return -1;

	return 0;
  }

  // Destructor
  void MatrixNetwork_dealloc(PyMatrixNetwork *self)
  {
    delete self->net;
    Py_TYPE(self)->tp_free((PyObject*) self);
  }

  /*
   * Wrappers
   * ========
   */
  PyObject *MatrixNetwork_output(PyMatrixNetwork *self,
                                 PyObject *inputs)
  {
    if (!(PyList_CheckExact(inputs)
          || (PyArray_NDIM(inputs) == 1 &&
              PyArray_TYPE(inputs) == NPY_DOUBLE))) {
      PyErr_Format(PyExc_ValueError,
                   "The input does not seem to be of a suitable list type.\
 This method only accepts a python list or a 1D numpy array of doubles.");
      return NULL;
	}

	// First convert to normal double array

	bool pylist = PyList_CheckExact(inputs);
	PyObject *pyval = NULL;
	double *ptr = NULL;

	double dInputs[self->net->INPUT_COUNT];

    if (pylist) {
      for (int i = 0; (unsigned int)i < self->net->INPUT_COUNT; i++) {
        // Actual python list
        pyval = PyList_GetItem(inputs, i);
        if (pyval == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input\
 values. Possibly wrong length?");
          return NULL;
        }
        dInputs[i] = PyFloat_AsDouble(pyval);
      }
    } else {
      // Numpy array
      ptr = (double *) PyArray_GETPTR1((PyArrayObject*) inputs, 0);
      if (ptr == NULL) {
        PyErr_Format(PyExc_ValueError,
                     "Something went wrong when iterating of input \
 values. Possibly wrong length?");
        return NULL;
      }
      std::copy(ptr, ptr + self->net->INPUT_COUNT,
                dInputs);
    }

	// compute output
	double dResult[self->net->OUTPUT_COUNT];
	self->net->output(dInputs, dResult);

	// Now convert to numpy array
	npy_intp dims[1] = { (int) self->net->OUTPUT_COUNT };
	PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    ptr = (double *) PyArray_GETPTR1((PyArrayObject*) result, 0);

    std::copy(dResult, dResult + self->net->OUTPUT_COUNT,
              ptr);

	return result;
  }

  /*
   * Getters/Setters
   * ===============
   */
  PyObject *MatrixNetwork_getNumOfInputs(PyMatrixNetwork *self,
                                         void *closure) {
    return Py_BuildValue("I", self->net->INPUT_COUNT);
  }
  PyObject *MatrixNetwork_getNumOfHidden(PyMatrixNetwork *self,
                                         void *closure) {
    return Py_BuildValue("I", self->net->HIDDEN_COUNT);
  }
  PyObject *MatrixNetwork_getNumOfOutput(PyMatrixNetwork *self,
                                         void *closure) {
    return Py_BuildValue("I", self->net->OUTPUT_COUNT);
  }
  PyObject *MatrixNetwork_getOutputActivationFunction(PyMatrixNetwork *self,
                                                      void *closure) {
    return Py_BuildValue("I",
                         (int) self->net->getOutputActivationFunction());
  }
  PyObject *MatrixNetwork_getHiddenActivationFunction(PyMatrixNetwork *self,
                                                      void *closure) {
    return Py_BuildValue("I",
                         (int) self->net->getHiddenActivationFunction());
  }
  int MatrixNetwork_setOutputActivationFunction(PyMatrixNetwork *self,
                                                PyObject *value,
                                                void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete this you bonehead");
      return -1;
    }

    if (! PyInt_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Function must be an integer.\
 For example net.LOGSIG");
      return 1;
    }

    long i = PyInt_AsLong(value);

    self->net->setOutputActivationFunction((ActivationFuncEnum) i);

    return 0;
  }
  int MatrixNetwork_setHiddenActivationFunction(PyMatrixNetwork *self,
                                                PyObject *value,
                                                void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete this you bonehead");
      return -1;
    }

    if (! PyInt_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Function must be an integer.\
 For example net.LOGSIG");
      return 1;
    }

    long i = PyInt_AsLong(value);

    self->net->setHiddenActivationFunction((ActivationFuncEnum) i);

    return 0;
  }
  PyObject *MatrixNetwork_getLogPerf(PyMatrixNetwork *self,
                                     void *closure) {
    double *aLogPerf = self->net->getLogPerf();

    if (nullptr == aLogPerf || NULL == aLogPerf ||
        1 > self->net->getLogPerfLength()) {
      PyErr_Format(PyExc_ValueError,
                   "You need to train first!");
      return NULL;
    }

	// Now convert to numpy array
    int outNeurons = (int) self->net->OUTPUT_COUNT;
    int epochs = (int) self->net->getLogPerfLength() / outNeurons;
	npy_intp dims[2] = { epochs , outNeurons };
	PyObject *result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *ptr = (double *) PyArray_GETPTR2((PyArrayObject*) result,
                                             0, 0);

    std::copy(aLogPerf, aLogPerf + self->net->getLogPerfLength(),
              ptr);

	return result;
  }


}
