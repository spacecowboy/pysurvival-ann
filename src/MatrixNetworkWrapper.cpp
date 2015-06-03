#include "MatrixNetworkWrapper.hpp"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include <algorithm>
#include "activationfunctions.hpp"
#include <stdio.h>

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
      { (char*)"inputcount", (char*)"hiddencount", \
        (char*)"outputcount", NULL };

    unsigned int numOfInputs, numOfHidden, numOfOutputs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist,
                                     &numOfInputs,
                                     &numOfHidden, &numOfOutputs)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers):\
 inputcount, hiddencount, outputcount");
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
          || (PyArray_NDIM((PyArrayObject *)inputs) == 1 &&
              PyArray_TYPE((PyArrayObject *)inputs) == NPY_DOUBLE))) {
      PyErr_Format(PyExc_ValueError,
                   "The input does not seem to be of a suitable list type.\
 This method only accepts a python list or a 1D numpy array of doubles.");
      return NULL;
    }

    // First convert to normal double array

    bool pylist = PyList_CheckExact(inputs);
    PyObject *pyval = NULL;
    double *ptr = NULL;

    std::vector<double> vInputs(self->net->INPUT_COUNT, 0.0);
    std::vector<double> vResult(self->net->OUTPUT_COUNT, 0.0);

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
        vInputs.at(i) = PyFloat_AsDouble(pyval);
      }
    } else {
      // Numpy array
      for (int i = 0; (unsigned int)i < self->net->INPUT_COUNT; i++) {
        ptr = (double *) PyArray_GETPTR1((PyArrayObject*) inputs, i);
        if (ptr == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input \
 values. Possibly wrong length?");
          return NULL;
        }
        vInputs.at(i) = *ptr;
      }
    }

    // compute output
    self->net->output(vInputs.begin(), vResult.begin());

    // Now convert to numpy array
    npy_intp dims[1] = { (int) self->net->OUTPUT_COUNT };
    PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    ptr = (double *) PyArray_GETPTR1((PyArrayObject*) result, 0);

    std::copy(vResult.begin(), vResult.end(),
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

    self->net->setOutputActivationFunction(getFuncFromNumber((int) i));

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

    self->net->setHiddenActivationFunction(getFuncFromNumber((int) i));

    return 0;
  }
  PyObject *MatrixNetwork_getLogPerf(PyMatrixNetwork *self,
                                     void *closure) {
    std::vector<double> aLogPerf = self->net->aLogPerf;

    if (1 > aLogPerf.size()) {
      PyErr_Format(PyExc_ValueError,
                   "You need to train first!");
      return NULL;
    }

    // Now convert to numpy array
    int outNeurons = (int) self->net->OUTPUT_COUNT;
    int epochs = (int) aLogPerf.size() / outNeurons;
    npy_intp dims[2] = { epochs , outNeurons };
    PyObject *result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *ptr = (double *) PyArray_GETPTR2((PyArrayObject*) result,
                                             0, 0);

    std::copy(aLogPerf.begin(), aLogPerf.end(), ptr);

    return result;
  }

  PyObject *MatrixNetwork_getWeights(PyMatrixNetwork *self,
                                     void *closure) {
    std::vector<double> weights = self->net->weights;
    unsigned int length = weights.size();

    if (length == 0) {
      PyErr_Format(PyExc_ValueError,
                   "No weights");
      return NULL;
    }

    // Now convert to numpy array
    npy_intp dims[1] = { (int) length };
    PyObject *result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *ptr = (double *) PyArray_GETPTR1((PyArrayObject*) result,
                                             0);

    std::copy(weights.begin(), weights.end(),
              ptr);

    return result;
  }

  int MatrixNetwork_setWeights(PyMatrixNetwork *self,
                               PyObject *inputs,
                               void *closure) {
    if (inputs == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete this you bonehead");
      return -1;
    }

    if (!(PyList_CheckExact(inputs)
          || (PyArray_NDIM((PyArrayObject *)inputs) == 1 &&
              PyArray_TYPE((PyArrayObject *)inputs) == NPY_DOUBLE))) {
      PyErr_Format(PyExc_ValueError,
                   "The input does not seem to be of a suitable list type.\
 This method only accepts a python list or a 1D numpy array of doubles.");
      return -1;
    }

    // First convert to normal double array
    unsigned int length = self->net->weights.size();
    bool pylist = PyList_CheckExact(inputs);
    PyObject *pyval = NULL;
    double *ptr = NULL;

    for (int i = 0; (unsigned int)i < length; i++) {
      if (pylist) {
        // Actual python list
        pyval = PyList_GetItem(inputs, i);
        if (pyval == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input\
 values. Possibly wrong length?");
          return -1;
        }
        self->net->weights.at(i) = PyFloat_AsDouble(pyval);
      } else {
        ptr = (double *) PyArray_GETPTR1((PyArrayObject*) inputs, i);
        if (ptr == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input \
 values. Possibly wrong length?");
          return -1;
        }
        // OK, correct length
        self->net->weights.at(i) = *ptr;
      }
    }

    return 0;
  }

  // Conns
  PyObject *MatrixNetwork_getConns(PyMatrixNetwork *self,
                                   void *closure) {
    unsigned int length = self->net->conns.size();
    if (length == 0) {
      PyErr_Format(PyExc_ValueError,
                   "Zero sized conns!");
      return NULL;
    }

    // Now convert to numpy array
    npy_intp dims[1] = { (npy_int) length };
    PyObject *result = PyArray_SimpleNew(1, dims, NPY_UINT);

    std::copy(self->net->conns.begin(), self->net->conns.end(),
              (unsigned int *) PyArray_GETPTR1((PyArrayObject*) result, 0));

    return result;
  }

  int MatrixNetwork_setConns(PyMatrixNetwork *self,
                               PyObject *inputs,
                               void *closure) {
    if (inputs == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete this you bonehead");
      return -1;
    }

    if (!(PyList_CheckExact(inputs)
          || (PyArray_NDIM((PyArrayObject *)inputs) == 1 &&
              PyArray_TYPE((PyArrayObject *)inputs) == NPY_UINT))) {
      PyErr_Format(PyExc_ValueError,
                   "The input does not seem to be of a suitable list type.\
 This method only accepts a python list or a 1D numpy array of UINTs.");
      return -1;
    }

    // First convert to normal double array
    unsigned int length = self->net->conns.size();
    bool pylist = PyList_CheckExact(inputs);
    PyObject *pyval = NULL;
    unsigned int *ptr = NULL;

    if (pylist) {
      for (int i = 0; (unsigned int) i < length; i++) {
        // Actual python list
        pyval = PyList_GetItem(inputs, i);
        if (pyval == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input\
 values. Possibly wrong length?");
          return -1;
        }
        self->net->conns.at(i) = PyLong_AsLong(pyval);
      }
    } else {
      for (int i = 0; (unsigned int) i < length; i++) {
        ptr = (unsigned int*) PyArray_GETPTR1((PyArrayObject*) inputs, i);
        if (ptr == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input values.");
          return -1;
        } else {
          self->net->conns.at(i) = *ptr;
        }
      }
    }

    return 0;
  }

  // ActFuncs
  PyObject *MatrixNetwork_getActFuncs(PyMatrixNetwork *self,
                                      void *closure) {
    unsigned int length = self->net->actFuncs.size();

    if (0 == length) {
      PyErr_Format(PyExc_ValueError,
                   "ActFuncs were zero sized!");
      return NULL;
    }

    // Now convert to numpy array
    npy_intp dims[1] = { (int) length };
    PyObject *result = PyArray_SimpleNew(1, dims, NPY_UINT);

    std::copy(self->net->actFuncs.begin(), self->net->actFuncs.end(),
              (unsigned int *) PyArray_GETPTR1((PyArrayObject*) result, 0));

    return result;
  }

  int MatrixNetwork_setActFuncs(PyMatrixNetwork *self,
                                PyObject *inputs,
                                void *closure) {
    if (inputs == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete this you bonehead");
      return -1;
    }

    if (!(PyList_CheckExact(inputs)
          || (PyArray_NDIM((PyArrayObject *)inputs) == 1  &&
              PyArray_TYPE((PyArrayObject *)inputs) == NPY_UINT))) {
      PyErr_Format(PyExc_ValueError,
                   "The input does not seem to be of a suitable list type.\
 This method only accepts a python list or a 1D numpy array of UINTs.");
      return -1;
    }

    // First convert to normal double array
    unsigned int length = self->net->actFuncs.size();
    bool pylist = PyList_CheckExact(inputs);
    PyObject *pyval = NULL;
    unsigned int *ptr = NULL;

    for (int i = 0; (unsigned int)i < length; i++) {
      if (pylist) {
        // Actual python list
        pyval = PyList_GetItem(inputs, i);
        if (pyval == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input\
 values. Possibly wrong length?");
          return -1;
        }
        self->net->actFuncs.at(i) = getFuncFromNumber(PyLong_AsLong(pyval));
      } else {
        ptr = (unsigned int *) PyArray_GETPTR1((PyArrayObject *) inputs, i);
        if (ptr == NULL) {
          PyErr_Format(PyExc_ValueError,
                       "Something went wrong when iterating of input \
 values. Possibly wrong length?");
          return -1;
        }
        self->net->actFuncs.at(i) = getFuncFromNumber(*ptr);
      }
    }

    return 0;
  }


  PyObject *MatrixNetwork_getInputDropoutProb(PyMatrixNetwork *self,
                                              void *closure) {
    return Py_BuildValue("d", self->net->inputDropoutProb);
  }
  int MatrixNetwork_setInputDropoutProb(PyMatrixNetwork *self, PyObject *value,
                                        void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    self->net->inputDropoutProb = val;
    return 0;
  }

  PyObject *MatrixNetwork_getHiddenDropoutProb(PyMatrixNetwork *self,
                                              void *closure) {
    return Py_BuildValue("d", self->net->hiddenDropoutProb);
  }
  int MatrixNetwork_setHiddenDropoutProb(PyMatrixNetwork *self, PyObject *value,
                                         void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    self->net->hiddenDropoutProb = val;
    return 0;
  }

}
