#include "Python.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include <algorithm>
#include "ErrorFunctions.hpp"
#include "WrapperHelpers.hpp"

extern "C" {
  PyObject *ErrorFuncs_getError(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
    int errorfunc;
    PyObject *targets = NULL;
    PyObject *outputs = NULL;
    // Check inputs
    static char *kwlist[] = {(char*)"errorfunc",
                             (char*)"targets",
                             (char*)"outputs",
                             NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOO", kwlist,
                                     &errorfunc, &targets, &outputs))
    {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be: errorfunc, targets , outputs");
      return NULL;
    }

    // Make sure they conform to required structure
    PyArrayObject *outputArray = NULL;
    PyArrayObject *targetArray = NULL;

    outputArray = (PyArrayObject *) PyArray_ContiguousFromObject(outputs,
                                                                 NPY_DOUBLE,
                                                                 1, 2);
    if (outputArray == NULL) {
      return NULL;
    }

    targetArray = (PyArrayObject *) PyArray_ContiguousFromObject(targets,
                                                                 NPY_DOUBLE,
                                                                 1, 2);
    if (targetArray == NULL) {
      Py_DECREF(outputArray);
      return NULL;
    }

    // Objects were converted successfully. But make sure they are the
    // same length and dimensions!

    int targetNDim = PyArray_NDIM(targetArray);
    int outputNDim = PyArray_NDIM(outputArray);
    int ndim = targetNDim;

    if (targetNDim != outputNDim)
    {
      // Decrement, set error and return
      PyErr_Format(PyExc_ValueError,
                   "Outputs and targets must have the same dimensions.");
      Py_DECREF(outputArray);
      Py_DECREF(targetArray);

      return NULL;
    }
    if (ndim > 2)
    {
      // Decrement, set error and return
      PyErr_Format(PyExc_ValueError,
                   "Can't be more than 2 dimensions!");
      Py_DECREF(outputArray);
      Py_DECREF(targetArray);

      return NULL;
    }
    // Check each individual dimension
    for (int d = 0; d < ndim; d++)
    {
      if (PyArray_DIM(outputArray, d) != PyArray_DIM(targetArray, d)) {
        // Decrement, set error and return
        PyErr_Format(PyExc_ValueError,
                     "Outputs and targets must have the same dimensions.");
        Py_DECREF(outputArray);
        Py_DECREF(targetArray);

        return NULL;
      }
    }

    // Arguments are valid!
    unsigned int rows = PyArray_DIM(outputArray, 0);
    unsigned int cols = 1;
    if (ndim > 1) {
      cols = PyArray_DIM(outputArray, 1);
    }
    unsigned int total = rows*cols;
    double errors[total];

    try {
      getAllErrors((ErrorFunction) errorfunc,
                   (double *)PyArray_DATA(targetArray),
                   rows,
                   cols,
                   (double *)PyArray_DATA(outputArray),
                   errors);
    } catch (const std::exception& ex) {
      PyErr_Format(PyExc_RuntimeError, "%s", ex.what());
      Py_DECREF(outputArray);
      Py_DECREF(targetArray);
      return NULL;
    }

    // Now convert to numpy array
    npy_intp dims[ndim];
        dims[0] = rows;
    if (ndim > 1)
    {
      dims[1] = cols;
    }
    PyObject *result = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    double *ptr = getArrayDataPtr((PyArrayObject*) result);

    std::copy(errors, errors + total,
              ptr);

    // Decrement counters for inputArray and targetArray
    Py_DECREF(outputArray);
    Py_DECREF(targetArray);

    return result;
  }


  PyObject *ErrorFuncs_getDeriv(PyObject *self,
                                PyObject *args,
                                PyObject *kwargs)
  {
    // Inputs
    // func, Y, length, numoutput, outputs
    int errorfunc;
    PyObject *targets = NULL;
    PyObject *outputs = NULL;
    // Check inputs
    static char *kwlist[] = {(char*)"errorfunc",        \
                             (char*)"targets",          \
                             (char*)"outputs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOO", kwlist,
                                     &errorfunc, &targets, &outputs))
    {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be: errorfunc, targets , outputs");
      return NULL;
    }

    // Make sure they conform to required structure
    PyArrayObject *outputArray = NULL;
    PyArrayObject *targetArray = NULL;

    outputArray = (PyArrayObject *)
      PyArray_ContiguousFromObject(outputs,
                                   NPY_DOUBLE, 1, 2);
    if (outputArray == NULL)
      return NULL;

    targetArray = (PyArrayObject *)
      PyArray_ContiguousFromObject(targets,
                                   NPY_DOUBLE, 1, 2);
    if (targetArray == NULL) {
      Py_DECREF(outputArray);
      return NULL;
    }

    // Objects were converted successfully. But make sure they are the
    // same length and dimensions!

    int targetNDim = PyArray_NDIM(targetArray);
    int outputNDim = PyArray_NDIM(outputArray);
    int ndim = targetNDim;

    if (targetNDim != outputNDim)
    {
      // Decrement, set error and return
      PyErr_Format(PyExc_ValueError,
                   "Outputs and targets must have the same dimensions.");
      Py_DECREF(outputArray);
      Py_DECREF(targetArray);

      return NULL;
    }
    if (ndim > 2)
    {
      // Decrement, set error and return
      PyErr_Format(PyExc_ValueError,
                   "Can't be more than 2 dimensions!");
      Py_DECREF(outputArray);
      Py_DECREF(targetArray);

      return NULL;
    }
    // Check each individual dimension
    for (int d = 0; d < ndim; d++)
    {
      if (PyArray_DIM(outputArray, d) != PyArray_DIM(targetArray, d)) {
        // Decrement, set error and return
        PyErr_Format(PyExc_ValueError,
                     "Outputs and targets must have the same dimensions.");
        Py_DECREF(outputArray);
        Py_DECREF(targetArray);

        return NULL;
      }
    }

    // Arguments are valid!
    unsigned int rows = PyArray_DIM(outputArray, 0);
    unsigned int cols = 1;
    if (ndim > 1)
    {
      cols = PyArray_DIM(outputArray, 1);
    }
    unsigned int total = rows*cols;
    double derivs[total];

    double *Ydata = (double *)PyArray_DATA(targetArray);
    double *outputdata = (double *)PyArray_DATA(outputArray);
    unsigned int index;

    try {
      for (unsigned int i = 0; i < rows; i++)
      {
        index = i * cols;
        getDerivative((ErrorFunction) errorfunc,
                      Ydata,
                      rows,
                      cols,
                      outputdata,
                      index,
                      derivs + index);
      }
    } catch (const std::exception& ex) {
      PyErr_Format(PyExc_RuntimeError, "%s", ex.what());
      Py_DECREF(outputArray);
      Py_DECREF(targetArray);
      return NULL;
    }

    // Now convert to numpy array
    npy_intp dims[ndim];
    dims[0] = rows;
    if (ndim > 1)
    {
      dims[1] = cols;
    }
    PyObject *result = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    double *ptr = getArrayDataPtr((PyArrayObject*) result);

    std::copy(derivs, derivs + total,
              ptr);

    // Decrement counters for inputArray and targetArray
    Py_DECREF(outputArray);
    Py_DECREF(targetArray);

    return result;
  }

}
