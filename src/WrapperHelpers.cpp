#include "WrapperHelpers.hpp"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C

bool isPythonList(PyObject *list)
{
  return PyList_CheckExact(list);
}

bool isNumpyArray(PyObject *list)
{
  return PyArray_Check(list);
}

// For Numpy Arrays.
bool isArrayTypeDouble(PyArrayObject *array)
{
  return PyArray_TYPE(array) == NPY_DOUBLE;
}

bool isArrayNDim(PyArrayObject *array, unsigned int n)
{
  return PyArray_NDIM(array) == n;
}

//   Supply methods for 1 and 2 dimensions only, as that's all I use
bool isArrayLength(PyArrayObject *array, unsigned int n)
{
  bool res = isArrayNDim(array, 1);
  if (res) {
    res = PyArray_DIM(array, 0) == n;
  }
  return res;
}

bool isArrayLength(PyArrayObject *array, unsigned int n, unsigned int m)
{
  bool res = isArrayNDim(array, 2);
  if (res) {
    res = PyArray_DIM(array, 0) == n;
  }
  if (res) {
    res = PyArray_DIM(array, 1) == m;
  }
  return res;
}

//   Assumes data is contiguous. Will use correct method depending on
//   dimension.
double *getArrayDataPtr(PyArrayObject *array)
{
  double *ptr = NULL;
  unsigned int ndim = PyArray_NDIM(array);
  if (ndim == 1) {
    ptr = (double *) PyArray_GETPTR1(array, 0);
  } else if (ndim == 2) {
    ptr = (double *) PyArray_GETPTR2(array, 0, 0);
  }
  return ptr;
}
