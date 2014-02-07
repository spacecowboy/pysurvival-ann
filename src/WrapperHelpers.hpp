#ifndef _WRAPPERHELPERS_H_
#define _WRAPPERHELPERS_H_

#include "PythonModule.h"
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C

bool isPythonList(PyObject *list);
bool isNumpyArray(PyObject *list);

// For Numpy Arrays.
bool isArrayTypeDouble(PyArrayObject *array);
bool isArrayNDim(PyArrayObject *array, unsigned int n);
//   Supply methods for 1 and 2 dimensions only, as that's all I use
bool isArrayLength(PyArrayObject *array, unsigned int n);
bool isArrayLength(PyArrayObject *array, unsigned int n, unsigned int m);

//   Assumes data is contiguous. Will use correct method depending on
//   dimension.
double *getArrayDataPtr(PyArrayObject *array);

// For Python Lists


#endif /* _WRAPPERHELPERS_H_ */
