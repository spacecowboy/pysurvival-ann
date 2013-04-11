#include "Python.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "CIndexWrapper.h"
#include "c_index.h"

extern "C" {
  PyObject *CIndex_getCindex(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *outputs = NULL;
    PyObject *targets = NULL;
    // Check inputs
    static char *kwlist[] = {(char*)"targets", \
                             (char*)"outputs", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
                                     &targets, &outputs)) {
      PyErr_Format(PyExc_ValueError, "Arguments should be: targets (2d array), outputs (2d array)");
      return NULL;
	}

	// Make sure they conform to required structure
	PyArrayObject *outputArray = NULL;
	PyArrayObject *targetArray = NULL;

	outputArray = (PyArrayObject *) PyArray_ContiguousFromObject(outputs, PyArray_DOUBLE, 1, 1);
	if (outputArray == NULL)
      return NULL;

	targetArray = (PyArrayObject *) PyArray_ContiguousFromObject(targets, PyArray_DOUBLE, 2, 2);
	if (targetArray == NULL) {
      Py_DECREF(outputArray);
      return NULL;
    }

	// Objects were converted successfully. But make sure they are the same length!

	if (outputArray->dimensions[0] != targetArray->dimensions[0] ||
		targetArray->dimensions[1] != 2) {
      // Decrement, set error and return
		PyErr_Format(PyExc_ValueError, "Outputs and targets must have the same number of rows. Also the target data must have 2 columns (time, event).");
		Py_DECREF(outputArray);
		Py_DECREF(targetArray);

		return NULL;
	}

	// Arguments are valid!
	double cindex = get_C_index((double *)outputArray->data, (double *)targetArray->data, outputArray->dimensions[0]);

	// Decrement counters for inputArray and targetArray
	Py_DECREF(outputArray);
	Py_DECREF(targetArray);

	// Return none
	return Py_BuildValue("d", cindex);
  }
}
