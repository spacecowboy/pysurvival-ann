#include "CascadeNetworkWrapper.h"
// For convenience macros in python3
#include "PythonModule.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "CascadeNetwork.h"

extern "C" {

  /*
   * Python init
   * -----------
   */
  int CascadeNetwork_init(PyCascadeNetwork *self, PyObject *args, PyObject *kwds) {
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
    self->super.super.net = new CascadeNetwork(numOfInputs, 1);

    if (self->super.super.net == NULL)
      return -1;

    self->super.super.net->initNodes();
    return 0;
  }

  /*
 * Getters and Setters
 */
  PyObject *CascadeNetwork_getMaxHidden(PyCascadeNetwork *self, void *closure) {
    return Py_BuildValue("I",
                         ((CascadeNetwork*)self->super.super.net)->getMaxHidden());
  }
  int CascadeNetwork_setMaxHidden(PyCascadeNetwork *self, PyObject *value,
                                  void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((CascadeNetwork*)self->super.super.net)->setMaxHidden((unsigned int) val);
    return 0;
  }

  PyObject *CascadeNetwork_getMaxHiddenEpochs(PyCascadeNetwork *self, void *closure) {
    return Py_BuildValue("I",
                         ((CascadeNetwork*)self->super.super.net)->getMaxHiddenEpochs());
  }
  int CascadeNetwork_setMaxHiddenEpochs(PyCascadeNetwork *self, PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((CascadeNetwork*)self->super.super.net)->setMaxHiddenEpochs((unsigned int) val);
    return 0;
  }


}
