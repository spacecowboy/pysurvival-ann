#include "CoxCascadeNetworkWrapper.h"
// For convenience macros in python3
#include "PythonModule.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "CoxCascadeNetwork.h"

extern "C" {

  /*
   * Python init
   * -----------
   */
  int CoxCascadeNetwork_init(PyCoxCascadeNetwork *self, PyObject *args, PyObject *kwds) {
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
    self->super.super.super.net = new CoxCascadeNetwork(numOfInputs);

    if (self->super.super.super.net == NULL)
      return -1;

    self->super.super.super.net->initNodes();
    return 0;
  }

  /*
 * Getters and Setters
 */


}
