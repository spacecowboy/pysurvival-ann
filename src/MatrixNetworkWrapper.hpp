#ifndef MATRIXNETWORKWRAPPER_HPP_
#define MATRIXNETWORKWRAPPER_HPP_

#include "PythonModule.h"
#include "structmember.h" // used to declare member list
#include "MatrixNetwork.hpp"

extern "C" {

  // Python Object
  typedef struct {
    PyObject_HEAD // Inherit from PyObject
    MatrixNetwork *net; // actual c++ network
  } PyMatrixNetwork;

  // Constructor
  PyObject *MatrixNetwork_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds);

  // Init
  int MatrixNetwork_init(PyMatrixNetwork *self,
                         PyObject *args, PyObject *kwds);

  // Destructor
  void MatrixNetwork_dealloc(PyMatrixNetwork *self);

  /*
   * Wrappers
   * ========
   */
  PyObject *MatrixNetwork_output(PyMatrixNetwork *self,
                                 PyObject *inputs);

  /*
   * Getters/Setters
   * ===============
   */
  PyObject *MatrixNetwork_getNumOfInputs(PyMatrixNetwork *self,
                                         void *closure);
  PyObject *MatrixNetwork_getNumOfHidden(PyMatrixNetwork *self,
                                         void *closure);
  PyObject *MatrixNetwork_getNumOfOutput(PyMatrixNetwork *self,
                                         void *closure);
  PyObject *MatrixNetwork_getOutputActivationFunction(PyMatrixNetwork *self,
                                                      void *closure);
  PyObject *MatrixNetwork_getHiddenActivationFunction(PyMatrixNetwork *self,
                                                      void *closure);
  int MatrixNetwork_setOutputActivationFunction(PyMatrixNetwork *self,
                                                PyObject *value,
                                                void *closure);
  int MatrixNetwork_setHiddenActivationFunction(PyMatrixNetwork *self,
                                                PyObject *value,
                                                void *closure);
  PyObject *MatrixNetwork_getLogPerf(PyMatrixNetwork *self,
                                     void *closure);

} // extern "C"

#endif //MATRIXNETWORKWRAPPER_H_
