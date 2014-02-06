#ifndef _ERRORFUNCTIONSWRAPPER_H_
#define _ERRORFUNCTIONSWRAPPER_H_

#include "Python.h"

extern "C" {

  PyObject *ErrorFuncs_getError(PyObject *self,
                                PyObject *args,
                                PyObject *kwargs);


  PyObject *ErrorFuncs_getDeriv(PyObject *self,
                                PyObject *args,
                                PyObject *kwargs);
}

#endif /* _ERRORFUNCTIONSWRAPPER_H_ */
