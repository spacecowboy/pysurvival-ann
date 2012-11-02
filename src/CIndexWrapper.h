/*
 * FFNetworkWrapper.h
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#ifndef CINDEXWRAPPER_H_
#define CINDEXWRAPPER_H_

#include "Python.h"

extern "C" {

  /*
   * Wrapper methods
   * ===============
   */
  PyObject *CIndex_getCindex(PyObject *self, PyObject *args, PyObject *kwargs);

}
#endif /* CINDEXWRAPPER_H_ */
