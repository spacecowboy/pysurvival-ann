/*
 * RPropNetworkWrapper.h
 *
 *  Created on: 29 okt 2013
 *      Author: Jonas Kalderstam
 */

#ifndef RPROPNETWORKWRAPPER_HPP_
#define RPROPNETWORKWRAPPER_HPP_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "MatrixNetworkWrapper.hpp"
//#include "RPropNetwork.hpp"

// Necessary for c++ functions to be callable from Python's C
extern "C" {

  typedef struct {
	PyMatrixNetwork super; // inherit from MatrixNetwork
  } PyRPropNetwork;


  // Setting some constants here
  void setRPropConstants(PyObject *dict);

  /*
   * Python constructor
   * ------------------
   */

  /*
   * Python init
   * -----------
   */
  int RPropNetwork_init(PyRPropNetwork *self, PyObject *args, PyObject *kwds);

  /*
   * Python destructor
   * -----------------
   */

  /*
   * Wrapper methods
   * ===============
   */

  PyObject *RPropNetwork_learn(PyRPropNetwork *self, PyObject *args,
                               PyObject *kwargs);

  /*
   * Getters and Setters
   */
  PyObject *RPropNetwork_getMaxEpochs(PyRPropNetwork *self, void *closure);
  int RPropNetwork_setMaxEpochs(PyRPropNetwork *self, PyObject *value,
                                void *closure);

  PyObject *RPropNetwork_getMaxError(PyRPropNetwork *self, void *closure);
  int RPropNetwork_setMaxError(PyRPropNetwork *self, PyObject *value,
                               void *closure);

  PyObject *RPropNetwork_getErrorFunction(PyRPropNetwork *self, void *closure);
  int RPropNetwork_setErrorFunction(PyRPropNetwork *self, PyObject *value,
                                    void *closure);


  /*
   * Pickle methods
   * ==============
   */

  //PyObject *FFNetwork_getnewargs(FFNetwork *self);
  //PyObject *FFNetwork_reduce(FFNetwork *self);

}
#endif /* RPROPWRAPPER_HPP_ */
