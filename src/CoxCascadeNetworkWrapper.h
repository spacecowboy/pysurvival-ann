#ifndef _COXCASCADENETWORKWRAPPER_H_
#define _COXCASCADENETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "CascadeNetworkWrapper.h"

extern "C" {

  typedef struct {
	PyCascadeNetwork super; // inherit from RPropNetwork
  } PyCoxCascadeNetwork;

/*
 * Python init
 * -----------
 */
  int CoxCascadeNetwork_init(PyCoxCascadeNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Wrapper methods
 * ===============
 */

  PyObject *CoxCascadeNetwork_learn(PyCoxCascadeNetwork *self, PyObject *args, PyObject *kwargs);


}

#endif /* _COXCASCADENETWORKWRAPPER_H_ */
