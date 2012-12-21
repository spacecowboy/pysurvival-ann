#ifndef _CASCADENETWORKWRAPPER_H_
#define _CASCADENETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
//#include "FFNetworkWrapper.h"
#include "RPropNetworkWrapper.h"

extern "C" {

  typedef struct {
	PyRPropNetwork super; // inherit from RPropNetwork
  } PyCascadeNetwork;

/*
 * Python init
 * -----------
 */
  int CascadeNetwork_init(PyCascadeNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Getters and Setters
 */
  PyObject *CascadeNetwork_getMaxHidden(PyCascadeNetwork *self, void *closure);
  int CascadeNetwork_setMaxHidden(PyCascadeNetwork *self, PyObject *value, void *closure);

  PyObject *CascadeNetwork_getMaxHiddenEpochs(PyCascadeNetwork *self, void *closure);
  int CascadeNetwork_setMaxHiddenEpochs(PyCascadeNetwork *self, PyObject *value, void *closure);


}

#endif /* _CASCADENETWORKWRAPPER_H_ */
