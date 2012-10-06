/*
 * FFNetworkWrapper.h
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#ifndef FFNETWORKWRAPPER_H_
#define FFNETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "FFNetwork.h"

// Necessary for c++ functions to be callable from Python's C
extern "C" {
/*
 * Python object
 */
typedef struct {
	PyObject_HEAD // inherit from PyObject
	FFNetwork *net; // actual network from c++
} PyFFNetwork;


/*
 * Python constructor
 * ------------------
 */
PyObject *FFNetwork_new(PyTypeObject *type, PyObject *args,
		PyObject *kwds);

/*
 * Python init
 * -----------
 */
int FFNetwork_init(PyFFNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */
void FFNetwork_dealloc(PyFFNetwork *self);

/*
 * Wrapper methods
 * ===============
 */

PyObject *FFNetwork_output(PyFFNetwork *self, PyObject *inputs);

PyObject *FFNetwork_connectHToH(PyFFNetwork *self, PyObject *args, PyObject *kwargs);

  PyObject *FFNetwork_connectHToI(PyFFNetwork *self, PyObject *args, PyObject *kwargs);

  PyObject *FFNetwork_connectHToB(PyFFNetwork *self, PyObject *args, PyObject *kwargs);

  PyObject *FFNetwork_connectOToH(PyFFNetwork *self, PyObject *args, PyObject *kwargs);

  PyObject *FFNetwork_connectOToI(PyFFNetwork *self, PyObject *args, PyObject *kwargs);

  PyObject *FFNetwork_connectOToB(PyFFNetwork *self, PyObject *args, PyObject *kwargs);

  PyObject *FFNetwork_setOutputActivationFunction(PyFFNetwork *self, PyObject *args, PyObject *kwargs);

  PyObject *FFNetwork_setHiddenActivationFunction(PyFFNetwork *self, PyObject *args, PyObject *kwargs);


/*
 * Getters and Setters
 */
PyObject *FFNetwork_getNumOfInputs(PyFFNetwork *self, void *closure);
PyObject *FFNetwork_getNumOfHidden(PyFFNetwork *self, void *closure);
PyObject *FFNetwork_getNumOfOutputs(PyFFNetwork *self, void *closure);

/*
 * Pickle methods
 * ==============
 */

//PyObject *FFNetwork_getnewargs(FFNetwork *self);
//PyObject *FFNetwork_reduce(FFNetwork *self);

}

#endif /* FFNETWORKWRAPPER_H_ */
