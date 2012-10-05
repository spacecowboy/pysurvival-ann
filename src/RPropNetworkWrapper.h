/*
 * FFNetworkWrapper.h
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#ifndef RPROPNETWORKWRAPPER_H_
#define RPROPNETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "FFNetworkWrapper.h"
//#include "RPropNetwork.h"

// Necessary for c++ functions to be callable from Python's C
extern "C" {

typedef struct {
	PyFFNetwork super; // inherit from FFNetwork
} PyRPropNetwork;

/*
 * Python constructor
 * ------------------
 */
//PyObject *RPropNetwork_new(PyTypeObject *type, PyObject *args,
//		PyObject *kwds);

/*
 * Python init
 * -----------
 */
int RPropNetwork_init(PyRPropNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */
//void RPropNetwork_dealloc(PyRPropNetwork *self);

/*
 * Wrapper methods
 * ===============
 */

PyObject *RPropNetwork_learn(PyRPropNetwork *self, PyObject *args, PyObject *kwargs);


/*
 * Pickle methods
 * ==============
 */

//PyObject *FFNetwork_getnewargs(FFNetwork *self);
//PyObject *FFNetwork_reduce(FFNetwork *self);

}
#endif /* RPROPWRAPPER_H_ */
