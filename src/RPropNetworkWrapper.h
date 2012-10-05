/*
 * FFNetworkWrapper.h
 *
 *  Created on: 1 okt 2012
 *      Author: jonas
 */

#ifndef PROPNETWORKWRAPPER_H_
#define RPOPNETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "RPropNetwork.h"

// Necessary for c++ functions to be callable from Python's C
extern "C" {

/*
 * Python constructor
 * ------------------
 */
PyObject *RPropNetwork_new(PyTypeObject *type, PyObject *args,
		PyObject *kwds);

/*
 * Python init
 * -----------
 */
int RPropNetwork_init(RPropNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */
void RPropNetwork_dealloc(RPropNetwork *self);

/*
 * Wrapper methods
 * ===============
 */

PyObject *RPropNetwork_output(RPropNetwork *self, PyObject *inputs);


/*
 * Pickle methods
 * ==============
 */

//PyObject *FFNetwork_getnewargs(FFNetwork *self);
//PyObject *FFNetwork_reduce(FFNetwork *self);

#endif /* RPROPWRAPPER_H_ */
