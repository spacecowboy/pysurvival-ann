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
 * Python constructor
 * ------------------
 */
PyObject *FFNetwork_new(PyTypeObject *type, PyObject *args,
		PyObject *kwds);

/*
 * Python init
 * -----------
 */
int FFNetwork_init(FFNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */
void FFNetwork_dealloc(FFNetwork *self);

/*
 * Wrapper methods
 * ===============
 */

PyObject *FFNetwork_output(FFNetwork *self, PyObject *inputs);


/*
 * Pickle methods
 * ==============
 */

//PyObject *FFNetwork_getnewargs(FFNetwork *self);
//PyObject *FFNetwork_reduce(FFNetwork *self);

}

#endif /* FFNETWORKWRAPPER_H_ */
