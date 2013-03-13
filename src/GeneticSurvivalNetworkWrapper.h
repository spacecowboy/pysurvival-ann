#ifndef GENETICSURVIVALNETWORKWRAPPER_H_
#define GENETICSURVIVALNETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "GeneticNetworkWrapper.h"

// Necessary for c++ functions to be callable from Python's C
extern "C" {

  typedef struct {
	PyGenNetwork super;
  } PyGenSurvNetwork;

/*
 * Python constructor
 * ------------------
 */

/*
 * Python init
 * -----------
 */
  int GenSurvNetwork_init(PyGenSurvNetwork *self, PyObject *args,
                          PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */

/*
 * Wrapper methods
 * ===============
 */

  PyObject *GenSurvNetwork_learn(PyGenSurvNetwork *self, PyObject *args, \
                                 PyObject *kwargs);

}
#endif /* GENETICSURVIVALNETWORKWRAPPER_H_*/
