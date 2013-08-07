
#ifndef GENETICSURVIVALMSENETWORKWRAPPER_H_
#define GENETICSURVIVALMSENETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "GeneticNetworkWrapper.h"

// Necessary for c++ functions to be callable from Python's C
extern "C" {

  typedef struct {
	PyGenNetwork super;
  } PyGenSurvMSENetwork;

/*
 * Python constructor
 * ------------------
 */

/*
 * Python init
 * -----------
 */
  int GenSurvMSENetwork_init(PyGenSurvMSENetwork *self, PyObject *args,
                          PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */

/*
 * Wrapper methods
 * ===============
 */

  PyObject *GenSurvMSENetwork_learn(PyGenSurvMSENetwork *self, PyObject *args, \
                                 PyObject *kwargs);

}
#endif /* GENETICSURVIVALMSENETWORKWRAPPER_H_*/
