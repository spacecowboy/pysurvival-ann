#ifndef GENETICSURVIVALNETWORKWRAPPER_H_
#define GENETICSURVIVALNETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "FFNetworkWrapper.h"

// Necessary for c++ functions to be callable from Python's C
extern "C" {

  typedef struct {
	PyFFNetwork super; // inherit from FFNetwork
  } PyGenSurvNetwork;

/*
 * Python constructor
 * ------------------
 */

/*
 * Python init
 * -----------
 */
  int GenSurvNetwork_init(PyGenSurvNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */

/*
 * Wrapper methods
 * ===============
 */

  PyObject *GenSurvNetwork_learn(PyGenSurvNetwork *self, PyObject *args, PyObject *kwargs);

/*
 * Getters and Setters
 */
  PyObject *GenSurvNetwork_getGenerations(PyGenSurvNetwork *self, void *closure);
  int GenSurvNetwork_setGenerations(PyGenSurvNetwork *self, PyObject *value, void *closure);

  PyObject *GenSurvNetwork_getPopulationSize(PyGenSurvNetwork *self, void *closure);
  int GenSurvNetwork_setPopulationSize(PyGenSurvNetwork *self, PyObject *value, void *closure);

  PyObject *GenSurvNetwork_getWeightMutationChance(PyGenSurvNetwork *self, void *closure);
  int GenSurvNetwork_setWeightMutationChance(PyGenSurvNetwork *self, PyObject *value, void *closure);

  PyObject *GenSurvNetwork_getWeightMutationHalfPoint(PyGenSurvNetwork *self, void *closure);
  int GenSurvNetwork_setWeightMutationHalfPoint(PyGenSurvNetwork *self, PyObject *value, void *closure);

  PyObject *GenSurvNetwork_getWeightMutationStdDev(PyGenSurvNetwork *self, void *closure);
  int GenSurvNetwork_setWeightMutationStdDev(PyGenSurvNetwork *self, PyObject *value, void *closure);

	 PyObject *GenSurvNetwork_getDecayL1(PyGenSurvNetwork *self,
										 void *closure);
	 int GenSurvNetwork_setDecayL1(PyGenSurvNetwork *self,
								   PyObject *value, void *closure);
	 PyObject *GenSurvNetwork_getDecayL2(PyGenSurvNetwork *self,
										 void *closure);
	 int GenSurvNetwork_setDecayL2(PyGenSurvNetwork *self,
								   PyObject *value, void *closure);


	 PyObject *GenSurvNetwork_getWeightElimination(PyGenSurvNetwork *self,
												   void *closure);
	 int GenSurvNetwork_setWeightElimination(PyGenSurvNetwork *self,
											 PyObject *value, void *closure);
	 PyObject *GenSurvNetwork_getWeightEliminationLambda(PyGenSurvNetwork *self,
														 void *closure);
	 int GenSurvNetwork_setWeightEliminationLambda(PyGenSurvNetwork *self,
												   PyObject *value,
												   void *closure);


/*
 * Pickle methods
 * ==============
 */

//PyObject *FFNetwork_getnewargs(FFNetwork *self);
//PyObject *FFNetwork_reduce(FFNetwork *self);


}
#endif /* GENETICSURVIVALNETWORKWRAPPER_H_*/
