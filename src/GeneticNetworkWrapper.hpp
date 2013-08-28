#ifndef _GENETICNETWORKWRAPPER_HPP_
#define _GENETICNETWORKWRAPPER_HP_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "MatrixNetworkWrapper.hpp"

// Necessary for c++ functions to be callable from Python's C
extern "C" {

  typedef struct {
    PyMatrixNetwork super; // inherit from MatrixNetwork
  } PyGenNetwork;

/*
 * Python constructor
 * ------------------
 */

/*
 * Python init
 * -----------
 */
  int GenNetwork_init(PyGenNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Python destructor
 * -----------------
 */

/*
 * Wrapper methods
 * ===============
 */

  PyObject *GenNetwork_learn(PyGenNetwork *self, PyObject *args,
                             PyObject *kwargs);

/*
 * Getters and Setters
 */
  PyObject *GenNetwork_getGenerations(PyGenNetwork *self, void *closure);
  int GenNetwork_setGenerations(PyGenNetwork *self, PyObject *value,
                                void *closure);

  PyObject *GenNetwork_getPopulationSize(PyGenNetwork *self, void *closure);
  int GenNetwork_setPopulationSize(PyGenNetwork *self, PyObject *value,
                                   void *closure);

  PyObject *GenNetwork_getWeightMutationChance(PyGenNetwork *self,
                                               void *closure);
  int GenNetwork_setWeightMutationChance(PyGenNetwork *self, PyObject *value,
                                         void *closure);

  PyObject *GenNetwork_getWeightMutationHalfPoint(PyGenNetwork *self,
                                                  void *closure);
  int GenNetwork_setWeightMutationHalfPoint(PyGenNetwork *self, PyObject *value,
                                            void *closure);

  PyObject *GenNetwork_getWeightMutationFactor(PyGenNetwork *self,
                                               void *closure);
  int GenNetwork_setWeightMutationFactor(PyGenNetwork *self, PyObject *value,
                                         void *closure);

  PyObject *GenNetwork_getDecayL1(PyGenNetwork *self,
                                  void *closure);
  int GenNetwork_setDecayL1(PyGenNetwork *self,
                            PyObject *value, void *closure);
  PyObject *GenNetwork_getDecayL2(PyGenNetwork *self,
                                  void *closure);
  int GenNetwork_setDecayL2(PyGenNetwork *self,
                            PyObject *value, void *closure);


  PyObject *GenNetwork_getWeightElimination(PyGenNetwork *self,
                                            void *closure);
  int GenNetwork_setWeightElimination(PyGenNetwork *self,
                                      PyObject *value, void *closure);
  PyObject *GenNetwork_getWeightEliminationLambda(PyGenNetwork *self,
                                                  void *closure);
  int GenNetwork_setWeightEliminationLambda(PyGenNetwork *self,
                                            PyObject *value,
                                            void *closure);

  /*  PyObject *GenNetwork_getResume(PyGenNetwork *self,
                                 void *closure);
  int GenNetwork_setResume(PyGenNetwork *self,
                           PyObject *value,
                           void *closure);
  */

  PyObject *GenNetwork_getCrossoverChance(PyGenNetwork *self,
                                          void *closure);
  int GenNetwork_setCrossoverChance(PyGenNetwork *self,
                                    PyObject *value,
                                    void *closure);


  PyObject *GenNetwork_getSelectionMethod(PyGenNetwork *self,
                                          void *closure);
  int GenNetwork_setSelectionMethod(PyGenNetwork *self,
                                    PyObject *value,
                                    void *closure);

  PyObject *GenNetwork_getCrossoverMethod(PyGenNetwork *self,
                                          void *closure);
  int GenNetwork_setCrossoverMethod(PyGenNetwork *self,
                                    PyObject *value,
                                    void *closure);
  PyObject *GenNetwork_getInsertMethod(PyGenNetwork *self,
                                       void *closure);
  int GenNetwork_setInsertMethod(PyGenNetwork *self,
                                 PyObject *value,
                                 void *closure);

  PyObject *GenNetwork_getFitnessFunction(PyGenNetwork *self,
                                          void *closure);
  int GenNetwork_setFitnessFunction(PyGenNetwork *self,
                                    PyObject *value,
                                    void *closure);



/*
 * Pickle methods
 * ==============
 */

//PyObject *FFNetwork_getnewargs(FFNetwork *self);
//PyObject *FFNetwork_reduce(FFNetwork *self);

}

#endif /* _GENETICNETWORKWRAPPER_H_ */
