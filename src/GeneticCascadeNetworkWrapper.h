#ifndef _GENETICCASCADENETWORKWRAPPER_H_
#define _GENETICCASCADENETWORKWRAPPER_H_

#include "Python.h"
#include "structmember.h" // used to declare member list
#include "CascadeNetworkWrapper.h"

extern "C" {

  typedef struct {
	PyCascadeNetwork super; // inherit from RPropNetwork
  } PyGeneticCascadeNetwork;

/*
 * Python init
 * -----------
 */
  int GeneticCascadeNetwork_init(PyGeneticCascadeNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Wrapper methods
 * ===============
 */

  PyObject *GeneticCascadeNetwork_learn(PyGeneticCascadeNetwork *self, PyObject *args, PyObject *kwargs);


  /*
    -------------
    LadderNetwork
    -------------
  */
    typedef struct {
	PyGeneticCascadeNetwork super; // inherit from RPropNetwork
  } PyGeneticLadderNetwork;

/*
 * Python init
 * -----------
 */
  int GeneticLadderNetwork_init(PyGeneticLadderNetwork *self, PyObject *args, PyObject *kwds);

/*
 * Wrapper methods
 * ===============
 */

  PyObject *GeneticLadderNetwork_learn(PyGeneticLadderNetwork *self, PyObject *args, PyObject *kwargs);


}

#endif /* _GENETICCASCADENETWORKWRAPPER_H_ */
