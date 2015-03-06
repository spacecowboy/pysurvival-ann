#include "GeneticNetworkWrapper.hpp"
// For convenience macros in python3
#include "PythonModule.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "GeneticNetwork.hpp"
#include <stdio.h>
#include "GeneticSelection.hpp"
#include "GeneticCrossover.hpp"
#include "GeneticFitness.hpp"
#include "Statistics.hpp"

extern "C" {

  // Static variables
  // Set some constants in the object's dictionary
  // Call with GenNetworkType.tp_dict
  void setGeneticNetworkConstants(PyObject *dict) {
    // Selection
    PyDict_SetItemString(dict, "SELECTION_GEOMETRIC",
                         Py_BuildValue("i",
                                       SelectionMethod::SELECTION_GEOMETRIC));
    PyDict_SetItemString(dict, "SELECTION_ROULETTE",
                         Py_BuildValue("i",
                                       SelectionMethod::SELECTION_ROULETTE));
    PyDict_SetItemString(dict, "SELECTION_TOURNAMENT",
                         Py_BuildValue("i",
                                       SelectionMethod::SELECTION_TOURNAMENT));

    // Crossover
    PyDict_SetItemString(dict, "CROSSOVER_ONEPOINT",
                         Py_BuildValue("i",
                                       CrossoverMethod::CROSSOVER_ONEPOINT));
    PyDict_SetItemString(dict, "CROSSOVER_TWOPOINT",
                         Py_BuildValue("i",
                                       CrossoverMethod::CROSSOVER_TWOPOINT));
    PyDict_SetItemString(dict, "CROSSOVER_UNIFORM",
                         Py_BuildValue("i",
                                       CrossoverMethod::CROSSOVER_UNIFORM));

    // Fitness
    PyDict_SetItemString(dict, "FITNESS_MSE",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_MSE));
    PyDict_SetItemString(dict, "FITNESS_CINDEX",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_CINDEX));
    PyDict_SetItemString(dict, "FITNESS_SURV_MSE",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_MSE_CENS));
    PyDict_SetItemString(dict, "FITNESS_SURV_LIKELIHOOD",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_SURV_LIKELIHOOD));
    PyDict_SetItemString(dict, "FITNESS_TARONEWARE_MEAN",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_TARONEWARE_MEAN));
    PyDict_SetItemString(dict, "FITNESS_TARONEWARE_HIGHLOW",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_TARONEWARE_HIGHLOW));
    PyDict_SetItemString(dict, "FITNESS_SURV_KAPLAN_MAX",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_SURV_KAPLAN_MAX));
    PyDict_SetItemString(dict, "FITNESS_SURV_KAPLAN_MIN",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_SURV_KAPLAN_MIN));
    PyDict_SetItemString(dict, "FITNESS_SURV_RISKGROUP_HIGH",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_SURV_RISKGROUP_HIGH));
    PyDict_SetItemString(dict, "FITNESS_SURV_RISKGROUP_LOW",
                         Py_BuildValue("i",
                                       FitnessFunction::FITNESS_SURV_RISKGROUP_LOW));

    // Statistics
        // Fitness
    PyDict_SetItemString(dict, "TARONEWARE_LOGRANK",
                         Py_BuildValue("i",
                                       TaroneWareType::LOGRANK));
    PyDict_SetItemString(dict, "TARONEWARE_GEHAN",
                         Py_BuildValue("i",
                                       TaroneWareType::GEHAN));
    PyDict_SetItemString(dict, "TARONEWARE_TARONEWARE",
                         Py_BuildValue("i",
                                       TaroneWareType::TARONEWARE));

  }




   // Python init
  int GenNetwork_init(PyGenNetwork *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = { (char*)"numOfInputs", \
                              (char*)"numOfHidden",
                              (char*)"numOfOutputs",
                              NULL };

    unsigned int numOfInputs, numOfHidden, numOfOutputs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist, &numOfInputs,
                                     &numOfHidden, &numOfOutputs)) {
      PyErr_Format(PyExc_ValueError,
                   "Arguments should be (all mandatory positive integers): \
numOfInputs, numOfHidden, numOfOutputs");
      return -1;
    }

    self->super.net = new GeneticNetwork(numOfInputs, numOfHidden,
                                         numOfOutputs);

    if (self->super.net == NULL)
      return -1;

    return 0;
  }

  /*
   * Wrapper methods
   * ===============
   */

  PyObject *GenNetwork_learn(PyGenNetwork *self, PyObject *args,
                                 PyObject *kwargs) {
    PyObject *inputs = NULL;
    PyObject *targets = NULL;
    // Check inputs
    static char *kwlist[] = {(char*)"inputs", \
                             (char*)"targets", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
                                     &inputs, &targets)) {
      PyErr_Format(PyExc_ValueError, "Arguments should be: inputs (2d array), \
targets (2d array)");
      return NULL;
    }

    // Make sure they conform to required structure
    PyArrayObject *inputArray = NULL;
    PyArrayObject *targetArray = NULL;

    inputArray =
      (PyArrayObject *)  PyArray_ContiguousFromObject(inputs,
                                                      NPY_DOUBLE, 2, 2);
    if (inputArray == NULL)
      return NULL;

    targetArray = (PyArrayObject *) PyArray_ContiguousFromObject(targets,
                                                                 NPY_DOUBLE,
                                                                 2, 2);
    if (targetArray == NULL) {
      Py_DECREF(inputArray);
      return NULL;
    }

    // Objects were converted successfully. But make sure they are the
    // correct length!
    int expectedTargetCount =
      getExpectedTargetCount(((GeneticNetwork*)self->super.net)->getFitnessFunction());
    if (expectedTargetCount < 1) {
      // Then expect output count
      expectedTargetCount = self->super.net->OUTPUT_COUNT;
    }

    int error = 0;
    if (PyArray_DIM(inputArray, 0)!= PyArray_DIM(targetArray, 0)) {
      error = 1;
      PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same\
 number of rows.");
    }
    if ((unsigned int)PyArray_DIM(inputArray, 1) != self->super.net->INPUT_COUNT) {
      error = 1;
      PyErr_Format(PyExc_ValueError, "Number of input columns must match number\
of input neurons in network.");
    }

    if ((int)PyArray_DIM(targetArray, 1) != expectedTargetCount) {
      error = 1;
      PyErr_Format(PyExc_ValueError, "Number of target columns does not match\
expected number based on fitness function.");
    }

    if (error == 1) {
      Py_DECREF(inputArray);
      Py_DECREF(targetArray);
      return NULL;
    }

    // Arguments are valid!
    std::vector<double> vInputs(PyArray_DIM(inputArray, 0) * PyArray_DIM(inputArray, 1),
                                0.0);
    std::vector<double> vTargets(PyArray_DIM(targetArray, 0) * PyArray_DIM(targetArray, 1),
                                0.0);

    int index;
    double *val = NULL;
    for (int i = 0; i < PyArray_DIM(inputArray, 0); i++) {
        for (int j = 0; j < PyArray_DIM(inputArray, 1); j++) {
            index = j + i * PyArray_DIM(inputArray, 1);
            val = (double *) PyArray_GETPTR2(inputArray, i, j);
            if (val == NULL) {
              Py_DECREF(inputArray);
              Py_DECREF(targetArray);
              PyErr_Format(PyExc_ValueError,
                           "Something went wrong when iterating of input \
 values. Possibly wrong length?");
              return NULL;
            }
            vInputs.at(index) = *val;
        }
        for (int j = 0; j < PyArray_DIM(targetArray, 1); j++) {
            index = j + i * PyArray_DIM(targetArray, 1);
            val = (double *) PyArray_GETPTR2(targetArray, i, j);
            if (val == NULL) {
              Py_DECREF(inputArray);
              Py_DECREF(targetArray);
              PyErr_Format(PyExc_ValueError,
                           "Something went wrong when iterating of input \
 values. Possibly wrong length?");
              return NULL;
            }
            vTargets.at(index) = *val;
        }
    }

    // Release the GIL
    //Py_BEGIN_ALLOW_THREADS;
    try {
      ((GeneticNetwork*)self->super.net)->learn(vInputs,
                                                vTargets,
                                                PyArray_DIM(inputArray, 0));
    } catch (const std::exception& ex) {
      printf("\nException thrown: %s\n", ex.what());
      PyErr_Format(PyExc_RuntimeError, "%s", ex.what());
      error = 1;
    }
    // Acquire the GIL again
    //Py_END_ALLOW_THREADS;

    // Decrement counters for inputArray and targetArray
    Py_DECREF(inputArray);
    Py_DECREF(targetArray);

    if (error > 0) {
      return NULL;
    } else {
      // Return none
      return Py_BuildValue("");
    }
  }

  PyObject *GenNetwork_getPredictionFitness(PyGenNetwork *self, PyObject *args,
                                            PyObject *kwargs) {
    PyObject *inputs = NULL;
    PyObject *targets = NULL;
    // Check inputs
    static char *kwlist[] = {(char*)"inputs", \
                             (char*)"targets", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
                                     &inputs, &targets)) {
      PyErr_Format(PyExc_ValueError, "Arguments should be: inputs (2d array), \
targets (2d array)");
      return NULL;
    }

    // Make sure they conform to required structure
    PyArrayObject *inputArray = NULL;
    PyArrayObject *targetArray = NULL;

    inputArray =
      (PyArrayObject *)  PyArray_ContiguousFromObject(inputs,
                                                      NPY_DOUBLE, 2, 2);
    if (inputArray == NULL)
      return NULL;

    targetArray = (PyArrayObject *) PyArray_ContiguousFromObject(targets,
                                                                 NPY_DOUBLE,
                                                                 2, 2);
    if (targetArray == NULL) {
      Py_DECREF(inputArray);
      return NULL;
    }

    // Objects were converted successfully. But make sure they are the
    // correct length!
    int expectedTargetCount =
      getExpectedTargetCount(((GeneticNetwork*)self->super.net)->getFitnessFunction());
    if (expectedTargetCount < 1) {
      // Then expect output count
      expectedTargetCount = self->super.net->OUTPUT_COUNT;
    }

    int error = 0;
    if (PyArray_DIM(inputArray, 0)!= PyArray_DIM(targetArray, 0)) {
      error = 1;
      PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same\
 number of rows.");
    }
    if ((unsigned int)PyArray_DIM(inputArray, 1) != self->super.net->INPUT_COUNT) {
      error = 1;
      PyErr_Format(PyExc_ValueError, "Number of input columns must match number\
of input neurons in network.");
    }

    if ((int)PyArray_DIM(targetArray, 1) != expectedTargetCount) {
      error = 1;
      PyErr_Format(PyExc_ValueError, "Number of target columns does not match\
expected number based on fitness function.");
    }

    if (error == 1) {
      Py_DECREF(inputArray);
      Py_DECREF(targetArray);
      return NULL;
    }

    // Arguments are valid!
    std::vector<double> vInputs(PyArray_DIM(inputArray, 0) * PyArray_DIM(inputArray, 1),
                                0.0);
    std::vector<double> vTargets(PyArray_DIM(targetArray, 0) * PyArray_DIM(targetArray, 1),
                                0.0);

    int index;
    double *val = NULL;
    for (int i = 0; i < PyArray_DIM(inputArray, 0); i++) {
        for (int j = 0; j < PyArray_DIM(inputArray, 1); j++) {
            index = j + i * PyArray_DIM(inputArray, 1);
            val = (double *) PyArray_GETPTR2(inputArray, i, j);
            if (val == NULL) {
              Py_DECREF(inputArray);
              Py_DECREF(targetArray);
              PyErr_Format(PyExc_ValueError,
                           "Something went wrong when iterating of input \
 values. Possibly wrong length?");
              return NULL;
            }
            vInputs.at(index) = *val;
        }
        for (int j = 0; j < PyArray_DIM(targetArray, 1); j++) {
            index = j + i * PyArray_DIM(targetArray, 1);
            val = (double *) PyArray_GETPTR2(targetArray, i, j);
            if (val == NULL) {
              Py_DECREF(inputArray);
              Py_DECREF(targetArray);
              PyErr_Format(PyExc_ValueError,
                           "Something went wrong when iterating of input \
 values. Possibly wrong length?");
              return NULL;
            }
            vTargets.at(index) = *val;
        }
    }

    // Release the GIL
    double fitness = 0;
    try {
      fitness = ((GeneticNetwork*)self->super.net)->getPredictionFitness(vInputs,
                                   vTargets,
                                   PyArray_DIM(inputArray, 0));
    } catch (const std::exception& ex) {
      printf("\nException thrown: %s\n", ex.what());
      PyErr_Format(PyExc_RuntimeError, "%s", ex.what());
      error = 1;
    }

    // Decrement counters for inputArray and targetArray
    Py_DECREF(inputArray);
    Py_DECREF(targetArray);

    if (error > 0) {
      return NULL;
    } else {
      // Return fitness
      return Py_BuildValue("d", fitness);
    }
  }

  /*
   * Getters and Setters
   */
  PyObject *GenNetwork_getGenerations(PyGenNetwork *self, void *closure) {
    return Py_BuildValue("I", ((GeneticNetwork*)self->super.net)->
                         getGenerations());
  }

  int GenNetwork_setGenerations(PyGenNetwork *self,
                                PyObject *value,
                                void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setGenerations((unsigned int) val);
    return 0;
  }

  PyObject *GenNetwork_getPopulationSize(PyGenNetwork *self, void *closure){
    return Py_BuildValue("I", ((GeneticNetwork*)self->super.net)->
                         getPopulationSize());
  }

  int GenNetwork_setPopulationSize(PyGenNetwork *self, PyObject *value,
                                   void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setPopulationSize((unsigned int) val);
    return 0;
  }

  PyObject *GenNetwork_getWeightMutationChance(PyGenNetwork *self,
                                               void *closure) {
    return Py_BuildValue("d", ((GeneticNetwork*)self->super.net)->
                         getWeightMutationChance());
  }

  int GenNetwork_setWeightMutationChance(PyGenNetwork *self, PyObject *value,
                                         void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setWeightMutationChance(val);
    return 0;
  }

  PyObject *GenNetwork_getWeightMutationHalfPoint(PyGenNetwork *self,
                                                  void *closure) {
    return Py_BuildValue("I", ((GeneticNetwork*)self->super.net)->
                         getWeightMutationHalfPoint());
  }

  int GenNetwork_setWeightMutationHalfPoint(PyGenNetwork *self, PyObject *value,
                                            void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    long val = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->
      setWeightMutationHalfPoint((unsigned int) val);
    return 0;
  }

  PyObject *GenNetwork_getWeightMutationFactor(PyGenNetwork *self,
                                               void *closure) {
    return Py_BuildValue("d", ((GeneticNetwork*)self->super.net)->
                         getWeightMutationFactor());
  }

  int GenNetwork_setWeightMutationFactor(PyGenNetwork *self, PyObject *value,
                                         void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setWeightMutationFactor(val);
    return 0;
  }


  PyObject *GenNetwork_getDecayL1(PyGenNetwork *self, void *closure) {
    return Py_BuildValue("d",
                         ((GeneticNetwork*)self->super.net)->getDecayL1());
  }

  int GenNetwork_setDecayL1(PyGenNetwork *self,
                            PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setDecayL1(val);
    return 0;
  }

  PyObject *GenNetwork_getDecayL2(PyGenNetwork *self, void *closure) {
    return Py_BuildValue("d",
                         ((GeneticNetwork*)self->super.net)->getDecayL2());
  }

  int GenNetwork_setDecayL2(PyGenNetwork *self,
                            PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setDecayL2(val);
    return 0;
  }

  PyObject *GenNetwork_getWeightElimination(PyGenNetwork *self,
                                            void *closure) {
    return Py_BuildValue("d",
                         ((GeneticNetwork*)self->super.net)->
                         getWeightElimination());
  }

  int GenNetwork_setWeightElimination(PyGenNetwork *self,
                                      PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setWeightElimination(val);
    return 0;
  }

  PyObject *GenNetwork_getWeightEliminationLambda(PyGenNetwork *self,
                                                  void *closure) {
    return Py_BuildValue("d",
                         ((GeneticNetwork*)self->super.net)->
                         getWeightEliminationLambda());
  }

  int GenNetwork_setWeightEliminationLambda(PyGenNetwork *self,
                                            PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setWeightEliminationLambda(val);
    return 0;
  }

  /*
  PyObject *GenNetwork_getResume(PyGenNetwork *self,
                                 void *closure) {
    if (((GeneticNetwork*)self->super.net)->
        getResume())
      Py_RETURN_TRUE;
    else
      Py_RETURN_FALSE;
  }

  int GenNetwork_setResume(PyGenNetwork *self,
                           PyObject *value, void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    int res = PyObject_IsTrue(value);

    if (res < 0)
      return -1;

    ((GeneticNetwork*)self->super.net)->setResume(res > 0);

    return 0;
  }
  */

  PyObject *GenNetwork_getCrossoverChance(PyGenNetwork *self,
                                          void *closure) {
    return Py_BuildValue("d", ((GeneticNetwork*)self->super.net)->
                         getCrossoverChance());
  }

  int GenNetwork_setCrossoverChance(PyGenNetwork *self, PyObject *value,
                                    void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setCrossoverChance(val);
    return 0;
  }


  PyObject *GenNetwork_getConnsMutationChance(PyGenNetwork *self,
                                              void *closure) {
    return Py_BuildValue("d", ((GeneticNetwork*)self->super.net)->
                         getConnsMutationChance());
  }
  int GenNetwork_setConnsMutationChance(PyGenNetwork *self,
                                        PyObject *value,
                                        void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setConnsMutationChance(val);
    return 0;
  }

  PyObject *GenNetwork_getActFuncMutationChance(PyGenNetwork *self,
                                                void *closure) {
    return Py_BuildValue("d", ((GeneticNetwork*)self->super.net)->
                         getActFuncMutationChance());
  }
  int GenNetwork_setActFuncMutationChance(PyGenNetwork *self,
                                          PyObject *value,
                                          void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->setActFuncMutationChance(val);
    return 0;
  }



  PyObject *GenNetwork_getSelectionMethod(PyGenNetwork *self,
                                          void *closure) {
    return Py_BuildValue("i", ((GeneticNetwork*)self->super.net)->
                         getSelectionMethod());
  }

  int GenNetwork_setSelectionMethod(PyGenNetwork *self, PyObject *value,
                                    void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    if (! PyInt_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Must be an integer value!");
      return 1;
    }

    long i = PyInt_AsLong(value);

    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->
      setSelectionMethod((SelectionMethod) i);
    return 0;
  }

  PyObject *GenNetwork_getCrossoverMethod(PyGenNetwork *self,
                                          void *closure) {
    return Py_BuildValue("i", ((GeneticNetwork*)self->super.net)->
                         getCrossoverMethod());
  }

  int GenNetwork_setCrossoverMethod(PyGenNetwork *self, PyObject *value,
                                    void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    if (! PyInt_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Must be an integer value!");
      return 1;
    }

    long i = PyInt_AsLong(value);

    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->
      setCrossoverMethod((CrossoverMethod) i);
    return 0;
  }

  PyObject *GenNetwork_getFitnessFunction(PyGenNetwork *self,
                                          void *closure) {
    return Py_BuildValue("i", ((GeneticNetwork*)self->super.net)->
                         getFitnessFunction());
  }

  int GenNetwork_setFitnessFunction(PyGenNetwork *self, PyObject *value,
                                    void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    if (! PyInt_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Must be an integer value!");
      return 1;
    }

    long i = PyInt_AsLong(value);

    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->
      setFitnessFunction((FitnessFunction) i);
    return 0;
  }

  PyObject *GenNetwork_getTaroneWareStatistic(PyGenNetwork *self,
                                              void *closure) {
    return Py_BuildValue("i", ((GeneticNetwork*)self->super.net)->
                         getTaroneWareStatistic());
  }

  int GenNetwork_setTaroneWareStatistic(PyGenNetwork *self, PyObject *value,
                                        void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    if (! PyInt_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Must be an integer value!");
      return 1;
    }

    long i = PyInt_AsLong(value);

    if (PyErr_Occurred()) {
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->
      setTaroneWareStatistic((TaroneWareType) i);
    return 0;
  }

  PyObject *GenNetwork_getMinGroup(PyGenNetwork *self,
                                   void *closure) {
    return Py_BuildValue("i", ((GeneticNetwork*)self->super.net)->
                         getMinGroup());
  }

  int GenNetwork_setMinGroup(PyGenNetwork *self, PyObject *value,
                             void *closure) {
    if (value == NULL) {
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute");
      return -1;
    }

    if (! PyInt_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Must be an integer value!");
      return 1;
    }

    long i = PyInt_AsLong(value);

    if (PyErr_Occurred()) {
      return -1;
    }

    if (i < 0) {
      PyErr_SetString(PyExc_TypeError, "Can't be less than zero!'");
      return -1;
    }

    ((GeneticNetwork*)self->super.net)->
      setMinGroup(i);
    return 0;
  }

}
