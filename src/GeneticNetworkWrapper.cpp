#include "GeneticNetworkWrapper.hpp"
// For convenience macros in python3
#include "PythonModule.h"
// Must include this before arrayobject
#include "ExtensionHeader.h"
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "GeneticNetwork.hpp"
#include <stdio.h>

extern "C" {

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
                                                      PyArray_DOUBLE, 2, 2);
    if (inputArray == NULL)
      return NULL;

    targetArray = (PyArrayObject *) PyArray_ContiguousFromObject(targets,
                                                                 PyArray_DOUBLE,
                                                                 2, 2);
    if (targetArray == NULL) {
      Py_DECREF(inputArray);
      return NULL;
    }

    // Objects were converted successfully. But make sure they are the same length!

    if (inputArray->dimensions[0] != targetArray->dimensions[0] ||
        (unsigned int)inputArray->dimensions[1] != self->super.net->INPUT_COUNT ||
        (unsigned int)targetArray->dimensions[1] < self->super.net->OUTPUT_COUNT)
      {
        // Decrement, set error and return
        PyErr_Format(PyExc_ValueError, "Inputs and targets must have the same\
 number of rows. Also the target columns cannot be less than the number of\
 output neurons.");
        Py_DECREF(inputArray);
        Py_DECREF(targetArray);

        return NULL;
      }

    // Arguments are valid!
/*
    // Make local versions so we don't fuck up Python thread shit
    double *inputsCopy = new double[inputArray->dimensions[0] *
                                    inputArray->dimensions[1]];
    double *targetsCopy = new double[targetArray->dimensions[0] *
                                    targetArray->dimensions[1]];
    unsigned int length = inputArray->dimensions[0];
    // Same length
    int index;
    for (int i = 0; i < inputArray->dimensions[0]; i++) {
        for (int j = 0; j < inputArray->dimensions[1]; j++) {
            index = j + i * inputArray->dimensions[1];
            inputsCopy[index] = inputArray->data[index];
        }
        for (int j = 0; j < targetArray->dimensions[1]; j++) {
            index = j + i * targetArray->dimensions[1];
            targetsCopy[index] = targetArray->data[index];
        }
    }
*/
    // Release the GIL
    Py_BEGIN_ALLOW_THREADS;
    ((GeneticNetwork*)self->super.net)->learn((double *) inputArray->data,
                                              (double *) targetArray->data,
                                              inputArray->dimensions[0]);
    // Acquire the GIL again
    Py_END_ALLOW_THREADS;
/*
    delete[] inputsCopy;
    delete[] targetsCopy;
*/
    // Decrement counters for inputArray and targetArray
    Py_DECREF(inputArray);
    Py_DECREF(targetArray);

    // Return none
    return Py_BuildValue("");
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
}
