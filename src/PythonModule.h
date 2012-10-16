/*
  Define Macros to keep python 2 compatibility while becoming
  compatible with python 3.
 */

#ifndef _PYTHONMODULE_H_
#define _PYTHONMODULE_H_

// This must always be first!
#include "Python.h"

// Define a couple of convenience macros, doing the version
// check once instead of everywhere.
#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
    static struct PyModuleDef moduledef = {               \
      PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
    ob = PyModule_Create(&moduledef);

  #define PyInt_FromLong PyLong_FromLong
  #define PyInt_AsLong  PyLong_AsLong

#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
    ob = Py_InitModule3(name, methods, doc);
#endif


#endif /* _PYTHONMODULE_H_ */
