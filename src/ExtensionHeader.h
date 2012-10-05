// This file should be included by all files which deal with
// numpy objects. It must be included BEFORE arrayobject
// is included.
#ifndef EXTENSIONHEADER_H_
#define EXTENSIONHEADER_H_

#define NO_IMPORT_ARRAY

// Also need unique_symbol which is defined here
#include "ModuleHeader.h"
//#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_ann_mod

#endif //EXTENSIONHEADER_H_

