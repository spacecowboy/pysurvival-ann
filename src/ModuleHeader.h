// It is required to have this defined to something
// for numpy to function when module is defined over
// several files.
// Only the PythonModule file should import this
// Other files should import ExtensionHeader.h instead
#ifndef MODULEHEADER_H_
#define MODULEHEADER_H_

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_ann_fowegjr

#endif //MODULEHEADER_H_
