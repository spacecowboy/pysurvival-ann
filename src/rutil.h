#ifndef _RUTIL_H_
#define _RUTIL_H_

#include <RInside.h>                    // for the embedded R via RInside
#include <vector>

Rcpp::DataFrame getDataFrameWithNames(Rcpp::CharacterVector *colNames,
                                      std::vector<std::vector<double>*> *listOfColumns);

Rcpp::DataFrame getDataFrame(std::vector<std::vector<double>*> *listOfColumns);

std::vector<double> *arrayToVector(unsigned int size, double *arr);

#endif /* _RUTIL_H_ */
