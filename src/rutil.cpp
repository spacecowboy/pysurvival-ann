//#include "rutil.h"
#include <RInside.h>                    // for the embedded R via RInside
#include <vector>
#include <stdio.h>
#include <string>

/**
 * Given a list of colNames and a vector of vectors, will create an R data frame
 * where each column is named according to colNames.
 * Be sure that colNames is the same length as listOfColumns.
 */
Rcpp::DataFrame getDataFrameWithNames(Rcpp::CharacterVector *colNames,
                             std::vector<std::vector<double>*> *listOfColumns) {
  Rcpp::List listResult(listOfColumns->size());
  for (unsigned int i = 0; i < listOfColumns->size(); i++) {
    // Add to list
    listResult[i] = *(listOfColumns->at(i));
  }

  // Set names
  listResult.attr("names") = *colNames;

  // Convert to data frame
  Rcpp::DataFrame dfResult(listResult);
  return dfResult;
}

/**
 * Creates a dataframe from the given listOfColumns. Will name each column
 * also. Make sure that each vector in listOfColumns is the same size!
 */
Rcpp::DataFrame getDataFrame(std::vector<std::vector<double>*> *listOfColumns) {
  Rcpp::CharacterVector colNames;
  for (unsigned int i = 0; i < listOfColumns->size(); i++) {
    std::stringstream sstm;
    sstm << "X" << i;
    // Add name
    colNames.push_back(sstm.str());
  }

  // Convert to data frame
  return getDataFrameWithNames(&colNames, listOfColumns);
}

/**
 * Returns a vector with elements matching the array.
 * Up to the caller to delete the vector.
 */
std::vector<double> *arrayToVector(unsigned int size, double *arr) {
  // Initialize with size and set all to zero
  std::vector<double> *vector = new std::vector<double>(size, 0);
  // Assign values as in the array
  vector->assign(arr, arr+size);

  return vector;
}
