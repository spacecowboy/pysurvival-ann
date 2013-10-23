#ifndef _RPROPNETWORK_HPP_
#define _RPROPNETWORK_HPP_

#include "MatrixNetwork.hpp"
#include "ErrorFunctions.hpp"
#include <vector>

using namespace std;

class RPropNetwork: public MatrixNetwork {
 public:

  unsigned int maxEpochs;
  double maxError;
  ErrorFunction errorFunction;

  // Methods
  RPropNetwork(const unsigned int numOfInputs,
               const unsigned int numOfHidden,
               const unsigned int numOfOutputs);

  /*
 * Expects the X and Y to be of equal number of rows.
 */
  virtual void learn(const double * const X,
                     const double * const Y,
                     const unsigned int length);


  // Getters and Setters
  unsigned int getMaxEpochs() const;
  void setMaxEpochs(unsigned int maxEpochs);
  double getMaxError() const;
  void setMaxError(double maxError);
  ErrorFunction getErrorFunction() const;
  void setErrorFunction(ErrorFunction val);
};

#endif /* _RPROPNETWORK_HPP_ */
