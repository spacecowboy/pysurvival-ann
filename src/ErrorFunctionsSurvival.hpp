#ifndef _ERRORFUNCTIONSSURVIVAL_HPP_
#define _ERRORFUNCTIONSSURVIVAL_HPP_

#include "ErrorFunctions.hpp"
#include <vector>

// KEY values
// A_i
const int KEY_A = 1;
// B_i
const int KEY_B = 2;
// C_i
const int KEY_C = 3;
// S_{n+1} | t_i
const int KEY_PAFTER = 4;
// Event of last one
const int KEY_LAST_EVENT = 5;
// Time of last one
const int KEY_LAST_TIME = 6;

class SurvErrorCache: public ErrorCache {
protected:
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> probAfter;
  double lastEvent;
  double lastTime;

  virtual void init(const double * const targets,
                    const unsigned int length);
public:
  SurvErrorCache();
  virtual ~SurvErrorCache();

  /**
   * Will be called before training
   */
  virtual void clear();

  /**
   * Key is value of interest, index is the current point in
   * target/prediction array.
   */
  virtual double getDouble(const int key, const unsigned int index);
};

double errorSurvMSE(const double * const Y,
                    const unsigned int length,
                    const unsigned int numOfOutput,
                    const double * const outputs);

void derivativeSurvMSE(const double * const Y,
                       const unsigned int length,
                       const unsigned int numOfOutput,
                       const double * const outputs,
                       const unsigned int index,
                       double * const result);

double errorSurvLikelihood(const double * const Y,
                           const unsigned int length,
                           const unsigned int numOfOutput,
                           const double * const outputs,
                           ErrorCache * const cache);

void derivativeSurvLikelihood(const double * const Y,
                              const unsigned int length,
                              const unsigned int numOfOutput,
                              const double * const outputs,
                              const unsigned int index,
                              ErrorCache * const cache,
                              double * const result);

#endif /* _ERRORFUNCTIONSSURVIVAL_HPP_ */
