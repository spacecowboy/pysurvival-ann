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

// Likelihood error help functions. Makes it easier to test.

/**
 * sortedIndices will have the indices in targets sorted by ascending
 * order of time. You will have to do targets[2*i] however. sortedIndices
 * will be < length.
 */
void getIndicesSortedByTime(const double * const targets,
                            const unsigned int lengt,
                            std::vector<unsigned int>& sortedIndices);

/*
 * This calculates the scaled probabity to have an event at each time
 * point, for each patient. Time points with events will have a
 * non-zero probability while time points with censored events will
 * have zero probability.
 *
 * The probabilities will only sum to one if the last time point had
 * an event.
 *
 * scaledProbs is cleared and set. It will have size length*length.
 */
void getScaledProbs(const double * const targets,
                    const unsigned int length,
                    const std::vector<unsigned int> &sortedIndices,
                    std::vector<double> &scaledProbs);

/*
 * Calculate the scaled probability values for the patient at index with
 * specified survival at that index. Scaled probabilities are
 * calculated as future probabilities divided by survival at index.
 *
 * scaledProbs is expected to be initialized to zero of size:
 * length*length, where length = probs.size(). This function will set
 * the values in range [index*length, (index+1)*length].
 */
void getScaledProbsFor(const std::vector<double> &probs,
                       const std::vector<unsigned int> &sortedIndices,
                       const std::vector<unsigned int>::const_iterator &it,
                       const double survivalAtIndex,
                       std::vector<double> &scaledProbs);

/*
 * Calculate the probability to survive longer than the last
 * event. This is the conditional probability S_{n+1} | p_i.
 *
 * Typically the probability is just p_{after} = 1.0 -
 * sum(probs). However, because the patient in question will have
 * lived to some time t_i (the censoring time), we have to adjust the
 * probability to take that into account. Thus making the result:
 *
 * =This I am not sure about! Isn't prob_i zero?!=
 * p_{after} = 1.0 - ( sum(probs) / prob_i ).
 *
 * sortedIt is an iterator of sortedIndices. Only probabilities
 * occuring afterwards are considered.
 */
double getScaledProbAfter(const double * const targets,
                          const unsigned int length,
                          const std::vector<double> &probs,
                          const std::vector<unsigned int> &sortedIndices,
                          const std::vector<unsigned int>::const_iterator &sortedIt);

/*
 * Calculate A of the equations for the index pointed to by sortedIt.
 */
double getPartA(const double * const targets,
                const unsigned int length,
                const std::vector<double> &probs,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator &sortedIt);

/*
 * Calculate B of the equations for the index pointed to by sortedIt.
 */
double getPartB(const double * const targets,
                const unsigned int length,
                const std::vector<double> &probs,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator &sortedIt);

/*
 * Calculate C of the equations for the index pointed to by sortedIt.
 */
double getPartC(const double * const targets,
                const unsigned int length,
                const std::vector<double> &probs,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator &sortedIt);

/*
 * Calculate the error of the prediction
 */
double getLikelihoodError(const double targetTime,
                          const double pred,
                          const double lastTime,
                          const double A,
                          const double B,
                          const double C,
                          const double probAfter);

/*
 * Calculate the derivative of the error for prediction.
 */
double getLikelihoodDeriv(const double targetTime,
                          const double pred,
                          const double lastTime,
                          const double B,
                          const double C,
                          const double probAfter);


// Errors and their derivatives

void errorSurvMSE(const double * const Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const double * const outputs,
                  double * const errors);

void derivativeSurvMSE(const double * const Y,
                       const unsigned int length,
                       const unsigned int numOfOutput,
                       const double * const outputs,
                       const unsigned int index,
                       double * const result);

void errorSurvLikelihood(const double * const Y,
                         const unsigned int length,
                         const unsigned int numOfOutput,
                         const double * const outputs,
                         ErrorCache * const cache,
                         double * const errors);

void derivativeSurvLikelihood(const double * const Y,
                              const unsigned int length,
                              const unsigned int numOfOutput,
                              const double * const outputs,
                              const unsigned int index,
                              ErrorCache * const cache,
                              double * const result);

#endif /* _ERRORFUNCTIONSSURVIVAL_HPP_ */
