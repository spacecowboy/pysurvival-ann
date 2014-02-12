#include "ErrorFunctionsSurvival.hpp"
#include "ErrorFunctions.hpp"
#include <cmath>
#include "global.hpp"
#include <string>
#include <vector>
#include <exception>
#include <stdio.h>

void getIndicesSortedByTime(const double * const targets,
                            const unsigned int length,
                            std::vector<unsigned int>& sortedIndices)
{
  sortedIndices.clear();
  sortedIndices.reserve(length);
  // Just simple insertion sort
  unsigned int index;
  std::vector<unsigned int>::iterator it;
  double time_i, time_j;
  bool needInsert;
  // Insert zero to start with
  sortedIndices.push_back(0);
  for (unsigned int i = 1; i < length; i++)
  {
    time_i = targets[2 * i];
    needInsert = true;
    for (it=sortedIndices.begin(); it != sortedIndices.end(); it++)
    {
      index = *it;
      time_j = targets[2 * index];

      if (time_i < time_j)
      {
        // Insert at this location
        sortedIndices.insert(it, i);
        needInsert = false;
        // End loop
        break;
      }
    } // inner for
    if (needInsert) {
      sortedIndices.push_back(i);
    }
  } // outer for
}


void getScaledProbs(const double * const targets,
                    const unsigned int length,
                    const std::vector<unsigned int> &sortedIndices,
                    std::vector<double> &scaledProbs)
{
  std::vector<unsigned int>::const_iterator it, laterIt;
  unsigned int index, laterIndex;
  double time, event;
  double atRisk, risk, surv, survDiff;
  std::vector<double> survival(length);
  std::vector<double> probs(length);

  // Scaled probs is a "2-dimensional" array, but implemented as 1D
  scaledProbs.clear();
  scaledProbs.resize(length*length, 0);

  // Set to zero initially
  probs.clear();
  probs.resize(length, 0);

  // Survival starts at 1 and decreases towards zero
  surv = 1.0;
  // First step there has been no change in survival.
  survDiff = 0;

  // First loop we calculate the survival and unscaled probabilities.
  for (it = sortedIndices.begin(); it != sortedIndices.end(); it++)
  {
    index = *it;

    time = targets[2 * index];
    event = targets[2 * index + 1];

    // Calculate survival for index. Will start at 1.
    surv -= survDiff;
    // And save for later
    survival[index] = surv;

    // Reset risk to default values
    atRisk = 0;
    risk = 0;

    // Calculate the risk, which is used to calculate survival
    for (laterIt = it; laterIt != sortedIndices.end(); laterIt++)
    {
      laterIndex = *laterIt;

      // Anything now or later is at risk.
      // Only events can have a risk associated with them.
      if (event) atRisk += 1;
    }
    // Risk is just the inverse
    if (atRisk > 0) {
      risk = 1.0 / atRisk;
    }

    // Probability of event is just risk * survival
    probs[index] = risk * surv;

    // Calculate next survDiff
    survDiff = risk * surv;
  }

  // This loop we calculate the scaled probabilities, based on the
  // survival and probabilities calculated previously. Need the entire
  // probabilities array here.
  for (it = sortedIndices.begin(); it != sortedIndices.end(); it++)
  {
    index = *it;
    getScaledProbsFor(probs, sortedIndices, it,
                      survival.at(index),
                      scaledProbs);
  }
}


void getScaledProbsFor(const std::vector<double> &probs,
                       const std::vector<unsigned int> &sortedIndices,
                       const std::vector<unsigned int>::const_iterator &sortedIt,
                       const double survivalAtIndex,
                       std::vector<double> &scaledProbs)
{
  std::vector<unsigned int>::const_iterator laterIt;
  unsigned int length, index, laterIndex;

  // Don't divide by zeros. Correct value already set.
  if (survivalAtIndex == 0) return;

  length = probs.size();
  index = *sortedIt;

  // Take yourself into account since events only have probability
  // there, while censored have zero probability there.
  for (laterIt = sortedIt; laterIt != sortedIndices.end(); laterIt++)
  {
    laterIndex = *laterIt;

    scaledProbs[index * length + laterIndex] =
      probs[laterIndex] / survivalAtIndex;
  }
}

double getScaledProbAfter(const double * const targets,
                          const unsigned int length,
                          const std::vector<double> &scaledProbs,
                          const std::vector<unsigned int> &sortedIndices,
                          const std::vector<unsigned int>::const_iterator &sortedIt)
{
  double probSum = 0;
  std::vector<unsigned int>::const_iterator laterIt;
  unsigned int index, laterIndex;

  // An event has zero probability of living beyond the last event.
  index = *sortedIt;
  if (targets[2*index + 1]) {
    return 0;
  }

  // Now we know that this is not an event
  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    laterIndex = *laterIt;

    probSum += scaledProbs.at(length * index + laterIndex);
  }

  return 1.0 - probSum;
}

double getPartA(const double * const targets,
                const unsigned int length,
                const std::vector<double> &scaledProbs,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator &sortedIt)
{
  unsigned int index, laterIndex;
  std::vector<unsigned int>::const_iterator laterIt;
  double laterTime, event, Ai;
  Ai = 0;

  index = *sortedIt;

  event = targets[2* index + 1];

  // Events will never have this included in their error.
  if (event) {
    return 0;
  }

  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    laterIndex = *laterIt;
    laterTime = targets[2 * laterIndex];
    // Probs is zero at censored points
    Ai += scaledProbs.at(length * index + laterIndex) *
      std::pow(laterTime, 2.0);
  }

  return Ai;
}

double getPartB(const double * const targets,
                const unsigned int length,
                const std::vector<double> &scaledProbs,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator &sortedIt)
{
  unsigned int index, laterIndex;
  std::vector<unsigned int>::const_iterator laterIt;
  double event, Bi;
  Bi = 0;

  index = *sortedIt;

  event = targets[2* index + 1];

  // Events will never have this included in their error.
  if (event) {
    return 0;
  }

  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    laterIndex = *laterIt;
    // Probs is zero at censored points
    Bi += scaledProbs.at(length * index + laterIndex);
  }

  return Bi;
}

double getPartC(const double * const targets,
                const unsigned int length,
                const std::vector<double> &scaledProbs,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator &sortedIt)
{
  unsigned int index, laterIndex;
  std::vector<unsigned int>::const_iterator laterIt;
  double laterTime, event, Ci;
  Ci = 0;

  index = *sortedIt;

  event = targets[2* index + 1];

  // Events will never have this included in their error.
  if (event) {
    return 0;
  }

  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    laterIndex = *laterIt;
    laterTime = targets[2 * laterIndex];
    // Probs is zero at censored points
    Ci -= 2 * scaledProbs.at(index * length + laterIndex) * laterTime;
  }

  return Ci;
}

double getLikelihoodError(const double targetTime,
                          const double pred,
                          const double lastTime,
                          const double A,
                          const double B,
                          const double C,
                          const double probAfter)
{
  double e;

  // Calculate base error
  e = A + pred * (pred * B + C);

  // Only consider last point if we are underestimating it
  if (pred < lastTime) {

    e += probAfter * std::pow(pred - lastTime, 2.0);
  }

  return e;
}

double getLikelihoodDeriv(const double targetTime,
                          const double pred,
                          const double lastTime,
                          const double B,
                          const double C,
                          const double probAfter)
{
  double d;

  // Calculate base error
  d = 2 * pred * B  + C;

  // Only consider last point if we are underestimating it
  if (pred < lastTime) {
    d += 2 * probAfter * (pred - lastTime);
  }

  return d;
}


SurvErrorCache::SurvErrorCache() :
  ErrorCache(),
  a(std::vector<double>()),
  b(std::vector<double>()),
  c(std::vector<double>()),
  probAfter(std::vector<double>()),
  lastEvent(0),
  lastTime(0)
{
}

SurvErrorCache::~SurvErrorCache()
{
}

void SurvErrorCache::clear()
{
  this->needInit = true;
  // Empty vectors
  this->a.clear();
  this->b.clear();
  this->c.clear();
  this->probAfter.clear();
  this->lastEvent = 0;
  this->lastTime = 0;
}

double SurvErrorCache::getDouble(const int key, const unsigned int index)
{
  double retval = 0;
  switch (key) {
  case KEY_A:
    retval = this->a.at(index);
    break;
  case KEY_B:
    retval = this->b.at(index);
    break;
  case KEY_C:
    retval = this->c.at(index);
    break;
  case KEY_PAFTER:
    retval = this->probAfter.at(index);
    break;
  case KEY_LAST_EVENT:
    retval = this->lastEvent;
    break;
  case KEY_LAST_TIME:
    retval = this->lastTime;
    break;
  default:
    retval = 0;
    break;
  }
  return retval;
}

void SurvErrorCache::init(const double * const targets,
                          const unsigned int length)
{
  // Targets is a 2 dimensional array with [time, event] pairs time is
  // any double value (ideally a positive one) event is either 1.0 or
  // 0.0.  Length is the number of such pairs.

  // Prepare vectors, clear and resize to length number of zeros
  this->a.clear();
  this->a.resize(length, 0);
  this->b.clear();
  this->b.resize(length, 0);
  this->c.clear();
  this->c.resize(length, 0);
  this->probAfter.clear();
  this->probAfter.resize(length, 0);

  // Due to how these calculations work, we have to work in time
  // order.
  std::vector<unsigned int> sortedIndices;
  getIndicesSortedByTime(targets, length, sortedIndices);
  // First, calculate the probability at each event.
  std::vector<double> scaledProbs;
  getScaledProbs(targets, length, sortedIndices, scaledProbs);

  // The rest of the calculations are done for each index
  std::vector<unsigned int>::const_iterator it;
  unsigned int index;

  for (it = sortedIndices.begin(); it != sortedIndices.end(); it++)
  {
    index = *it;

    this->probAfter[index] = getScaledProbAfter(targets, length,
                                                scaledProbs,
                                                sortedIndices, it);
    this->a[index] = getPartA(targets, length, scaledProbs,
                              sortedIndices, it);
    this->b[index] = getPartB(targets, length, scaledProbs,
                              sortedIndices, it);
    this->c[index] = getPartC(targets, length, scaledProbs,
                              sortedIndices, it);
    // Remember last time
    this->lastTime = targets[2 * index];
    this->lastEvent = targets[2 * index + 1];
  }
}

void errorSurvMSE(const double * const Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const double * const outputs,
                  const unsigned int index,
                  double * const errors)
{
  unsigned int i;
  double time, event, output;

  // Only first neuron has none-zero error
  for (i = 0; i < numOfOutput; i++) {
    errors[index + i] = 0;
  }

  // Rest is just square-error if we are underestimating
  time = Y[index];
  event = Y[index + 1];
  output = outputs[index];

  if ((event == 0 && output < time) || event != 0)
  {
    // Censored event which we are underestimating
    // Or real event
    errors[index] = std::pow(output - time, 2.0) / 2.0;
  }
}

void derivativeSurvMSE(const double * const Y,
                       const unsigned int length,
                       const unsigned int numOfOutput,
                       const double * const outputs,
                       const unsigned int index,
                       double * const result)
{
  unsigned int i;
  double time = Y[index];
  double event = Y[index + 1];
  double pred = outputs[index];

  for (i = 0; i < numOfOutput; i++) {
    result[i] = 0;
  }

  // Only for events or underestimated censored
  if ((event == 0 && pred < time) || event != 0)
  {
    // Sign is important. dE = -(time - pred) = pred - time
    result[0] = pred - time;
  }
}

void errorSurvLikelihood(const double * const Y,
                         const unsigned int length,
                         const unsigned int numOfOutput,
                         const double * const outputs,
                         const unsigned int index,
                         ErrorCache * const cache,
                         double * const errors)
{
  double time, event, pred;
  unsigned int n, i;

  // Cache can't be null
  if (NULL == cache) {
    throw std::invalid_argument("ErrorCache was null");
  }
  // Verify that cache has been initialized
  cache->verifyInit(Y, length);

  // Init to zero. Only concerned with first index later.
  for (n = 0; n < numOfOutput; n++) {
    errors[index + n] = 0;
  }

  time = Y[index];
  event = Y[index + 1];
  pred = outputs[index];

  if (event == 1) {
    errors[index] = std::pow(time - pred, 2.0);
  }
  else
  {
    i = index / numOfOutput;
    errors[index] = getLikelihoodError(time, pred,
                                       cache->getDouble(KEY_LAST_TIME, i),
                                       cache->getDouble(KEY_A, i),
                                       cache->getDouble(KEY_B, i),
                                       cache->getDouble(KEY_C, i),
                                       cache->getDouble(KEY_PAFTER, i));
  }
}

void derivativeSurvLikelihood(const double * const Y,
                              const unsigned int length,
                              const unsigned int numOfOutput,
                              const double * const outputs,
                              const unsigned int idx,
                              ErrorCache * const cache,
                              double * const result)
{
  double time, event, pred;
  unsigned int index, i;
  // Cache can't be null
  if (NULL == cache) {
    throw std::invalid_argument("ErrorCache was null");
  }
  // Verify that cache has been initialized
  cache->verifyInit(Y, length);

  // Set all to zero. Only first neuron's error is of concern.
  for(i = 0; i < numOfOutput; i++) result[i] = 0;

  time = Y[idx];
  event = Y[idx + 1];
  pred = outputs[idx];

  // Survival function only cares about first output neuron, so get
  // the correct index to use for the cache.
  index = idx / numOfOutput;

  if (event == 1)
  {
    // Sign is important. dE = d(T - Y)^2 = -2 (T - Y) = 2 (Y - T)
    result[0] = 2 * (pred - time);
  }
  else
  {
    result[0] = getLikelihoodDeriv(time, pred,
                                   cache->getDouble(KEY_LAST_TIME, index),
                                   cache->getDouble(KEY_B, index),
                                   cache->getDouble(KEY_C, index),
                                   cache->getDouble(KEY_PAFTER, index));
  }
}
