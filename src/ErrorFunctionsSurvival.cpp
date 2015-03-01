#include "ErrorFunctionsSurvival.hpp"
#include "ErrorFunctions.hpp"
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <stdio.h>

void getIndicesSortedByTime(const std::vector<double> &targets,
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
    time_i = targets.at(2 * i);
    needInsert = true;
    for (it=sortedIndices.begin(); it != sortedIndices.end(); it++)
    {
      index = *it;
      time_j = targets.at(2 * index);

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


void getProbsAndSurvival(const std::vector<double> &targets,
                         const unsigned int length,
                         const std::vector<unsigned int> &sortedIndices,
                         std::vector<double> &probs,
                         std::vector<double> &survival)
{
  std::vector<unsigned int>::const_iterator it, laterIt;
  unsigned int index;
  double event;
  double atRisk, risk, surv, survDiff;

  // Set to zero initially
  probs.clear();
  probs.resize(length, 0);
  survival.clear();
  survival.resize(length, 1);

  // Survival starts at 1 and decreases towards zero
  surv = 1.0;
  // First step there has been no change in survival.
  survDiff = 0;

  // First loop we calculate the survival and unscaled probabilities.
  for (it = sortedIndices.begin(); it != sortedIndices.end(); it++)
  {
    index = *it;

    //time = targets.at(2 * index);
    event = targets.at(2 * index + 1);

    // Calculate survival for index. Will start at 1.
    surv -= survDiff;
    // And save for later
    survival.at(index) = surv;

    // Reset risk to default values
    atRisk = 0;
    risk = 0;

    // Calculate the risk, which is used to calculate survival
    // Anything now or later is at risk.
    // Only events can have a risk associated with them.
    if (event) {
      for (laterIt = it; laterIt != sortedIndices.end(); laterIt++)
      {
        atRisk += 1;
      }
    }
    // Risk is just the inverse
    // But only if any are at risk, else zero
    if (atRisk > 0) {
      risk = 1.0 / atRisk;
    }

    // Probability of event is just risk * survival
    probs.at(index) = risk * surv;

    // Calculate next survDiff
    survDiff = risk * surv;
  }
}


double getScaledProbFor(const std::vector<double> &probs,
                        const std::vector<double> &survival,
                        const std::vector<unsigned int>::const_iterator sortedIt,
                        const std::vector<unsigned int>::const_iterator laterIt)
{
  unsigned int laterIndex, index;
  laterIndex = *laterIt;
  index = *sortedIt;

  // Don't divide by zeros. // TODO is zero correct?
  if (survival.at(index) == 0) return 0;

  // Take yourself into account since events only have probability
  // there, while censored have zero probability there.
  return probs.at(laterIndex) / survival.at(index);
}

double getScaledProbAfter(const std::vector<double> &targets,
                          const std::vector<double> &probs,
                          const std::vector<double> &survival,
                          const std::vector<unsigned int> &sortedIndices,
                          const std::vector<unsigned int>::const_iterator sortedIt)
{
  double probSum = 0;
  std::vector<unsigned int>::const_iterator laterIt;
  unsigned int index;

  // An event has zero probability of living beyond the last event.
  index = *sortedIt;
  if (targets.at(2*index + 1)) {
    return 0;
  }

  // Now we know that this is not an event
  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    probSum += getScaledProbFor(probs, survival, sortedIt, laterIt);
  }

  return 1.0 - probSum;
}

double getPartA(const std::vector<double> &targets,
                const std::vector<double> &probs,
                const std::vector<double> &survival,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator sortedIt)
{
  unsigned int index, laterIndex;
  std::vector<unsigned int>::const_iterator laterIt;
  double laterTime, event, Ai;
  Ai = 0;

  index = *sortedIt;

  event = targets.at(2* index + 1);

  // Events will never have this included in their error.
  if (event) {
    return 0;
  }

  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    laterIndex = *laterIt;
    laterTime = targets.at(2 * laterIndex);
    // Probs is zero at censored points
    Ai += getScaledProbFor(probs, survival, sortedIt, laterIt) *
        std::pow(laterTime, 2.0);
  }

  return Ai;
}

double getPartB(const std::vector<double> &targets,
                const std::vector<double> &probs,
                const std::vector<double> &survival,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator sortedIt)
{
  unsigned int index;
  std::vector<unsigned int>::const_iterator laterIt;
  double event, Bi;
  Bi = 0;

  index = *sortedIt;

  event = targets.at(2* index + 1);

  // Events will never have this included in their error.
  if (event) {
    return 0;
  }

  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    // Probs is zero at censored points
    Bi += getScaledProbFor(probs, survival, sortedIt, laterIt);
  }

  return Bi;
}

double getPartC(const std::vector<double> &targets,
                const std::vector<double> &probs,
                const std::vector<double> &survival,
                const std::vector<unsigned int> &sortedIndices,
                const std::vector<unsigned int>::const_iterator sortedIt)
{
  unsigned int index, laterIndex;
  std::vector<unsigned int>::const_iterator laterIt;
  double laterTime, event, Ci;
  Ci = 0;

  index = *sortedIt;

  event = targets.at(2* index + 1);

  // Events will never have this included in their error.
  if (event) {
    return 0;
  }

  for (laterIt = sortedIt + 1; laterIt != sortedIndices.end(); laterIt++)
  {
    laterIndex = *laterIt;
    laterTime = targets.at(2 * laterIndex);
    // Probs is zero at censored points
    Ci -= 2 * getScaledProbFor(probs, survival, sortedIt, laterIt) * laterTime;
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

void SurvErrorCache::init(const std::vector<double> &targets,
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
  std::vector<unsigned int> *sortedIndices =
      new std::vector<unsigned int>(length);
  getIndicesSortedByTime(targets, length, *sortedIndices);

  // First, calculate the probability at each event.
  std::vector<double> *probs = new std::vector<double>(length);
  std::vector<double> *survival = new std::vector<double>(length);
  getProbsAndSurvival(targets, length, *sortedIndices, *probs, *survival);

  // The rest of the calculations are done for each index
  std::vector<unsigned int>::const_iterator it;
  unsigned int i, index;

  // This would be the idiomatic c++ way of doing it. but a simple int
  // is used to enable OpenMP parallelization.
  //for (it = sortedIndices->begin(); it != sortedIndices->end(); it++)

# pragma omp parallel for default(shared) private(it, index) schedule(dynamic)
  for (i = 0; i < length; i++) {
    it = sortedIndices->begin() + i;
    index = *it;

    this->probAfter.at(index) = getScaledProbAfter(targets,
                                                   *probs,
                                                   *survival,
                                                   *sortedIndices, it);
    this->a.at(index) = getPartA(targets, *probs, *survival,
                                 *sortedIndices, it);
    this->b.at(index) = getPartB(targets, *probs, *survival,
                                 *sortedIndices, it);
    this->c.at(index) = getPartC(targets, *probs, *survival,
                                 *sortedIndices, it);
  }
  // End parallel

  // Set the last values
  it = sortedIndices->end();
  it--;
  index = *it;
  this->lastTime = targets.at(2 * index);
  this->lastEvent = targets.at(2 * index + 1);

  delete sortedIndices;
  delete probs;
  delete survival;
}

void errorSurvMSE(const std::vector<double> &Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const std::vector<double> &outputs,
                  const unsigned int index,
                  std::vector<double> &errors)
{
  unsigned int i;
  double time, event, output;

  // Only first neuron has none-zero error
  for (i = 0; i < numOfOutput; i++) {
    errors.at(index + i) = 0;
  }

  // Rest is just square-error if we are underestimating
  time = Y.at(index);
  event = Y.at(index + 1);
  output = outputs.at(index);

  if ((event == 0 && output < time) || event != 0)
  {
    // Censored event which we are underestimating
    // Or real event
    errors.at(index) = std::pow(output - time, 2.0) / 2.0;
  }
}

void derivativeSurvMSE(const std::vector<double> &Y,
                       const unsigned int length,
                       const unsigned int numOfOutput,
                       const std::vector<double> &outputs,
                       const unsigned int index,
                       std::vector<double>::iterator result)
{
  unsigned int i;
  double time = Y.at(index);
  double event = Y.at(index + 1);
  double pred = outputs.at(index);

  for (i = 0; i < numOfOutput; i++) {
    *(result + i) = 0;
  }

  // Only for events or underestimated censored
  if ((event == 0 && pred < time) || event != 0)
  {
    // Sign is important. dE = -(time - pred) = pred - time
    *(result) = pred - time;
  }
}

void errorSurvLikelihood(const std::vector<double> &Y,
                         const unsigned int length,
                         const unsigned int numOfOutput,
                         const std::vector<double> &outputs,
                         const unsigned int index,
                         ErrorCache * const cache,
                         std::vector<double> &errors)
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
    errors.at(index + n) = 0;
  }

  time = Y.at(index);
  event = Y.at(index + 1);
  pred = outputs.at(index);

  if (event == 1) {
    errors.at(index) = std::pow(time - pred, 2.0);
  }
  else
  {
    i = index / numOfOutput;
    errors.at(index) = getLikelihoodError(time, pred,
                                       cache->getDouble(KEY_LAST_TIME, i),
                                       cache->getDouble(KEY_A, i),
                                       cache->getDouble(KEY_B, i),
                                       cache->getDouble(KEY_C, i),
                                       cache->getDouble(KEY_PAFTER, i));
  }
}

void derivativeSurvLikelihood(const std::vector<double> &Y,
                              const unsigned int length,
                              const unsigned int numOfOutput,
                              const std::vector<double> &outputs,
                              const unsigned int idx,
                              ErrorCache * const cache,
                              std::vector<double>::iterator result)
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
  for(i = 0; i < numOfOutput; i++) *(result + i) = 0;

  time = Y.at(idx);
  event = Y.at(idx + 1);
  pred = outputs.at(idx);

  // Survival function only cares about first output neuron, so get
  // the correct index to use for the cache.
  index = idx / numOfOutput;

  if (event == 1)
  {
    // Sign is important. dE = d(T - Y)^2 = -2 (T - Y) = 2 (Y - T)
    *(result) = 2 * (pred - time);
  }
  else
  {
    *(result) = getLikelihoodDeriv(time, pred,
                                   cache->getDouble(KEY_LAST_TIME, index),
                                   cache->getDouble(KEY_B, index),
                                   cache->getDouble(KEY_C, index),
                                   cache->getDouble(KEY_PAFTER, index));
  }
}
