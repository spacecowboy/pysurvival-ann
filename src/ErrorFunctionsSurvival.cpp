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


void getProbs(const double * const targets,
              const unsigned int length,
              const std::vector<unsigned int> &sortedIndices,
              std::vector<double> &probs)
{
  std::vector<unsigned int>::const_iterator it, laterIt;
  unsigned int index, laterIndex;
  double time, event;
  double atRisk, risk, survival, survDiff;

  // Set to zero initially
  probs.clear();
  probs.resize(length, 0);

  // Survival starts at 1 and decreases towards zero
  survival = 1.0;
  // First step there has been no change in survival.
  survDiff = 0;

  for (it = sortedIndices.begin(); it != sortedIndices.end(); it++)
  {
    index = *it;

    time = targets[2 * index];
    event = targets[2 * index + 1];

    // Calculate survival for index. Will start at 1.
    survival -= survDiff;

    // Reset risk to default values
    atRisk = 0;
    risk = 0;

    // Calculate the risk, which is used to calculate survival
    for (laterIt = it; laterIt != sortedIndices.end(); laterIt++)
    {
      laterIndex = *laterIt;

      // Anything now or later is at risk
      if (event) atRisk += 1;
    }
    // Risk is just the inverse
    if (atRisk > 0) {
      risk = 1.0 / atRisk;
    }

    // Probability of event is just risk * survival
    probs[index] = risk * survival;

    // Calculate next survDiff
    survDiff = risk * survival;
  }
}

double getScaledProbAfter(const double * const targets,
                          const unsigned int length,
                          const std::vector<double> &probs,
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

    probSum += probs.at(laterIndex);
  }

  return 1.0 - probSum;
}

double getPartA(const double * const targets,
                const unsigned int length,
                const std::vector<double> &probs,
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
    Ai += probs.at(laterIndex) * std::pow(laterTime, 2.0);
  }

  return Ai;
}

double getPartB(const double * const targets,
                const unsigned int length,
                const std::vector<double> &probs,
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
    Bi += probs.at(laterIndex);
  }

  return Bi;
}

double getPartC(const double * const targets,
                const unsigned int length,
                const std::vector<double> &probs,
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
    Ci -= 2 * probs.at(laterIndex) * laterTime;
  }

  return Ci;
}

double getLikelihoodError(const double * const targets,
                          const unsigned int length,
                          const unsigned int index,
                          const double pred,
                          const double A,
                          const double B,
                          const double C,
                          const double probAfter)
{
  double e, time, lastTime;

  // Fetch the times for convenience
  time = targets[2 * index];
  lastTime = targets[2 * (length - 1)];

  // Calculate base error
  e = A + pred * (pred * B + C);

  // Should this compare the prediction instead?
  if (time < lastTime) {
    e += probAfter * std::pow(pred - lastTime, 2.0);
  }

  return e;
}

double getLikelihoodDeriv(const double * const targets,
                          const unsigned int length,
                          const unsigned int index,
                          const double pred,
                          const double B,
                          const double C,
                          const double probAfter)
{
  double d, time, lastTime;

  // Fetch the times for convenience
  time = targets[2 * index];
  lastTime = targets[2 * (length - 1)];

  // Calculate base error
  d = 2 * pred * B  + C;

  // Should this compare the prediction instead?
  if (time < lastTime) {
    d += probAfter * (pred - lastTime);
  }

  return d;
}


SurvErrorCache::SurvErrorCache() :
  ErrorCache(),
  a(std::vector<double>()),
  b(std::vector<double>()),
  c(std::vector<double>()),
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
    //std::cout << "Got A ";
    retval = this->a.at(index);
    break;
  case KEY_B:
    //std::cout << "Got B ";
    retval = this->b.at(index);
    break;
  case KEY_C:
    //std::cout << "Got C ";
    retval = this->c.at(index);
    break;
  case KEY_PAFTER:
    //std::cout << "Got PAfter ";
    retval = this->probAfter.at(index);
    break;
  case KEY_LAST_EVENT:
    //std::cout << "Got Event ";
    retval = this->lastEvent;
    break;
  case KEY_LAST_TIME:
    //std::cout << "Got time ";
    retval = this->lastTime;
    break;
  default:
    retval = 0;
    break;
  }
  //std::cout << "Got double " << retval << std::endl;
  return retval;
}

void SurvErrorCache::init(const double * const targets,
                          const unsigned int length)
{
  // Targets is a 2 dimensional array with [time, event] pairs time is
  // any double value (ideally a positive one) event is either 1.0 or
  // 0.0.  Length is the number of such pairs.

  // Prepeare vectors, clear and resize to length number of zeros
  this->a.clear();
  this->a.resize(length, 0);
  this->b.clear();
  this->b.resize(length, 0);
  this->c.clear();
  this->c.resize(length, 0);
  this->probAfter.clear();
  this->probAfter.resize(length, 0);

  // Due to how these calculations work, we have to work in time order
  std::vector<unsigned int> sortedIndices;
  getIndicesSortedByTime(targets, length, sortedIndices);

  // Number of patients at risk
  //double atRisk[length];
  std::vector<double> atRisk(length, 0);
  // Risk measure
  //double risk[length];
  std::vector<double> risk(length, 0);
  // Survival fraction
  //double surv[length];
  std::vector<double> surv(length, 1);
  // Probability of event (at events) is just risk * surv
  //double prob[length];
  std::vector<double> prob(length, 0);
  // Difference to last index in survival
  double survDiff = 0;
  // Surv at last point
  double survAfter = 0;
  unsigned int index, laterIndex, prevIndex;
  bool firstEvent = true;

  // Keep track of later events for all indices This is a 2d vector
  std::vector< std::vector<unsigned int> > allLaterEvents(length);

  double time, event, laterTime, laterEvent;

  std::vector<unsigned int>::const_iterator it, laterIt;

  // Start with first time as last
  this->lastTime = targets[0];
  this->lastEvent = targets[1];

  // Need to set this in case we have some patients that are censored first
  prevIndex = 2*length;

  // First calculate the risk and survival
  for (it = sortedIndices.begin(); it != sortedIndices.end(); it++)
  {
    index = *it;

    time = targets[2 * index];
    event = targets[2 * index + 1];

    // Want to store the very last time point for later
    if (time > this->lastTime) {
      this->lastTime = time;
      this->lastEvent = event;
    }

    // Init to zero
    atRisk[index] = 0;
    risk[index] = 0;
    prob[index] = 0;
    surv[index] = 0;

    // Look at later events.
    for (laterIt = it + 1; laterIt != sortedIndices.end(); laterIt++)
    {
      laterIndex = *laterIt;

      laterTime = targets[2 * laterIndex];
      laterEvent = targets[2 * laterIndex + 1];

      // Remember later events
      if (laterTime > time && laterEvent == 1) {
        // Access later events array for current index and add this
        // later index
        allLaterEvents[index].push_back(laterIndex);
      }
      // Anything now or later is at risk
      if (laterTime >= time && event) {
        atRisk[index] += 1;
      }
    }

    // Calculate survival and risk
    if (event == 1) {
      // Risk is just inverse of number at risk
      risk[index] = 1.0 / atRisk.at(index);

      if (firstEvent) {
        // By definition
        surv[index] = 1.0;
        firstEvent = false;
      }
      else {
        surv[index] = surv.at(prevIndex) + survDiff;
      }

      prob[index] = risk.at(index) * surv.at(index);

      // Calculate surv diff for next round
      survDiff =  - risk.at(index) * surv.at(index);
      // Remember previous index next time
      prevIndex = index;
    }
    else {
      if (firstEvent) {
        // No event has been seen yet
        risk[index] = 0.0;
        surv[index] = 1.0;
      }
      else {
        // Not an event, nothing has changed
        risk[index] = risk.at(prevIndex);
        surv[index] = surv.at(prevIndex);
      }
      // Zero probability of event
      prob[index] = 0;
    }
  }

  //if (targets[2 * sortedIndices.back() + 1] == 1) {
  if (this->lastEvent == 1)
  {
    // Last point is an event, by definition pAfter is zero
    survAfter = 0;
  }
  else {
    // Last point is censored
    survAfter = surv.at(sortedIndices.back()) + survDiff;
  }

  // Now, calculate the precomputed parts of each index
  // Order is irrelevant here
  for (index = 0; index < length; index++) {
    time = targets[2 * index];
    event = targets[2 * index + 1];

    if (event) {
      // This is already done in initialization step
      this->a[index] = 0;
      this->b[index] = 0;
      this->c[index] = 0;
      this->probAfter[index] = 0;
    }
    else {
      // Starts at probability 1
      this->probAfter[index] = 1;
      this->a[index] = 0;
      this->b[index] = 0;
      this->c[index] = 0;
      // Iterate over later events
      for (laterIt = allLaterEvents[index].begin();
           laterIt != allLaterEvents[index].end();
           laterIt++)
      {
        laterIndex = *laterIt;
        // Decrease for each event probability
        this->probAfter[index] -= prob.at(laterIndex);
        // a_i = sum[ prob_i * T_i^2 ]
        this->a[index] += (prob.at(laterIndex) *
                           (targets[2 * laterIndex] * targets[2 * laterIndex]));
        // b_i = sum[ prob_i ]
        this->b[index] += prob.at(laterIndex);
        // c_i = sum[ prob_i * -2 * T_i ]
        this->c[index] += -2 * prob.at(laterIndex) * targets[2 * laterIndex];
      }
    }
  }
}

double errorSurvMSE(const double * const Y,
                    const unsigned int length,
                    const unsigned int numOfOutput,
                    const double * const outputs)
{
  unsigned int i;
  double error = 0, time, event, output;
  // Evaluate each input set
  // Average over all inputs and number of outputs
  for (i = 0; i < length; i++)
  {
    // Plus two because there is an event column as well
    time = Y[2*i];
    event = Y[2*i + 1];
    // No event column in predictions
    output = outputs[2*i];
    if ((event == 0 && output < time) || event != 0)
    {
      // Censored event which we are underestimating
      // Or real event
      error += std::pow(output - time, 2.0) / 2.0;
    }
  }
  return error / ((double) length * numOfOutput);
}

void derivativeSurvMSE(const double * const Y,
                       const unsigned int length,
                       const unsigned int numOfOutput,
                       const double * const outputs,
                       const unsigned int index,
                       double * const result)
{
  double time = Y[index];
  double event = Y[index + 1];
  double pred = outputs[index];

  result[0] = 0;
  result[1] = 0;

  // Only for events or underestimated censored
  if ((event == 0 && pred < time) || event != 0)
  {
    // Sign is important. dE = -(time - pred) = pred - time
    result[0] = pred - time;
  }
}

double errorSurvLikelihood(const double * const Y,
                           const unsigned int length,
                           const unsigned int numOfOutput,
                           const double * const outputs,
                           ErrorCache * const cache)
{
  // Cache can't be null
  if (NULL == cache) {
    throw std::invalid_argument("ErrorCache was null");
  }
  // Verify that cache has been initialized
  cache->verifyInit(Y, length);

  double error = 0;

  for (int i = 0; i < length; i++)
  {
    double time = Y[2 * i];
    double event = Y[2 * i + 1];
    double pred = outputs[numOfOutput * i];
    double local_error = 0;

    if (event == 1)
    {
      local_error = std::pow(time - pred, 2.0);
    }
    else
    {
      local_error += cache->getDouble(KEY_A, i);
      local_error += pred * (pred * cache->getDouble(KEY_B, i) +
                             cache->getDouble(KEY_C, i));

      // Error due to tail-censored elements
      if (cache->getDouble(KEY_LAST_EVENT, i) == 0 &&
          pred < cache->getDouble(KEY_LAST_TIME, i))
      {
        local_error += cache->getDouble(KEY_PAFTER, i) *
          std::pow(cache->getDouble(KEY_LAST_TIME, i) - pred, 2.0);
      }
    }
    error += local_error;
  }

  return error;
}

void derivativeSurvLikelihood(const double * const Y,
                              const unsigned int length,
                              const unsigned int numOfOutput,
                              const double * const outputs,
                              const unsigned int idx,
                              ErrorCache * const cache,
                              double * const result)
{
  // Cache can't be null
  if (NULL == cache) {
    throw std::invalid_argument("ErrorCache was null");
  }
  // Verify that cache has been initialized
  cache->verifyInit(Y, length);

  double time = Y[idx];
  double event = Y[idx + 1];
  double pred = outputs[idx];

  // Survival function only cares about first output neuron
  unsigned int index = idx / numOfOutput;

  // Only first neuron is used
  result[1] = 0;
  if (event == 1)
  {
    // Sign is important. dE = d(T - Y)^2 = -2 (T - Y) = 2 (Y - T)
    result[0] = 2 * (pred - time);
    // This print out is reported reasonable: -10 to 10 or so.
    //std::cout << std::endl << "unc " << result[0] << std::endl;
  }
  else
  {
    // Later events
    std::cout << std::endl << "B " << cache->getDouble(KEY_B, index);
    result[0] = 2 * pred * cache->getDouble(KEY_B, index);
    std::cout << std::endl << "C " << cache->getDouble(KEY_C, index);
    result[0] += cache->getDouble(KEY_C, index);

    std::cout << std::endl << "PA " << cache->getDouble(KEY_PAFTER, index);
    std::cout << std::endl << "LE " << cache->getDouble(KEY_LAST_EVENT, index);
    std::cout << std::endl << "LT " << cache->getDouble(KEY_LAST_TIME, index);
    // Tail censored ones
    if (cache->getDouble(KEY_LAST_EVENT, index) == 0 &&
        pred < cache->getDouble(KEY_LAST_TIME, index))
    {
      result[0] += cache->getDouble(KEY_PAFTER, index) * 2 *
        (cache->getDouble(KEY_LAST_TIME, index) - pred);
    }
    // This print out is reported as "-nan"
    std::cout << std::endl << "cen " << result[0] << std::endl;
  }
}
