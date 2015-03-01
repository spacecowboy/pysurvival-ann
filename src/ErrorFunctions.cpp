#include "ErrorFunctions.hpp"
#include "ErrorFunctionsGeneral.hpp"
#include "ErrorFunctionsSurvival.hpp"
#include <cmath>
#include <stdexcept>

// Implement a basic cache
ErrorCache::ErrorCache():
  needInit(true)
{}

ErrorCache::~ErrorCache()
{}

void ErrorCache::clear()
{
  throw std::invalid_argument("You should use a derived class\
 instead of ErrorCache directly.");
}

void ErrorCache::init(const std::vector<double> &targets,
                      const unsigned int length)
{
  throw std::invalid_argument("You should use a derived class\
 instead of ErrorCache directly.");
}

double ErrorCache::getDouble(const int key,
                           const unsigned int index)
{
  throw std::invalid_argument("You should use a derived class\
 instead of ErrorCache directly.");
}

void ErrorCache::verifyInit(const std::vector<double> &targets,
                            const unsigned int length)
{
  if (needInit) {
    this->init(targets, length);
    needInit = false;
  }
}

// Get a suitable Error Cache
ErrorCache *getErrorCache(ErrorFunction func)
{
  ErrorCache *cache = nullptr;
  switch (func) {
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    cache = new SurvErrorCache();
    break;
  default:
    cache = nullptr;
    break;
  }
  return cache;
}

void averagePatternError(const std::vector<double> &errors,
                         const unsigned int length,
                         const unsigned int numOfOutput,
                         std::vector<double> &avgErrors)
{
  unsigned int i, n;
  // Each output neuron is averaged separately
  for (n = 0; n < numOfOutput; n++) {
    avgErrors.at(n) = 0;
    for (i = 0; i < length; i++) {
      avgErrors.at(n) += errors.at(i * numOfOutput + n);
    }
    avgErrors.at(n) /= (double) length;
  }
}

void getAllErrors(ErrorFunction func,
                  const std::vector<double> &Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const std::vector<double> &outputs,
                  ErrorCache * const cache,
                  std::vector<double> &errors)
{
  unsigned int index;
  // Iterate over all patterns
  for (index = 0;
       index < length * numOfOutput;
       index += numOfOutput)
  {
    getError(func, Y, length, numOfOutput, outputs, index, cache, errors);
  }
}

void getAllErrors(ErrorFunction func,
                  const std::vector<double> &Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const std::vector<double> &outputs,
                  std::vector<double> &errors)
{
  // Get a cache
  ErrorCache *cache = getErrorCache(func);
  // Calculate error
  getAllErrors(func, Y, length, numOfOutput, outputs, cache, errors);
  // If a cache was allocated, deallocate it again
  if (cache != nullptr) {
    delete cache;
  }
}

// Evaluate the specified function
void getError(ErrorFunction func,
              const std::vector<double> &Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const std::vector<double> &outputs,
              const unsigned int index,
              std::vector<double> &errors)
{
  // Get a cache
  ErrorCache *cache = getErrorCache(func);
  // Calculate error
  getError(func, Y, length, numOfOutput, outputs, index, cache, errors);
  // If a cache was allocated, deallocate it again
  if (cache != nullptr) {
    delete cache;
  }
}
void getError(ErrorFunction func,
              const std::vector<double> &Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const std::vector<double> &outputs,
              const unsigned int index,
              ErrorCache * const cache,
              std::vector<double> &errors)
{
  switch (func) {
  case ErrorFunction::ERROR_SURV_MSE:
    errorSurvMSE(Y, length, numOfOutput, outputs, index, errors);
    break;
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    errorSurvLikelihood(Y, length, numOfOutput, outputs,
                        index, cache, errors);
    break;
  case ErrorFunction::ERROR_MSE:
  default:
    errorMSE(Y, length, numOfOutput, outputs, index, errors);
    break;
  }
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 * result.size = length * numOfOutput
 */
void getDerivative(ErrorFunction func,
                   const std::vector<double> &Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const std::vector<double> &outputs,
                   const unsigned int index,
                   std::vector<double>::iterator result)
{
  // Get a cache
  ErrorCache *cache = getErrorCache(func);
  // Calculate derivative
  getDerivative(func, Y, length, numOfOutput, outputs, index, cache, result);
  // If a cache was allocated, deallocate it again
  if (cache != nullptr) {
    delete cache;
  }
}
void getDerivative(ErrorFunction func,
                   const std::vector<double> &Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const std::vector<double> &outputs,
                   const unsigned int index,
                   ErrorCache * const cache,
                   std::vector<double>::iterator result)
{
  switch (func) {
  case ErrorFunction::ERROR_SURV_MSE:
    derivativeSurvMSE(Y, length, numOfOutput, outputs, index, result);
    break;
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    derivativeSurvLikelihood(Y, length, numOfOutput,
                             outputs, index, cache, result);
    break;
  case ErrorFunction::ERROR_MSE:
  default:
    derivativeMSE(Y, length, numOfOutput, outputs, index, result);
    break;
  }
}
