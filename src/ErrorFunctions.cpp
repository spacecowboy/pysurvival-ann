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

void ErrorCache::init(const double * const targets,
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

void ErrorCache::verifyInit(const double * const targets,
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
  ErrorCache *cache = NULL;
  switch (func) {
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    cache = new SurvErrorCache();
    break;
  default:
    cache = NULL;
    break;
  }
  return cache;
}

// Evaluate the specified function
void getError(ErrorFunction func,
              const double * const Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const double * const outputs,
              double * const errors)
{
  // Get a cache
  ErrorCache *cache = getErrorCache(func);
  // Calculate error
  getError(func, Y, length, numOfOutput, outputs, NULL, errors);
  // If a cache was allocated, deallocate it again
  if (cache != NULL) {
    delete cache;
  }
}
void getError(ErrorFunction func,
              const double * const Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const double * const outputs,
              ErrorCache * const cache,
              double * const errors)
{
  switch (func) {
  case ErrorFunction::ERROR_SURV_MSE:
    errorSurvMSE(Y, length, numOfOutput, outputs, errors);
    break;
  case ErrorFunction::ERROR_SURV_LIKELIHOOD:
    errorSurvLikelihood(Y, length, numOfOutput,
                        outputs, cache, errors);
    break;
  case ErrorFunction::ERROR_MSE:
  default:
    errorMSE(Y, length, numOfOutput, outputs, errors);
    break;
  }
}

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 * result.size = length * numOfOutput
 */
void getDerivative(ErrorFunction func,
                   const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   double * const result)
{
  getDerivative(func, Y, length, numOfOutput, outputs, index, NULL, result);
}
void getDerivative(ErrorFunction func,
                   const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   ErrorCache * const cache,
                   double * const result)
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
