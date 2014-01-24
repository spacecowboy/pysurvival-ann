#ifndef _ERRORFUNCTIONS_HPP_
#define _ERRORFUNCTIONS_HPP_

/*
 * An error function returns a number where lower is better
 */
enum class ErrorFunction { ERROR_MSE,
    ERROR_SURV_MSE,
    ERROR_SURV_LIKELIHOOD};

/**
 * An object of this class can be used by error functions to save
 * variables which do not change depending on index.
 */
class ErrorCache {
protected:
  bool needInit;
  /**
   * Sets up the cache
   */
  virtual void init(const double * const targets,
                    const unsigned int length);

public:
  ErrorCache();
  virtual ~ErrorCache();
  /**
   * Will be called before training
   */
  virtual void clear();

  /**
   * Key is value of interest, index is the current point in
   * target/prediction array.
   */
  virtual double getDouble(const int key, const unsigned int index);

  /**
   * Calls init if it has not been called since last clear.
   * This is the only method that is not required to implement,
   * as long as derived class handles "needInit" variable in clear.
   */
  virtual void verifyInit(const double * const targets,
                          const unsigned int length);
};

// Get a cache object suitable for the specified function
// Might be null.
ErrorCache *getErrorCache(ErrorFunction func);


/*
 * Signature for an error function
 */
typedef double (*ErrorFunctionPtr)(const double * const Y,
                                   const unsigned int length,
                                   const unsigned int numOfOutput,
                                   const double * const outputs,
                                   const ErrorCache * const cache);

/*
 * Given an enum value, returns appropriate function pointer
 */
//ErrorFunctionPtr getErrorFunctionPtr(const ErrorFunction val);

// Evaluate the specified function
double getError(ErrorFunction func,
                const double * const Y,
                const unsigned int length,
                const unsigned int numOfOutput,
                const double * const outputs,
                ErrorCache * const cache);
double getError(ErrorFunction func,
                const double * const Y,
                const unsigned int length,
                const unsigned int numOfOutput,
                const double * const outputs);

/**
 * Y.size = length * numOfOutput
 * outputs.size = length * numOfOutput
 * result.size = numOfOutput
 */
void getDerivative(ErrorFunction func,
                   const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   ErrorCache * const cache,
                   double * const result);
void getDerivative(ErrorFunction func,
                   const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   double * const result);

#endif /* _ERRORFUNCTIONS_HPP_ */
