#ifndef _ERRORFUNCTIONS_HPP_
#define _ERRORFUNCTIONS_HPP_

/*
 * An error function returns a number where lower is better
 */
enum class ErrorFunction { ERROR_MSE,
    ERROR_SURV_MSE,
    ERROR_SURV_LIKELIHOOD};

/*
 * Signature for an error function
 */
typedef double (*ErrorFunctionPtr)(const double * const Y,
                                   const unsigned int length,
                                   const unsigned int numOfOutput,
                                   const double * const outputs);

/*
 * Given an enum value, returns appropriate function pointer
 */
//ErrorFunctionPtr getErrorFunctionPtr(const ErrorFunction val);

// Evaluate the specified function
double getError(ErrorFunction func,
                const double * const Y,
                const unsigned int length,
                const unsigned int numOfOutput,
                const double * const outputs);

void getDerivative(ErrorFunction func,
                   const double * const target,
                   const double * const output,
                   const unsigned int length,
                   double * const result);

#endif /* _ERRORFUNCTIONS_HPP_ */
