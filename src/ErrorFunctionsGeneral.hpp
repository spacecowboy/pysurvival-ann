#ifndef _ERRORFUNCTIONSGENERAL_HPP_
#define _ERRORFUNCTIONSGENERAL_HPP_

double errorMSE(const double * const Y,
                const unsigned int length,
                const unsigned int numOfOutput,
                const double * const outputs);

void derivativeMSE(const double * const target,
                   const double * const output,
                   const unsigned int length,
                   double * const result);

#endif /* _ERRORFUNCTIONSGENERAL_HPP_ */
