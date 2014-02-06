#ifndef _ERRORFUNCTIONSGENERAL_HPP_
#define _ERRORFUNCTIONSGENERAL_HPP_

void errorMSE(const double * const Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const double * const outputs,
              double * const errors);

void derivativeMSE(const double * const Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const double * const outputs,
                   const unsigned int index,
                   double * const result);

#endif /* _ERRORFUNCTIONSGENERAL_HPP_ */
