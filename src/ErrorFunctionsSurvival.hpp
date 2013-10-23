#ifndef _ERRORFUNCTIONSSURVIVAL_HPP_
#define _ERRORFUNCTIONSSURVIVAL_HPP_

double errorSurvMSE(const double * const Y,
                    const unsigned int length,
                    const unsigned int numOfOutput,
                    const double * const outputs);

void derivativeSurvMSE(const double * const target,
                       const double * const output,
                       const unsigned int length,
                       double * const result);

double errorSurvLikelihood(const double * const Y,
                           const unsigned int length,
                           const unsigned int numOfOutput,
                           const double * const outputs);

void derivativeSurvLikelihood(const double * const target,
                              const double * const output,
                              const unsigned int length,
                              double * const result);

#endif /* _ERRORFUNCTIONSSURVIVAL_HPP_ */
