#ifndef _GENETICFITNESS_HPP_
#define _GENETICFITNESS_HPP_


/*
 * A fitness function returns a number where higher is better
 */
enum class FitnessFunction { FITNESS_MSE,
        FITNESS_CINDEX,
        FITNESS_MSE_CENS };

/*
 * Signature for a fitness function
 */
typedef double (*FitnessFunctionPtr)(const double * const X,
                                     const double * const Y,
                                     const unsigned int length,
                                     const unsigned int numOfOutput,
                                     const double * const outputs);

/*
 * Given an enum value, returns appropriate function pointer
 */
FitnessFunctionPtr getFitnessFunctionPtr(const FitnessFunction val);

double fitnessMSE(const double * const X,
                  const double * const Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const double * const outputs);

double fitnessCIndex(const double * const X,
                     const double * const Y,
                     const unsigned int length,
                     const unsigned int numOfOutput,
                     const double * const outputs);

double fitnessMSECens(const double * const X,
                      const double * const Y,
                      const unsigned int length,
                      const unsigned int numOfOutput,
                      const double * const outputs);

#endif // _GENETICFITNESS_H_
