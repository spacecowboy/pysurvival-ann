#ifndef _GENETICFITNESS_HPP_
#define _GENETICFITNESS_HPP_


/*
 * A fitness function returns a number where higher is better
 */
enum class FitnessFunction { FITNESS_MSE,
    FITNESS_CINDEX,
    FITNESS_MSE_CENS,
    FITNESS_SURV_LIKELIHOOD,
    FITNESS_TARONEWARE_MEAN};

/*
 * Signature for a fitness function
 */
// typedef double (*FitnessFunctionPtr)(const double * const Y,
//                                      const unsigned int length,
//                                      const unsigned int numOfOutput,
//                                      const double * const outputs);

// /*
//  * Given an enum value, returns appropriate function pointer
//  */
// FitnessFunctionPtr getFitnessFunctionPtr(const FitnessFunction val);

// Evaluate the specified function
double getFitness(const FitnessFunction func,
                  const double * const Y,
                  const unsigned int length,
                  const unsigned int numOfOutput,
                  const double * const outputs);

double fitnessCIndex(const double * const Y,
                     const unsigned int length,
                     const unsigned int numOfOutput,
                     const double * const outputs);

// Anything less than 1 should be interpreted as number of outputs
int getExpectedTargetCount(const FitnessFunction func);

#endif // _GENETICFITNESS_H_
