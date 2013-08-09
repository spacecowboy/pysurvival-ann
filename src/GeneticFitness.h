#ifndef _GENETICFITNESS_H_
#define _GENETICFITNESS_H_

class GeneticNetwork;

/*
 * A fitness function returns a number where higher is better
 */
enum fitness_function_t { FITNESS_CINDEX,
                          FITNESS_MSE_CENS };

/*
 * Signature for a fitness function
 */
typedef double (*fitness_func_ptr)(GeneticNetwork &net,
                                 const double * const X,
                                 const double * const Y,
                                 const unsigned int length,
                                 double * const outputs);

/*
 * Given an enum value, returns appropriate function pointer
 */
fitness_func_ptr getFitnessFunctionPtr(const long val);

double fitness_cindex(GeneticNetwork &net,
                      const double * const X,
                      const double * const Y,
                      const unsigned int length,
                      double * const outputs);

double fitness_mse_cens(GeneticNetwork &net,
                        const double * const X,
                        const double * const Y,
                        const unsigned int length,
                        double * const outputs);

#endif // _GENETICFITNESS_H_
