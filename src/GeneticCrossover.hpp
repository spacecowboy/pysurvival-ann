#ifndef _GENETICCROSSOVER_HPP_
#define _GENETICCROSSOVER_HPP_

#include "MatrixNetwork.hpp"

enum class CrossoverMethod {CROSSOVER_UNIFORM,
    CROSSOVER_ONEPOINT,
    CROSSOVER_TWOPOINT};

/*
 * Signature for a crossover method. Sister can be null, and is
 * ignored in that case.
 */
typedef void (*crossover_func_ptr)(MatrixNetwork &mother,
                                   MatrixNetwork &father,
                                   MatrixNetwork &brother,
                                   MatrixNetwork &sister);

// Returns a specific function pointer
crossover_func_ptr getCrossoverFunctionPtr(const CrossoverMethod val);

// Runs the specific method
void evaluteCrossoverFunction(const CrossoverMethod val,
                              MatrixNetwork &mother,
                              MatrixNetwork &father,
                              MatrixNetwork &brother,
                              MatrixNetwork &sister);

void crossoverUniform(MatrixNetwork &mother,
                                   MatrixNetwork &father,
                                   MatrixNetwork &brother,
                                   MatrixNetwork &sister);

void crossoverOnepoint(MatrixNetwork &mother,
                                   MatrixNetwork &father,
                                   MatrixNetwork &brother,
                                   MatrixNetwork &sister);

void crossoverTwopoint(MatrixNetwork &mother,
                                   MatrixNetwork &father,
                                   MatrixNetwork &brother,
                                   MatrixNetwork &sister);


#endif // _GENETICCROSSOVER_HPP_
