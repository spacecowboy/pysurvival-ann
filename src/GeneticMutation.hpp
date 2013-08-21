#ifndef _GENETICMUTATION_HPP_
#define _GENETICMUTATION_HPP_

#include "MatrixNetwork.hpp"

// Randomizes all parts with chance 1.0
void randomize(MatrixNetwork &net,
               const double weightStdDev);

void mutateWeights(MatrixNetwork &net,
                   const double chance,
                   const double stddev);

void mutateConns(MatrixNetwork &net,
                 const double chance);

void mutateActFuncs(MatrixNetwork &net,
                    const double chance);

#endif // _GENETICMUTATION_HPP_
