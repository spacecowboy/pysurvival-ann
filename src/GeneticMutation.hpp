#ifndef _GENETICMUTATION_HPP_
#define _GENETICMUTATION_HPP_

#include "MatrixNetwork.hpp"
#include "Random.hpp"

class GeneticMutator {
protected:
  Random &rand;

public:
  GeneticMutator(Random &rand);
  virtual ~GeneticMutator();

// Randomizes all parts with chance 1.0
  void randomizeNetwork(MatrixNetwork &net,
                        const double weightStdDev);

  void mutateWeights(MatrixNetwork &net,
                     const double chance,
                     const double stddev);

  void mutateConns(MatrixNetwork &net,
                   const double chance);

  void mutateActFuncs(MatrixNetwork &net,
                      const double chance);

};

#endif // _GENETICMUTATION_HPP_
