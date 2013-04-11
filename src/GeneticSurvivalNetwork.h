#ifndef GENETICSURVIVALNETWORK_H_
#define GENETICSURVIVALNETWORK_H_

#include "GeneticNetwork.h"
#include "boost/random.hpp"

class GeneticSurvivalNetwork: public GeneticNetwork {
 public:
  // Methods
  GeneticSurvivalNetwork(const unsigned int numOfInputs,
                         const unsigned int numOfHidden);

  /*
   * Evaluates a network, including possible weight decays
   */
    virtual double evaluateNetwork(GeneticNetwork &net,
                                   const double * const X,
                                   const double * const Y,
                                   const unsigned int length,
                                   double * const outputs);

    // Used to build initial population
    virtual GeneticNetwork*
    getGeneticNetwork(GeneticNetwork &cloner,
                      boost::variate_generator<boost::mt19937&,
                      boost::normal_distribution<double> > &gaussian,
                      boost::variate_generator<boost::mt19937&,
                      boost::uniform_real<> > &uniform);

};

#endif /* GENETICSURVIVALNETWORK_H_ */
