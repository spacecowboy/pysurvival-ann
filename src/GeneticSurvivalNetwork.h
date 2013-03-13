#ifndef GENETICSURVIVALNETWORK_H_
#define GENETICSURVIVALNETWORK_H_

#include "GeneticNetwork.h"
#include "boost/random.hpp"

class GeneticSurvivalNetwork: public GeneticNetwork {
 public:
  // Methods
  GeneticSurvivalNetwork(unsigned int numOfInputs, unsigned int numOfHidden);

  /*
   * Evaluates a network, including possible weight decays
   */
  virtual double evaluateNetwork(GeneticNetwork *net, double *X,
                                 double *Y, unsigned int length,
                                 double *outputs);

    // Used to build initial population
  virtual GeneticNetwork*
    getGeneticNetwork(GeneticNetwork *cloner,
                      boost::variate_generator<boost::mt19937&,
                      boost::normal_distribution<double> >* gaussian,
                      boost::variate_generator<boost::mt19937&,
                      boost::uniform_int<> > *uniform);

};

#endif /* GENETICSURVIVALNETWORK_H_ */
