#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include "boost/random.hpp"
#include <time.h>
//#include <ctime>

class Random {

protected:
  boost::mt19937 *eng; // a core engine class

  // Normal distribution for weight mutation, 0 mean and 1 stddev
  // We can then get any normal distribution with y = mean + stddev * x
  boost::normal_distribution<double> *gauss_dist;
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >
    *gaussian_num;
  // Uniform distribution 0 to 1 (inclusive)
  boost::uniform_real<double> *uni_dist;
  boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >
  *uniform_num;
  // Geometric
  boost::geometric_distribution<int, double> *geo_dist;
  boost::variate_generator<boost::mt19937&,
                           boost::geometric_distribution<int, double> >
  *geometric_num;

  void init(const unsigned int seed);

 public:

  Random();
  Random(const unsigned int seed);
  virtual ~Random();

  // 0 - Max (exclusive)
  int geometric(const int max);
  // 0 - 1
  double uniform();
  // mean 0, stddev 1
  double normal();

  // 0 to UINT_MAX
  unsigned int uint();

  // min - max (exclusive)
  unsigned int uniformNumber(const unsigned int min,
                             const unsigned int max);

  // min - max (exclusive)
  // max must be at most the same as the length of weights
  unsigned int weightedNumber(const double * const weights,
                              const unsigned int min,
                              const unsigned int max);

};


#endif // RANDOM_HPP_
