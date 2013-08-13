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

 public:

  Random();
  virtual ~Random();

  /*
   * All functions range from 0 to 1, except geometric
   * which returns 0 to max (exclusive)
   */
  int geometric(const int max);
  double uniform();
  double normal();

};
