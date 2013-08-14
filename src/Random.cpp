#include "Random.h"
#include "boost/random.hpp"
#include <time.h>
#include <algorithm> // std::fill

Random::Random()
{
  eng = new boost::mt19937(); // a core engine class
  eng->seed(time(NULL));

  // Normal distribution for weight mutation, 0 mean and 1 stddev
  // We can then get any normal distribution with y = mean + stddev * x
  gauss_dist = new boost::normal_distribution<double>(0, 1);
  gaussian_num = new boost::variate_generator<boost::mt19937&,
                                          boost::normal_distribution<double> >
    (*eng, *gauss_dist);

  // Geometric distribution for selecting parents
  geo_dist = new boost::geometric_distribution<int, double>(0.95);
  geometric_num = new boost::variate_generator<boost::mt19937&,
boost::geometric_distribution<int, double> >
    (*eng, *geo_dist);

  // Uniform distribution 0 to 1 (inclusive)
  uni_dist = new boost::uniform_real<double>(0.0, 1.0);
  uniform_num = new boost::variate_generator<boost::mt19937&,
                                         boost::uniform_real<double> >
    (*eng, *uni_dist);
}

Random::~Random() {
  delete eng;
  delete gauss_dist;
  delete gaussian_num;
  delete geo_dist;
  delete geometric_num;
  delete uni_dist;
  delete uniform_num;
}


int Random::geometric(int max) {
  if (max <= 0) {
    throw 666;
  }

  int val = (*geometric_num)() - 1;
  while (val > max) {
    val = (*geometric_num)() -1;
  }
  return val;
}

double Random::uniform() {
  return (*uniform_num)();
}

double Random::normal() {
  return (*gaussian_num)();
}

unsigned int Random::uniformNumber(const unsigned int min,
                                   const unsigned int max) {
  double weights[max];
  std::fill(weights, weights + max, 1);

  return weightedNumber(weights, min, max);
}

unsigned int Random::weightedNumber(const double * const weights,
                                    const unsigned int min,
                                    const unsigned int max) {
  double sum = 0, inc = 0, roll;
  unsigned int i, result = min;
  for (i = min; i < max; i++) {
    sum += weights[i];
  }

  roll = uniform() * sum;
  for (i = min; i < max; i++) {
    result = i;
    inc += weights[i];

    if (inc >= roll) {
      // Done
      break;
    }
  }

  return result;
}
