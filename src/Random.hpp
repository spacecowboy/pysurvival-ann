#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <random>

class Random {

protected:
  std::mt19937_64 *eng; // a core engine class

  // Normal distribution for weight mutation, 0 mean and 1 stddev
  // We can then get any normal distribution with y = mean + stddev * x
  std::normal_distribution<double> *distNormal;
  // Uniform distribution 0 to 1 (inclusive)
  std::uniform_real_distribution<double> *distUniform;
  // Uniform int
  std::uniform_int_distribution<unsigned int> *distUniformInt;
  // Geometric
  std::geometric_distribution<int> *distGeometric;

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
  // 0 or 1
  unsigned int randBit();

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
