#include "Random.hpp"
#include <random>
#include <algorithm> // std::fill
#include <stdio.h>
#include <limits> // max int

Random::Random()
{
  // Use true random seed
  std::random_device rd;
  init(rd());
}

Random::Random(const unsigned int seed)
{
  init(seed);
}

void Random::init(const unsigned int seed) {
  eng = new std::mt19937_64(seed); // a core engine class

  // Normal distribution for weight mutation, 0 mean and 1 stddev
  // We can then get any normal distribution with y = mean + stddev * x
  distNormal = new std::normal_distribution<double>(0, 1);

  // Geometric distribution for selecting parents
  distGeometric = new std::geometric_distribution<int>(0.95);

  // Uniform distribution 0 to 1 (inclusive)
  distUniform = new std::uniform_real_distribution<double>(0.0, 1.0);
}

Random::~Random() {
  delete eng;
  delete distNormal;
  delete distGeometric;
  delete distUniform;
}


int Random::geometric(int max) {
  if (max <= 0) {
    throw 666;
  }

  int val = (*distGeometric)(*eng);
  while (val > max) {
    val = (*distGeometric)(*eng);
  }
  return val;
}

double Random::uniform() {
  return (*distUniform)(*eng);
}

double Random::normal() {
  return (*distNormal)(*eng);
}

unsigned int Random::uint() {
  return uniformNumber(0, std::numeric_limits<unsigned int>::max());
}

unsigned int Random::uniformNumber(const unsigned int min,
                                   const unsigned int max) {
  std::uniform_int_distribution<unsigned int>
    int_dist(min, max - 1);
  return int_dist(*eng);
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
