#include "GeneticCrossover.hpp"
#include "MatrixNetwork.hpp"
#include <algorithm> // std::copy

// Returns a specific function pointer
/*
crossover_func_ptr getCrossoverFunctionPtr(const CrossoverMethod val) {
  switch (val) {
  case CrossoverMethod::CROSSOVER_UNIFORM:
    return *crossoverUniform;
  case CrossoverMethod::CROSSOVER_TWOPOINT:
    return *crossoverTwopoint;
  case CrossoverMethod::CROSSOVER_ONEPOINT:
  default:
    return *crossoverOnepoint;
  }
}
*/

GeneticCrosser::GeneticCrosser(Random &rand) :
  rand(rand)
{}

GeneticCrosser::~GeneticCrosser()
{}

// Runs the specific method
void GeneticCrosser::evaluateCrossoverFunction(const CrossoverMethod val,
                                               MatrixNetwork &mother,
                                               MatrixNetwork &father,
                                               MatrixNetwork &brother,
                                               MatrixNetwork &sister) {
  //crossover_func_ptr func = getCrossoverFunctionPtr(val);
  //(*func)(mother, father, brother, sister);

  switch (val) {
  case CrossoverMethod::CROSSOVER_UNIFORM:
    return crossoverUniform(mother, father, brother, sister);
  case CrossoverMethod::CROSSOVER_TWOPOINT:
    return crossoverTwopoint(mother, father, brother, sister);
  case CrossoverMethod::CROSSOVER_ONEPOINT:
  default:
    return crossoverOnepoint(mother, father, brother, sister);
  }

}

// Guarantees that low < high, as long as limit is > 1
void GeneticCrosser::getTwoUniform(unsigned int min,
                                   unsigned int max,
                                   unsigned int *low,
                                   unsigned int *high) {
  unsigned int point1, point2;
  point1 = this->rand.uniformNumber(min, max);
  point2 = point1;
  while (point2 == point1) {
    point2 = this->rand.uniformNumber(min, max);
  }

  if (point1 < point2) {
    *low = point1;
    *high = point2;
  }
  else {
    *low = point2;
    *high = point1;
  }
}

void GeneticCrosser::crossoverUniform(MatrixNetwork &mother,
                                      MatrixNetwork &father,
                                      MatrixNetwork &brother,
                                      MatrixNetwork &sister) {
  // Each weight is assigned randomly
  unsigned int i;
  for (i = 0; i < mother.LENGTH * mother.LENGTH; i++) {
    // activation functions are only length long
    if (i < mother.LENGTH) {
      if (this->rand.uniform() < 0.5) {
        brother.actFuncs.at(i) = mother.actFuncs.at(i);
        sister.actFuncs.at(i) = father.actFuncs.at(i);
      }
      else {
        sister.actFuncs.at(i) = mother.actFuncs.at(i);
        brother.actFuncs.at(i) = father.actFuncs.at(i);
      }
    }
    // Now connections
    if (this->rand.uniform() < 0.5) {
      brother.conns.at(i) = mother.conns.at(i);
      sister.conns.at(i) = father.conns.at(i);
    }
    else {
      sister.conns.at(i) = mother.conns.at(i);
      brother.conns.at(i) = father.conns.at(i);
    }
    // And weights
    if (this->rand.uniform() < 0.5) {
      brother.weights.at(i) = mother.weights.at(i);
      sister.weights.at(i) = father.weights.at(i);
    }
    else {
      sister.weights.at(i) = mother.weights.at(i);
      brother.weights.at(i) = father.weights.at(i);
    }
  }
}

void GeneticCrosser::crossoverOnepoint(MatrixNetwork &mother,
                                       MatrixNetwork &father,
                                       MatrixNetwork &brother,
                                       MatrixNetwork &sister) {
  unsigned int point, limit;
  // #### Activation Functions ####
  // activation functions are only length long
  limit = mother.LENGTH;
  point = this->rand.uniformNumber(0, limit);
  // First brother
  std::copy(mother.actFuncs.begin(), mother.actFuncs.begin() + point,
            brother.actFuncs.begin());
  std::copy(father.actFuncs.begin() + point, father.actFuncs.end(),
            brother.actFuncs.begin() + point);
  // Then sister
  std::copy(father.actFuncs.begin(), father.actFuncs.begin() + point,
            sister.actFuncs.begin());
  std::copy(mother.actFuncs.begin() + point, mother.actFuncs.end(),
            sister.actFuncs.begin() + point);


  // #### Connections ####
  limit = mother.LENGTH * mother.LENGTH;
  point = this->rand.uniformNumber(0, limit);
  // First brother
  std::copy(mother.conns.begin(), mother.conns.begin() + point,
            brother.conns.begin());
  std::copy(father.conns.begin() + point, father.conns.end(),
            brother.conns.begin() + point);
  // Then sister
  std::copy(father.conns.begin(), father.conns.begin() + point,
            sister.conns.begin());
  std::copy(mother.conns.begin() + point, mother.conns.end(),
            sister.conns.begin() + point);


  // #### Weights ####
  limit = mother.LENGTH * mother.LENGTH;
  point = this->rand.uniformNumber(0, limit);
  // First brother
  std::copy(mother.weights.begin(), mother.weights.begin() + point,
            brother.weights.begin());
  std::copy(father.weights.begin() + point, father.weights.end(),
            brother.weights.begin() + point);
  // Then sister
  std::copy(father.weights.begin(), father.weights.begin() + point,
            sister.weights.begin());
  std::copy(mother.weights.begin() + point, mother.weights.end(),
            sister.weights.begin() + point);

}

void GeneticCrosser::crossoverTwopoint(MatrixNetwork &mother,
                                       MatrixNetwork &father,
                                       MatrixNetwork &brother,
                                       MatrixNetwork &sister) {
  unsigned int pointLow, pointHigh, limit;
  // #### Activation Functions ####
  // activation functions are only length long
  limit = mother.LENGTH;

  // Make sure points are not at edges
  getTwoUniform(1, limit - 1,
                &pointLow,
                &pointHigh);

  // First brother
  std::copy(mother.actFuncs.begin(), mother.actFuncs.begin() + pointLow,
            brother.actFuncs.begin());
  std::copy(father.actFuncs.begin() + pointLow, father.actFuncs.begin() + pointHigh,
            brother.actFuncs.begin() + pointLow);
  std::copy(mother.actFuncs.begin() + pointHigh, mother.actFuncs.end(),
            brother.actFuncs.begin() + pointHigh);

  // Then sister
  std::copy(father.actFuncs.begin(), father.actFuncs.begin() + pointLow,
            sister.actFuncs.begin());
  std::copy(mother.actFuncs.begin() + pointLow, mother.actFuncs.begin() + pointHigh,
            sister.actFuncs.begin() + pointLow);
  std::copy(father.actFuncs.begin() + pointHigh, father.actFuncs.end(),
            sister.actFuncs.begin() + pointHigh);


  // #### Connections ####
  limit = mother.LENGTH * mother.LENGTH;

  // Make sure points are not at edges
  getTwoUniform(1, limit - 1,
                &pointLow,
                &pointHigh);

  // First brother
  std::copy(mother.conns.begin(), mother.conns.begin() + pointLow,
            brother.conns.begin());
  std::copy(father.conns.begin() + pointLow, father.conns.begin() + pointHigh,
            brother.conns.begin() + pointLow);
  std::copy(mother.conns.begin() + pointHigh, mother.conns.end(),
            brother.conns.begin() + pointHigh);

  // Then sister
  std::copy(father.conns.begin(), father.conns.begin() + pointLow,
            sister.conns.begin());
  std::copy(mother.conns.begin() + pointLow, mother.conns.begin() + pointHigh,
            sister.conns.begin() + pointLow);
  std::copy(father.conns.begin() + pointHigh, father.conns.end(),
            sister.conns.begin() + pointHigh);


  // #### Weights ####
  limit = mother.LENGTH * mother.LENGTH;

  // Make sure points are not at edges
  getTwoUniform(1, limit - 1,
                &pointLow,
                &pointHigh);

  // First brother
  std::copy(mother.weights.begin(), mother.weights.begin() + pointLow,
            brother.weights.begin());
  std::copy(father.weights.begin() + pointLow, father.weights.begin() + pointHigh,
            brother.weights.begin() + pointLow);
  std::copy(mother.weights.begin() + pointHigh, mother.weights.end(),
            brother.weights.begin() + pointHigh);

  // Then sister
  std::copy(father.weights.begin(), father.weights.begin() + pointLow,
            sister.weights.begin());
  std::copy(mother.weights.begin() + pointLow, mother.weights.begin() + pointHigh,
            sister.weights.begin() + pointLow);
  std::copy(father.weights.begin() + pointHigh, father.weights.end(),
            sister.weights.begin() + pointHigh);

}
