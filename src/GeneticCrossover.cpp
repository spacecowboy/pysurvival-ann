#include "GeneticCrossover.hpp"
#include "MatrixNetwork.hpp"
#include "global.hpp"
#include <algorithm> // std::copy

// Returns a specific function pointer
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

// Runs the specific method
void evaluateCrossoverFunction(const CrossoverMethod val,
                               MatrixNetwork &mother,
                               MatrixNetwork &father,
                               MatrixNetwork &brother,
                               MatrixNetwork &sister) {
  crossover_func_ptr func = getCrossoverFunctionPtr(val);
  (*func)(mother, father, brother, sister);
}

// Guarantees that low < high, as long as limit is > 1
void getTwoUniform(unsigned int min,
                   unsigned int max,
                   unsigned int *low,
                   unsigned int *high) {
  unsigned int point1, point2;
  point1 = JGN_rand.uniformNumber(min, max);
  point2 = point1;
  while (point2 == point1) {
    point2 = JGN_rand.uniformNumber(min, max);
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

void crossoverUniform(MatrixNetwork &mother,
                      MatrixNetwork &father,
                      MatrixNetwork &brother,
                      MatrixNetwork &sister) {
  // Each weight is assigned randomly
  unsigned int i;
  for (i = 0; i < mother.LENGTH * mother.LENGTH; i++) {
    // activation functions are only length long
    if (i < mother.LENGTH) {
      if (JGN_rand.uniform() < 0.5) {
        brother.actFuncs[i] = mother.actFuncs[i];
        sister.actFuncs[i] = father.actFuncs[i];
      }
      else {
        sister.actFuncs[i] = mother.actFuncs[i];
        brother.actFuncs[i] = father.actFuncs[i];
      }
    }
    // Now connections
    if (JGN_rand.uniform() < 0.5) {
      brother.conns[i] = mother.conns[i];
      sister.conns[i] = father.conns[i];
    }
    else {
      sister.conns[i] = mother.conns[i];
      brother.conns[i] = father.conns[i];
    }
    // And weights
    if (JGN_rand.uniform() < 0.5) {
      brother.weights[i] = mother.weights[i];
      sister.weights[i] = father.weights[i];
    }
    else {
      sister.weights[i] = mother.weights[i];
      brother.weights[i] = father.weights[i];
    }
  }
}

void crossoverOnepoint(MatrixNetwork &mother,
                       MatrixNetwork &father,
                       MatrixNetwork &brother,
                       MatrixNetwork &sister) {
  unsigned int point, limit;
  // #### Activation Functions ####
  // activation functions are only length long
  limit = mother.LENGTH;
  point = JGN_rand.uniformNumber(0, limit);
  // First brother
  std::copy(mother.actFuncs, mother.actFuncs + point,
            brother.actFuncs);
  std::copy(father.actFuncs + point, father.actFuncs + limit,
            brother.actFuncs + point);
  // Then sister
  std::copy(father.actFuncs, father.actFuncs + point,
            sister.actFuncs);
  std::copy(mother.actFuncs + point, mother.actFuncs + limit,
            sister.actFuncs + point);


  // #### Connections ####
  limit = mother.LENGTH * mother.LENGTH;
  point = JGN_rand.uniformNumber(0, limit);
  // First brother
  std::copy(mother.conns, mother.conns + point,
            brother.conns);
  std::copy(father.conns + point, father.conns + limit,
            brother.conns + point);
  // Then sister
  std::copy(father.conns, father.conns + point,
            sister.conns);
  std::copy(mother.conns + point, mother.conns + limit,
            sister.conns + point);


  // #### Weights ####
  limit = mother.LENGTH * mother.LENGTH;
  point = JGN_rand.uniformNumber(0, limit);
  // First brother
  std::copy(mother.weights, mother.weights + point,
            brother.weights);
  std::copy(father.weights + point, father.weights + limit,
            brother.weights + point);
  // Then sister
  std::copy(father.weights, father.weights + point,
            sister.weights);
  std::copy(mother.weights + point, mother.weights + limit,
            sister.weights + point);

}

void crossoverTwopoint(MatrixNetwork &mother,
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
  std::copy(mother.actFuncs, mother.actFuncs + pointLow,
            brother.actFuncs);
  std::copy(father.actFuncs + pointLow, father.actFuncs + pointHigh,
            brother.actFuncs + pointLow);
  std::copy(mother.actFuncs + pointHigh, mother.actFuncs + limit,
            brother.actFuncs + pointHigh);

  // Then sister
  std::copy(father.actFuncs, father.actFuncs + pointLow,
            sister.actFuncs);
  std::copy(mother.actFuncs + pointLow, mother.actFuncs + pointHigh,
            sister.actFuncs + pointLow);
  std::copy(father.actFuncs + pointHigh, father.actFuncs + limit,
            sister.actFuncs + pointHigh);


  // #### Connections ####
  limit = mother.LENGTH * mother.LENGTH;

  // Make sure points are not at edges
  getTwoUniform(1, limit - 1,
                &pointLow,
                &pointHigh);

  // First brother
  std::copy(mother.conns, mother.conns + pointLow,
            brother.conns);
  std::copy(father.conns + pointLow, father.conns + pointHigh,
            brother.conns + pointLow);
  std::copy(mother.conns + pointHigh, mother.conns + limit,
            brother.conns + pointHigh);

  // Then sister
  std::copy(father.conns, father.conns + pointLow,
            sister.conns);
  std::copy(mother.conns + pointLow, mother.conns + pointHigh,
            sister.conns + pointLow);
  std::copy(father.conns + pointHigh, father.conns + limit,
            sister.conns + pointHigh);


  // #### Weights ####
  limit = mother.LENGTH * mother.LENGTH;

  // Make sure points are not at edges
  getTwoUniform(1, limit - 1,
                &pointLow,
                &pointHigh);

  // First brother
  std::copy(mother.weights, mother.weights + pointLow,
            brother.weights);
  std::copy(father.weights + pointLow, father.weights + pointHigh,
            brother.weights + pointLow);
  std::copy(mother.weights + pointHigh, mother.weights + limit,
            brother.weights + pointHigh);

  // Then sister
  std::copy(father.weights, father.weights + pointLow,
            sister.weights);
  std::copy(mother.weights + pointLow, mother.weights + pointHigh,
            sister.weights + pointLow);
  std::copy(father.weights + pointHigh, father.weights + limit,
            sister.weights + pointHigh);

}
