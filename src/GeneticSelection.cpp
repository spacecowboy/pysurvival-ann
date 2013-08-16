#include "GeneticSelection.hpp"
#include "Random.hpp"
#include "global.hpp"
#include <mutex>
#include <vector>

using namespace std;

void getSelection(SelectionMethod method,
                  Random &random,
                  vector<double> &sortedFitness,
                  const unsigned int max,
                  unsigned int &first,
                  unsigned int &second) {
}

// This should be private, or made thread safe.
// Remember to not lock twice, see other selectTournament
void selectTournament(Random &random,
                      vector<double> &sortedFitness,
                      const unsigned int max,
                      unsigned int &first) {
  unsigned int i, j;

  i = random.uniformNumber(0, max);
  j = i;
  while (j == i) {
    j = random.uniformNumber(0, max);
  }

  // sorted list, so first wins
  if (i < j) {
    first = i;
  }
  else {
    first = j;
  }
}
void selectTournament(Random &random,
                      vector<double> &sortedFitness,
                      const unsigned int max,
                      unsigned int &first,
                      unsigned int &second) {
  lockPopulation();
  selectTournament(random, sortedFitness, max, first);
  second = first;
  while (first == second) {
    selectTournament(random, sortedFitness, max, second);
  }
  unlockPopulation();
}

void selectRoulette(Random &random,
                    vector<double> &sortedFitness,
                    const unsigned int max,
                    unsigned int &first,
                    unsigned int &second) {
  lockPopulation();
  first = random.weightedNumber(&sortedFitness[0],
                                0, max);
  second = first;
  while (second == first) {
    second = random.weightedNumber(&sortedFitness[0],
                                   0, max);
  }
  unlockPopulation();
}

void selectGeometric(Random &random,
                     vector<double> &sortedFitness,
                     const unsigned int max,
                     unsigned int &first,
                     unsigned int &second) {
  first = random.geometric(max);
  second = first;
  while (first == second) {
    second = random.geometric(max);
  }
}
