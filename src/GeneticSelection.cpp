#include "GeneticSelection.hpp"
#include "global.hpp"
#include <vector>

using namespace std;

void getSelection(SelectionMethod method,
                  vector<double> &sortedFitness,
                  const unsigned int max,
                  unsigned int &first,
                  unsigned int &second) {
}

// This should be private, or made thread safe.
// Remember to not lock twice, see other selectTournament
void selectTournament(vector<double> &sortedFitness,
                      const unsigned int max,
                      unsigned int &first) {
  unsigned int i, j;

  i = JGN_rand.uniformNumber(0, max);
  j = i;
  while (j == i) {
    j = JGN_rand.uniformNumber(0, max);
  }

  // sorted list, so first wins
  if (i < j) {
    first = i;
  }
  else {
    first = j;
  }
}
void selectTournament(vector<double> &sortedFitness,
                      const unsigned int max,
                      unsigned int &first,
                      unsigned int &second) {
  selectTournament(sortedFitness, max, first);
  second = first;
  while (first == second) {
    selectTournament(sortedFitness, max, second);
  }
}

void selectRoulette(vector<double> &sortedFitness,
                    const unsigned int max,
                    unsigned int &first,
                    unsigned int &second) {
  first = JGN_rand.weightedNumber(&sortedFitness[0],
                                0, max);
  second = first;
  while (second == first) {
    second = JGN_rand.weightedNumber(&sortedFitness[0],
                                   0, max);
  }
}

void selectGeometric(vector<double> &sortedFitness,
                     const unsigned int max,
                     unsigned int &first,
                     unsigned int &second) {
  first = JGN_rand.geometric(max);
  second = first;
  while (first == second) {
    second = JGN_rand.geometric(max);
  }
}
