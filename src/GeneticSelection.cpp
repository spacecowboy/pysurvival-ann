#include "GeneticSelection.hpp"
#include "global.hpp"
#include <vector>

using namespace std;

GeneticSelector::GeneticSelector(Random &rand) :
  rand(rand)
{
}

GeneticSelector::~GeneticSelector()
{}

void GeneticSelector::getSelection(SelectionMethod method,
                                   vector<double> &sortedFitness,
                                   const unsigned int max,
                                   unsigned int *first,
                                   unsigned int *second) {
  switch (method) {
  case SelectionMethod::SELECTION_ROULETTE:
    selectRoulette(sortedFitness, max, first, second);
    break;
  case SelectionMethod::SELECTION_GEOMETRIC:
    selectGeometric(sortedFitness, max, first, second);
    break;
  case SelectionMethod::SELECTION_TOURNAMENT:
  default:
    selectTournament(sortedFitness, max, first, second);
    break;
  }
}

// This should be private, or made thread safe.
// Remember to not lock twice, see other selectTournament
void GeneticSelector::selectTournament(vector<double> &sortedFitness,
                                       const unsigned int max,
                                       unsigned int *first) {
  unsigned int i, j;

  i = this->rand.uniformNumber(0, max);
  j = i;
  while (j == i) {
    j = this->rand.uniformNumber(0, max);
  }

  // sorted list, so first wins
  if (i < j) {
    *first = i;
  }
  else {
    *first = j;
  }
}
void GeneticSelector::selectTournament(vector<double> &sortedFitness,
                      const unsigned int max,
                      unsigned int *first,
                      unsigned int *second) {
  selectTournament(sortedFitness, max, first);
  *second = *first;
  while (*first == *second) {
    selectTournament(sortedFitness, max, second);
  }
}

void GeneticSelector::selectRoulette(vector<double> &sortedFitness,
                    const unsigned int max,
                    unsigned int *first,
                    unsigned int *second) {
  *first = this->rand.weightedNumber(&sortedFitness[0],
                                   0, max);
  *second = *first;
  while (second == first) {
    *second = this->rand.weightedNumber(&sortedFitness[0],
                                      0, max);
  }
}

void GeneticSelector::selectGeometric(vector<double> &sortedFitness,
                     const unsigned int max,
                     unsigned int *first,
                     unsigned int *second) {
  *first = this->rand.geometric(max);
  *second = *first;
  while (*first == *second) {
    *second = this->rand.geometric(max);
  }
}
