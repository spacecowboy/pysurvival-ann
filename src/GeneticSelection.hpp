#ifndef _GENETICSELECTION_H_
#define _GENETICSELECTION_H_

#include "Random.hpp"
#include <vector>

enum class SelectionMethod {SELECTION_GEOMETRIC,
        SELECTION_ROULETTE,
        SELECTION_TOURNAMENT };

/*
 * Signature for a selection function.
 */
typedef void (*selection_func_ptr)(
    std::vector<double> &sortedFitness,
    const unsigned int max,
    unsigned int &first,
    unsigned int &second);

/*
 * Evaluates the specified function
 */
void getSelection(SelectionMethod method,
                  std::vector<double> &sortedFitness,
                  const unsigned int max,
                  unsigned int &first,
                  unsigned int &second);

void selectTournament(std::vector<double> &sortedFitness,
                      const unsigned int max,
                      unsigned int &first,
                      unsigned int &second);

void selectRoulette(std::vector<double> &sortedFitness,
                    const unsigned int max,
                    unsigned int &first,
                    unsigned int &second);

void selectGeometric(std::vector<double> &sortedFitness,
                     const unsigned int max,
                     unsigned int &first,
                     unsigned int &second);

#endif /* _GENETICSELECTION_H_ */
