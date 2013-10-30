#ifndef GLOBAL_HPP_ // header guards
#define GLOBAL_HPP_

#include "Random.hpp"
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

// All global variables should use prefix JGN_

// extern tells compiler it is delcared elsewhere (global.cpp)
// Used to lock threads accessing the population data
extern std::mutex JGN_mutexPopulation;

// Blocks until mutex is locked
void JGN_lockPopulation();

void JGN_unlockPopulation();


// Error functions might need to use a cache during runtime to speed
// things up.  ERROR_SURV_LIKELIHOOD, goes from squared to linear
// running time.
extern std::unordered_map<std::string,
                          std::vector<double> > JGN_errorCacheVectorMap;

#endif // GLOBAL_HPP_
