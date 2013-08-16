#ifndef GLOBAL_HPP_ // header guards
#define GLOBAL_HPP_

#include "Random.hpp"
#include <mutex>

// All global variables should use prefix JGN_

// GLobal random number generator
extern Random JGN_rand;

// extern tells compiler it is delcared elsewhere (global.cpp)
// Used to lock threads accessing the population data
extern std::mutex JGN_mutexPopulation;

// Blocks until mutex is locked
void JGN_lockPopulation();

void JGN_unlockPopulation();

#endif // GLOBAL_HPP_
