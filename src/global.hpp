#ifndef GLOBAL_HPP_ // header guards
#define GLOBAL_HPP_

#include "Random.hpp"
#include <mutex>

// GLobal random number generator
extern Random randNum;

// extern tells compiler it is delcared elsewhere (global.cpp)
// Used to lock threads accessing the population data
extern std::mutex mutexPopulation;

// Blocks until mutex is locked
void lockPopulation();

void unlockPopulation();

#endif // GLOBAL_HPP_
