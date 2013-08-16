#include "global.hpp"
#include "Random.hpp"
#include <mutex>

// Global random number generator
Random JGN_rand;

/*
To lock:
    std::lock_guard<std::mutex> lock(mutexPopulation);

To unlock:
    mutexPopulation.unlock();
*/
std::mutex JGN_mutexPopulation;


void JGN_lockPopulation() {
  std::lock_guard<std::mutex> lock(JGN_mutexPopulation);
}

void JGN_unlockPopulation() {
  JGN_mutexPopulation.unlock();
}
