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
  //std::lock_guard<std::mutex> lock(JGN_mutexPopulation);
  // Cant use that, because it is released when lock goes out of scope
  JGN_mutexPopulation.lock();
}

void JGN_unlockPopulation() {
  JGN_mutexPopulation.unlock();
}
