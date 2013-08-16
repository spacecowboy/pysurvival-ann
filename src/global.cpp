#include "global.hpp"
#include "Random.hpp"
#include <mutex>

// Global random number generator
Random randNum;

/*
To lock:
    std::lock_guard<std::mutex> lock(mutexPopulation);

To unlock:
    mutexPopulation.unlock();
*/
std::mutex mutexPopulation;


void lockPopulation() {
  std::lock_guard<std::mutex> lock(mutexPopulation);
}

void unlockPopulation() {
  mutexPopulation.unlock();
}
