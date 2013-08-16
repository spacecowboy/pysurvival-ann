#include "global.hpp"
#include <mutex>

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
