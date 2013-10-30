#include "global.hpp"
#include <mutex>
#include <unordered_map>

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

// For caching during evaluation
std::unordered_map<std::string,
                   std::vector<double> > JGN_errorCacheVectorMap;
