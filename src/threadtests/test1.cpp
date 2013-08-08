//Create a group of C++11 threads from the main program
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <ctime>

static const int num_threads = 10;

//This function will be called from a thread

void call_from_thread(int tid) {
    std::cout << "Launched by thread " << tid << std::endl;
    for (double i = 1; i < 99999999; i++) {
        double x = i*i + i*i + i*i + i*i;
    }
}

int main() {
    std::thread t[num_threads];

    time_t start, end;
    time(&start);
    //Launch a group of threads
    for (int i = 0; i < num_threads; ++i) {
        t[i] = std::thread(call_from_thread, i);
    }

    std::cout << "Launched from the main\n";

    //Join the threads with the main thread
    for (int i = 0; i < num_threads; ++i) {
        t[i].join();
    }
    time(&end);

    std::cout << "Time to finish: " << difftime(end, start) << "s" << std::endl;

    return 0;
}
