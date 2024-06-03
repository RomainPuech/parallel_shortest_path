#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char **argv) {
    const int num_threads = argc > 1 ? std::stoi(argv[1]) : 1;
    std::chrono::duration<double, std::micro> total_time(0);

    for (int i = 0; i < num_threads; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        std::thread([](){}).detach();
        auto end = std::chrono::high_resolution_clock::now();
        total_time += end - start;
    }

    double average_time = total_time.count() / num_threads;
    std::cout << "Average time per thread: " << average_time << " microseconds" << std::endl;

    return 0;
}

