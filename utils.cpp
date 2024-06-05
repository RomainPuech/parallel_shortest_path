/*
File for utils

Incldues:

---- DATA STRUCTURES ----
struct perishable_pointer

class ll_collection


---- FUCNTIONS ----

*/

#include <iostream>
#include <limits>
#include <list>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.hpp"

int n_digits(int n) {
  if (n == 0) {
    return 1;
  } else if (n == -1) {
    return 2;
  }
  int digits = 0;
  while (n) {
    n /= 10;
    digits++;
  }
  return digits;
}

void print_spaced(int x, int n) {
  int digits = n_digits(x);
  std::cout << x;
  for (int i = 0; i < n - digits; i++) {
    std::cout << " ";
  }
}

void printDistMatrix(std::vector<std::vector<int>> distances, int V) {
  std::vector<int> max_dist(V + 1, -1);
  for (int X = 0; X < V; X++) {
    max_dist[X] = std::max(max_dist[X], X);
    for (int Y = 0; Y < V; Y++) {
      max_dist[Y + 1] = std::max(max_dist[Y + 1], distances[X][Y]);
      if (distances[X][Y] == -1) {
        max_dist[Y + 1] = std::max(max_dist[Y + 1], 10);
      }
    }
  }
  max_dist[0] = V - 1;
  for (int X = 0; X < V + 1; X++) {
    max_dist[X] = n_digits(max_dist[X]) + 1;
  }
  // max_dist[i] is now the maximum number of digits in the i-th column (0 is index)

  std::cout << "Distances: \n";
  for (int X = 0; X < V + 1; X++) {
    if (X == 0) {
      std::cout << "TO ";
      for (int i = 0; i < max_dist[0] + 5 - 3; ++i) {
        std::cout << " ";
      }
    } else {
      std::cout << "FROM ";
      print_spaced(X - 1, max_dist[0]);
    }
    std::cout << " ";

    for (int Y = 1; Y < V + 1; Y++) {
      if (X == 0) {
        print_spaced(Y - 1, max_dist[Y]);
      } else {
        print_spaced(distances[X - 1][Y - 1], max_dist[Y]);
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n\n"
            << std::flush;
}
