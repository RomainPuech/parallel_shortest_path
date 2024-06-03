#ifndef UTILS_HPP
#define UTILS_HPP

#include <limits>
#include <list>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <mutex>
#include <vector>
#include <random>
#include <thread>

template <typename T>
struct perishable_pointer;

template <typename T>
class ll_collection;

struct Edge;

struct SourceTargetReturn;

struct SourceAllReturn;

struct AllTerminalReturn;

template <typename T>
struct perishable_pointer {
  T *ptr;
  int tag;
  perishable_pointer();
  perishable_pointer(T *ptr_, int tag_);
};

template <typename T>
class ll_collection {
  protected:
    int V;
    int p;
    std::atomic<int> counter;
    std::vector<std::mutex> p_locks;
    std::vector<std::mutex> V_locks;

  public:
    std::vector<std::list<T>> data;
    std::vector<perishable_pointer<T>> perishable_pointers;

    ll_collection(int V_, int p_);
    bool replace_if_better(T element, int index, int current_tag, std::vector<int> &dist);
    bool contains(int index,int current_tag);
    void reset();
    void display();
};

struct Edge {
  int from;
  int vertex;
  int cost;
  Edge(int vertex_, int cost_) {
    vertex = vertex_;
    cost = cost_;
  }
  Edge(int from_, int vertex_, int cost_) {
    from = from_;
    vertex = vertex_;
    cost = cost_;
  }
  bool operator==(const Edge &other) const { return from == other.from && vertex == other.vertex && cost == other.cost; }
};


template <>
struct std::hash<Edge> {
  std::size_t operator()(const Edge &e) const {
    // Combine the hashes of the individual members
    std::size_t h0 = std::hash<int>()(e.from);
    std::size_t h1 = std::hash<int>()(e.vertex);
    std::size_t h2 = std::hash<int>()(e.cost);
    return h0 ^ (h1 << 1) ^ (h2 << 2);
  }
};

struct SourceTargetReturn {
  std::vector<int> path;
  std::vector<int> distance;
  SourceTargetReturn(std::vector<int> path_, std::vector<int> distance_) : path(path_), distance(distance_) {
    // Set to -1 all unreachable nodes
    for (size_t i = 0; i < distance.size(); i++) {
      if (distance[i] == INT_MAX) {
        distance[i] = -1;
      }
    }
  }
};


struct SourceAllReturn {
  std::vector<int> distances;
  SourceAllReturn(std::vector<int> distances_) : distances(distances_) {
    // Set to -1 all unreachable nodes
    for (size_t i = 0; i < distances.size(); i++) {
      if (distances[i] == INT_MAX) {
        distances[i] = -1;
      }
    }
  }
};


struct AllTerminalReturn {
  std::vector<std::vector<int>> distances;
  AllTerminalReturn(std::vector<std::vector<int>> distances_) : distances(distances_) {
    // Set to -1 all unreachable nodes
    for (size_t i = 0; i < distances.size(); i++) {
      for (size_t j = 0; j < distances[i].size(); j++) {
        if (distances[i][j] == INT_MAX) {
          distances[i][j] = -1;
        }
      }
    }
  }
};

int n_digits(int n);

void print_spaced(int x, int n);

void printDistMatrix(std::vector<std::vector<int>> distances, int V);

#endif // UTILS_HPP
