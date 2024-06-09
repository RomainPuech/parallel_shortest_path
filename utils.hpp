#ifndef UTILS_HPP
#define UTILS_HPP

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

enum class Algorithm { // Enum for all the algorithms implemented, TODO: Make use of it
  DIJKSTRA,
  BELLMAN_FORD,
  FLOYD_WARSHALL,
  PARALLEL_DIJKSTRA,
  PARALLEL_BELLMAN_FORD,
  PARALLEL_FLOYD_WARSHALL,
};

template <typename T>
struct perishable_pointer {
  T *ptr;
  int tag;
  int ll_index;
  perishable_pointer() : ptr(nullptr), tag(-1), ll_index(-1) {}
  perishable_pointer(T *ptr_, int tag_, int ll_index_) : ptr(ptr_), tag(tag_), ll_index(ll_index_) {}
};

template <typename T>
class ll_collection {
  // doesn t support remove!
protected:
  size_t V;
  size_t p;
  std::atomic<int> counter; // counter for the number of elements in the collection
  // TO BE REPLACED BY C ARRAYS
  std::vector<std::mutex> p_locks; // locks for each sublist
  std::vector<std::mutex> V_locks; // locks for each pointer

public:
  std::vector<std::list<T>> data;                         // list of sublists
  std::vector<perishable_pointer<T>> perishable_pointers; // pointers to the elements. To initialize to nullptr

  ll_collection(size_t V_, size_t p_) : V(V_), p(p_) {
    data = std::vector<std::list<T>>(p);
    perishable_pointers = std::vector<perishable_pointer<T>>(V, perishable_pointer<T>()); // might be long? but initialize it only once and change tag
    p_locks = std::vector<std::mutex>(p);
    V_locks = std::vector<std::mutex>(V);
    counter = 0;
  }

  bool replace_if_better(T element, int index, int current_tag, std::vector<int> &dist) {
    int candidate_dist = dist[element.from] + element.cost;
    if ((candidate_dist < dist[element.vertex]) and ((perishable_pointers[index].tag != current_tag) || (candidate_dist < dist[(perishable_pointers[index].ptr)->from] + (perishable_pointers[index].ptr)->cost))) {
      // take a lock.
      V_locks[index].lock();
      // no need to lock distance as it is not modified in this phase
      if ((candidate_dist < dist[element.vertex]) and ((perishable_pointers[index].tag != current_tag) || (candidate_dist < dist[(perishable_pointers[index].ptr)->from] + (perishable_pointers[index].ptr)->cost))) {
        if(perishable_pointers[index].tag != current_tag) {
          // if the element is not in the collection, just add it
          int ll_index = counter++;
          p_locks[ll_index % p].lock();
          data[ll_index % p].push_back(element); // TODO: define non-blocking pushback, that returns pointer to it
          perishable_pointers[index] = perishable_pointer(&data[ll_index % p].back(), current_tag, ll_index % p);
          p_locks[ll_index % p].unlock();
        } else {
          // if the element is in the collection, update it
          p_locks[perishable_pointers[index].ll_index].lock();
          (perishable_pointers[index].ptr)->from = element.from; // notice: only defined for T = Edge
          (perishable_pointers[index].ptr)->cost = element.cost;
          p_locks[perishable_pointers[index].ll_index].unlock();
        }
        V_locks[index].unlock();
        return true;
      }
    }
    V_locks[index].unlock();
    return false;
  }

  bool contains(int index, int current_tag) {
    return perishable_pointers[index].tag == current_tag;
  }

  void reset() {
    // delete all content of linkedlists
    for (size_t i = 0; i < p; i++) {
      data[i].clear();
    }
  }
  void display() {
    for (size_t i = 0; i < p; i++) {
      std::cout << "List " << i << ": ";
      for (T element : data[i]) {
        std::cout << "(" << element.from << "," << element.vertex << "," << element.cost << "),  ";
      }
      std::cout << "\n";
    }
  }
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

// Define a hash function
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
