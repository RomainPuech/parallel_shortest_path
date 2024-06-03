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


enum class Algorithm {  // Enum for all the algorithms implemented, TODO: Make use of it
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
  perishable_pointer() : ptr(nullptr), tag(-1) {}
  perishable_pointer(T *ptr_, int tag_) : ptr(ptr_), tag(tag_) {}
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

        ll_collection(int V_, int p_): V(V_), p(p_) {
            data = std::vector<std::list<T>>(p);
            perishable_pointers = std::vector<perishable_pointer<T>>(V, perishable_pointer<T>()); // might be long? but initialize it only once and change tag
            p_locks = std::vector<std::mutex>(p);
            V_locks = std::vector<std::mutex>(V); 
            counter = 0;
        }

        bool replace_if_better(T element, int index, int current_tag, std::vector<int> &dist) {
            if ( (perishable_pointers[index].tag != current_tag) || (  (perishable_pointers[index].ptr)->cost + dist[(perishable_pointers[index].ptr)->from] < dist[(perishable_pointers[index].ptr)->vertex]    ) ){
                // take a lock.
                // no need to lock distance as it is not modified in this phase
                V_locks[index].lock();
                if ( (perishable_pointers[index].tag != current_tag) || (  (perishable_pointers[index].ptr)->cost + dist[(perishable_pointers[index].ptr)->from] < dist[(perishable_pointers[index].ptr)->vertex]    ) ){
                    int ll_index = counter++;
                    p_locks[ll_index % p].lock();
                    data[ll_index % p].push_back(element); // TODO: define non-blocking pushback, that returns pointer to it
                    perishable_pointers[index] = perishable_pointer(&data[ll_index % p].back(), current_tag);
                    p_locks[ll_index % p].unlock();
                    V_locks[index].unlock();
                    return true;
                }
            }
            return false;
        }

        bool contains(int index,int current_tag) { return perishable_pointers[index].tag == current_tag; }

        void reset() {  // delete all content of linkedlists
            for (int i = 0; i < p; i++){
                data[i].clear();
            }
        }

        void display() {
            for(int i=0;i<p;i++){
                std::cout << "List " << i << ": ";
                for (T element : data[i]){
                    std::cout << "(" << element.from << ","<< element.vertex << "," << element.cost << "),  ";
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