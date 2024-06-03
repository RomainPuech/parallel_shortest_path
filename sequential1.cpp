#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std::chrono;

#include <barrier>
#include <mutex>
#include <random>
#include <thread>


#pragma GCC diagnostic ignored "-Wvla"

// Custom data structures
// template <typename T>
// struct perishable_pointer {
//   T *ptr;
//   int tag;
//   perishable_pointer() : ptr(nullptr), tag(-1) {}
//   perishable_pointer(T *ptr_, int tag_) : ptr(ptr_), tag(tag_) {}
// };

// template <typename T>
// class ll_collection{
//   // doesn t support remove!
//   protected:
//     int V;
//     int p;
//     std::atomic<int> counter; // counter for the number of elements in the collection
//     // TO BE REPLACED BY C ARRAYS
//     std::vector<std::mutex> p_locks; // locks for each sublist
//     std::vector<std::mutex> V_locks; // locks for each pointer

//   public:
//     std::vector<std::list<T>> data; // list of sublists
//     std::vector<perishable_pointer<T>> perishable_pointers; // pointers to the elements. To initialize to nullptr

//     ll_collection(int V_, int p_): V(V_), p(p_){
//       data = std::vector<std::list<T>>(p);
//       perishable_pointers = std::vector<perishable_pointer<T>>(V, perishable_pointer<T>()); // might be long? but initialize it only once and change tag
//       p_locks = std::vector<std::mutex>(p);
//       V_locks = std::vector<std::mutex>(V); 
//       counter = 0;

//     }

//     bool replace_if_better(T element, int index, int current_tag, std::vector<int> &dist){
//       if ( (perishable_pointers[index].tag != current_tag) || (  (perishable_pointers[index].ptr)->cost + dist[(perishable_pointers[index].ptr)->from] < dist[(perishable_pointers[index].ptr)->vertex]    ) ){
//         // take a lock.
//         // no need to lock distance as it is not modified in this phase
//         V_locks[index].lock();
//         if ( (perishable_pointers[index].tag != current_tag) || (  (perishable_pointers[index].ptr)->cost + dist[(perishable_pointers[index].ptr)->from] < dist[(perishable_pointers[index].ptr)->vertex]    ) ){
//           int ll_index = counter++;
//           p_locks[ll_index % p].lock();
//           data[ll_index % p].push_back(element); // TODO: define non-blocking pushback, that returns pointer to it
//           perishable_pointers[index] = perishable_pointer(&data[ll_index % p].back(), current_tag);
//           p_locks[ll_index % p].unlock();
//           V_locks[index].unlock();
//           return true;
//         }
//       }
//       return false;
//     }

//     bool contains(int index,int current_tag){
//       return perishable_pointers[index].tag == current_tag;
//     }

//     void reset(){
//       // delete all content of linkedlists
//       for (int i = 0; i < p; i++){
//         data[i].clear();
//       }
//     }
//     void display(){
//       for(int i=0;i<p;i++){
//         std::cout << "List " << i << ": ";
//         for (T element : data[i]){
//           std::cout << "(" << element.from << ","<< element.vertex << "," << element.cost << "),  ";
//         }
//         std::cout << "\n";
//       }
//     }

// };

// struct Edge {
//   int from;
//   int vertex;
//   int cost;
//   Edge(int vertex_, int cost_) {
//     vertex = vertex_;
//     cost = cost_;
//   }
//   Edge(int from_, int vertex_, int cost_) {
//     from = from_;
//     vertex = vertex_;
//     cost = cost_;
//   }
//   bool operator==(const Edge &other) const { return from == other.from && vertex == other.vertex && cost == other.cost; }
// };

// Define a hash function
// struct std::hash<Edge> {
//   std::size_t operator()(const Edge &e) const {
//     // Combine the hashes of the individual members
//     std::size_t h0 = std::hash<int>()(e.from);
//     std::size_t h1 = std::hash<int>()(e.vertex);
//     std::size_t h2 = std::hash<int>()(e.cost);
//     return h0 ^ (h1 << 1) ^ (h2 << 2);
//   }
// };

// struct SourceTargetReturn {
//   std::vector<int> path;
//   std::vector<int> distance;
//   SourceTargetReturn(std::vector<int> path_, std::vector<int> distance_) : path(path_), distance(distance_) {
//     // Set to -1 all unreachable nodes
//     for (size_t i = 0; i < distance.size(); i++) {
//       if (distance[i] == INT_MAX) {
//         distance[i] = -1;
//       }
//     }
//   }
// };

// struct SourceAllReturn {
//   std::vector<int> distances;
//   SourceAllReturn(std::vector<int> distances_) : distances(distances_) {
//     // Set to -1 all unreachable nodes
//     for (size_t i = 0; i < distances.size(); i++) {
//       if (distances[i] == INT_MAX) {
//         distances[i] = -1;
//       }
//     }
//   }
// };

// struct AllTerminalReturn {
//   std::vector<std::vector<int>> distances;
//   AllTerminalReturn(std::vector<std::vector<int>> distances_) : distances(distances_) {
//     // Set to -1 all unreachable nodes
//     for (size_t i = 0; i < distances.size(); i++) {
//       for (size_t j = 0; j < distances[i].size(); j++) {
//         if (distances[i][j] == INT_MAX) {
//           distances[i][j] = -1;
//         }
//       }
//     }
//   }
// };

// int n_digits(int n) {
//   if (n == 0) {
//     return 1;
//   } else if (n == -1) {
//     return 2;
//   }
//   int digits = 0;
//   while (n) {
//     n /= 10;
//     digits++;
//   }
//   return digits;
// }

// void print_spaced(int x, int n) {
//   int digits = n_digits(x);
//   std::cout << x;
//   for (int i = 0; i < n - digits; i++) {
//     std::cout << " ";
//   }
// }

// void printDistMatrix(std::vector<std::vector<int>> distances, int V) {
//   std::vector<int> max_dist(V + 1, -1);
//   for (int X = 0; X < V; X++) {
//     max_dist[X] = std::max(max_dist[X], X);
//     for (int Y = 0; Y < V; Y++) {
//       max_dist[Y + 1] = std::max(max_dist[Y + 1], distances[X][Y]);
//       if (distances[X][Y] == -1) {
//         max_dist[Y + 1] = std::max(max_dist[Y + 1], 10);
//       }
//     }
//   }
//   max_dist[0] = V - 1;
//   for (int X = 0; X < V + 1; X++) {
//     max_dist[X] = n_digits(max_dist[X]) + 1;
//   }
//   // max_dist[i] is now the maximum number of digits in the i-th column (0 is index)

//   std::cout << "Distances: \n";
//   for (int X = 0; X < V + 1; X++) {
//     if (X == 0) {
//       std::cout << "TO ";
//       for (int i = 0; i < max_dist[0] + 5 - 3; ++i) {
//         std::cout << " ";
//       }
//     } else {
//       std::cout << "FROM ";
//       print_spaced(X - 1, max_dist[0]);
//     }
//     std::cout << " ";

//     for (int Y = 1; Y < V + 1; Y++) {
//       if (X == 0) {
//         print_spaced(Y - 1, max_dist[Y]);
//       } else {
//         print_spaced(distances[X - 1][Y - 1], max_dist[Y]);
//       }
//     }
//     std::cout << "\n";
//   }
//   std::cout << "\n\n"
//             << std::flush;
// }

class Graph {
  // Directed weighted graph
  const int V;
  std::unordered_set<Edge> *adj;

public:
  int delta;
  int n_threads;
  Graph(int V, int delta, int n_threads_) : V(V), delta(delta) {
    adj = new std::unordered_set<Edge>[V];
    n_threads = std::min(n_threads_, V);
  }

  void addEdge(int v, int w, int c) {
    if (v < V && w < V && c >= 0) {
      adj[v].insert(Edge(v, w, c));
    } else {
      throw "Invalid vertex";
    }
  }

  void display() {
    for (int i = 0; i < V; i++) {
      std::cout << i << ": ";
      for (const Edge e : adj[i]) {
        std::cout << e.vertex << "(" << e.cost << ") ";
      }
      std::cout << "\n";
    }
  }

  static Graph generate_graph_parallel(int n_vertices, double edge_density, int max_cost, int n_threads, int delt) {
    Graph g(n_vertices, delt, n_threads);
    std::vector<std::thread> threads(n_threads - 1);
    int block_size = n_vertices / n_threads;
    //int n_edges = n_vertices * (n_vertices - 1) * edge_density;
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread([&g, i, block_size, n_vertices, edge_density, max_cost]() {
        std::hash<std::thread::id> hasher;
        static thread_local std::mt19937 generator = std::mt19937(clock() + hasher(std::this_thread::get_id()));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int it = i * block_size; it < (i + 1) * block_size; it++) {
          for (int j = 0; j < n_vertices; j++) {
            if (it != j && distribution(generator) < edge_density) {
              g.addEdge(it, j, (int)(distribution(generator) * max_cost));
            }
          }
        }
      });
    }
    std::hash<std::thread::id> hasher;
    static thread_local std::mt19937 generator = std::mt19937(clock() + hasher(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int it = (n_threads - 1) * block_size; it < n_vertices; it++) {
      for (int j = 0; j < n_vertices; j++) {
        if (it != j && distribution(generator) < edge_density) {
          g.addEdge(it, j, (int)(distribution(generator) * max_cost));
        }
      }
    }
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }
    // make matrix out of adjacency of g
    std::vector<std::vector<int>> dist(n_vertices, std::vector<int>(n_vertices, -1));
    for (int i = 0; i < n_vertices; i++) {
      for (const Edge e : g.adj[i]) {
        dist[i][e.vertex] = e.cost;
      }
      std::cout << "\n";
    }
    return g;
  }

  void compare_algorithms(int s, int d, bool debug = true) {
    std::vector<std::string> names_ST{"DijkstraSourceTarget", "Delta", "CustomDelta"};//, "UNWEIGHTED BFS_SourceTarget", "UNWEIGHTED DFS_SourceTarget"};
    std::vector<SourceTargetReturn (Graph::*)(int, int)> ST_Funcs{&Graph::DijkstraSourceTarget, &Graph::parallelDeltaStepping,&Graph::customParallelDeltaStepping};//, &Graph::BFS_ST, &Graph::DFS_ST};
    for (size_t i = 0; i < names_ST.size(); i++) {
      std::cout << "   " << names_ST[i] << ": " << std::flush;
      auto start = high_resolution_clock::now();
      SourceTargetReturn r = ((*this).*ST_Funcs[i])(s, d);
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << (double)duration.count() / 1000 << " milliseconds. \n\n";
      if (debug) {
        std::cout << "Distances: \n";
        for (int v = 0; v < V; v++) {
          std::cout << v << ": " << r.distance[v] << "\n";
        }

        if (r.distance[d] != INT_MAX) {
          std::cout << "Path: ";
          for (int v : r.path) {
            std::cout << v << " ";
          }
        } else {
          std::cout << "No path found.";
        }
        std::cout << "\n\n";
      }
      std::cout << std::flush;
      if (r.distance[d] != INT_MAX) {
          std::cout << "Distance: " << r.distance[d] << "\n";
          std::cout << "Path: ";
          for (int v : r.path) {
            std::cout << v << " ";
          }
        } else {
          std::cout << "No path found.";
        }
        std::cout << "\n\n";
    }
    
  return;
    // Define similar things for all terminal from the source-all terminal problem
    std::vector<std::string> names_SA{"DijkstraSourceAll", "UNWEIGHTED BFS_SourceAll", "UNWEIGHTED DFS_SourceAll"};
    std::vector<SourceAllReturn (Graph::*)(int, int)> SA_Funcs{&Graph::DijkstraSourceAll, &Graph::BFS_AT, &Graph::DFS_AT};
    for (size_t i = 0; i < names_SA.size(); i++) {
      std::cout << "   " << names_SA[i] << ": " << std::flush;
      auto start = high_resolution_clock::now();
      SourceAllReturn r = ((*this).*SA_Funcs[i])(s, d);
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << (double)duration.count() / 1000 << " milliseconds. \n\n";
      if (debug) {
        std::cout << "Distances: \n";
        for (int v = 0; v < V; v++) {
          std::cout << v << ": " << r.distances[v] << "\n";
        }
        std::cout << "\n\n";
      }
      std::cout << std::flush;
      if (r.distances[d] != INT_MAX) {
          std::cout << "Distance: " << r.distances[d] << "\n";
      }
        std::cout << "\n\n";
    }

    // Generate All-Terminal from Source-All-Terminal (SEQUENTIAL)
    for (size_t i = 0; i < names_SA.size(); i++) {
      std::cout << "   Sequential All-Terminal " << names_SA[i] << ": " << std::flush;
      auto start = high_resolution_clock::now();
      AllTerminalReturn r = SourceAll_To_AllTerminal(SA_Funcs[i]);
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << (double)duration.count() / 1000 << " milliseconds. \n\n";
      if (debug) {
        printDistMatrix(r.distances, V);
      }
      std::cout << std::flush;
    }

    // Generate All-Terminal from Source-All-Terminal (PARALLEL)
    for (size_t i = 0; i < names_SA.size(); i++) {
      std::cout << "   Parallel All-Terminal " << names_SA[i] << ": " << std::flush;
      auto start = high_resolution_clock::now();
      AllTerminalReturn r = SourceAll_To_AllTerminalParallel(SA_Funcs[i]);
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << (double)duration.count() / 1000 << " milliseconds. \n\n";
      if (debug) {
        printDistMatrix(r.distances, V);
      }
      std::cout << std::flush;
    }

    // Do All-Terminal algorithms
    std::vector<std::string> names_AT{"Floyd_Warshall_Sequential", "Floyd_Warshall_Parallel"};
    std::vector<AllTerminalReturn (Graph::*)()> AT_Funcs{&Graph::Floyd_Warshall_Sequential, &Graph::Floyd_Warshall_Parallel};
    for (size_t i = 0; i < names_AT.size(); i++) {
      std::cout << "   " << names_AT[i] << ": " << std::flush;
      auto start = high_resolution_clock::now();
      AllTerminalReturn r = ((*this).*AT_Funcs[i])();
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << (double)duration.count() / 1000 << " milliseconds. \n\n";
      if (debug) {
        printDistMatrix(r.distances, V);
      }
      std::cout << std::flush;
    }
  }

  AllTerminalReturn SourceAll_To_AllTerminal(SourceAllReturn (Graph::*F)(int, int)) {
    std::vector<std::vector<int>> distances;
    for (int i = 0; i < V; i++) {
      SourceAllReturn r = ((*this).*F)(i, i); // could do i, -1 too
      distances.push_back(r.distances);
    }
    return AllTerminalReturn(distances);
  }

  AllTerminalReturn SourceAll_To_AllTerminalParallel(SourceAllReturn (Graph::*F)(int, int)) {
    std::vector<std::vector<int>> distances(V);
    std::vector<std::thread> threads(n_threads);
    int block_size = V / n_threads;
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread([this, &distances, F, i, block_size]() {
        for (int it = i * block_size; it < (i + 1) * block_size; it++) {
          SourceAllReturn r = ((*this).*F)(it, it); // could do it, -1 too
          distances[it] = r.distances;
        }
      });
    }
    for (int it = (n_threads - 1) * block_size; it < V; it++) {
      SourceAllReturn r = ((*this).*F)(it, it); // could do it, -1 too
      distances[it] = r.distances;
    }
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }
    return AllTerminalReturn(distances);
  }

  SourceTargetReturn BFS_ST(int s, int d) { return FS(s, d, false, true); }
  SourceAllReturn BFS_AT(int s, int d) { return SourceAllReturn(FS(s, d, true, true).distance); }
  SourceTargetReturn DFS_ST(int s, int d) { return FS(s, d, false, false); }
  SourceAllReturn DFS_AT(int s, int d) { return SourceAllReturn(FS(s, d, true, false).distance); }

  // FS implementation (BFS, DFS, all_targets or not)
  SourceTargetReturn FS(int s, int d, bool all_targets, bool BFS) {
    std::vector<int> rpath;          // Path reconstruction
    std::unordered_set<int> visited; // Vertices already visited

    std::vector<int> prev(V, -1);      // Previous vertex in path list
    std::vector<int> dist(V, INT_MAX); // Distance list

    for (int i = 0; i < V; i++) {
      prev[i] = -1;
    }

    std::queue<int> queue;
    std::vector<int> pile;
    if (BFS) {
      queue.push(s);
    } else {
      pile.push_back(s);
    }
    visited.insert(s);
    dist[s] = 0;

    while ((BFS && !queue.empty()) || (!BFS && !pile.empty())) {
      int act;
      if (BFS) {
        act = queue.front();
        queue.pop();
      } else {
        act = pile.back();
        pile.pop_back();
      }

      if (!all_targets && act == d) {
        break;
      }
      for (const Edge e : adj[act]) {
        if (visited.find(e.vertex) == visited.end()) {
          if (BFS) {
            queue.push(e.vertex);
          } else {
            pile.push_back(e.vertex);
          }
          visited.insert(e.vertex);
          prev[e.vertex] = act;
          dist[e.vertex] = dist[act] + 1;
        }
      }
    }

    int u = d;
    while (u != -1) {
      rpath.push_back(u);
      u = prev[u];
    }

    std::vector<int> path;
    for (size_t i = 0; i < rpath.size(); ++i) {
      path.push_back(rpath[rpath.size() - i - 1]);
    }

    return SourceTargetReturn(path, dist);
  }

  SourceAllReturn DijkstraSourceAll(int s, int d) { return SourceAllReturn(Dijkstra(s, d, true).distance); }
  SourceTargetReturn DijkstraSourceTarget(int s, int d) { return Dijkstra(s, d, false); }

  // Dijkstra implementation for (positively) weighted graphs
  SourceTargetReturn Dijkstra(int s, int d, bool all_targets) {
    std::vector<int> rpath;                      // Path reconstruction
    std::unordered_set<int> visited;             // Vertices already visited
    std::unordered_set<int> reachable_unvisited; // Next vertices to visit (should make dijkstra faster)
    std::vector<int> dist(V);                    // Distance list
    std::vector<int> prev(V);                    // Previous vertex in path list

    for (int i = 0; i < V; i++) {
      dist[i] = INT_MAX;
      prev[i] = -1;
    }

    dist[s] = 0;
    reachable_unvisited.insert(s);
    while (!reachable_unvisited.empty()) {
      // Pick closest element in reachables
      int mindist = INT_MAX;
      int minvert = -1;
      for (const int &unvisited_vertex : reachable_unvisited) {
        if (dist[unvisited_vertex] < mindist) {
          minvert = unvisited_vertex;
          mindist = dist[minvert];
        }
      }

      if (!all_targets && minvert == d){break;}

      // Set it to visited
      reachable_unvisited.erase(minvert);
      visited.insert(minvert);

      // Update distances/predecessors of unvisited neighbors
      for (const Edge e : adj[minvert]) {
        if (visited.find(e.vertex) == visited.end()) {
          reachable_unvisited.insert(e.vertex);
          if (dist[e.vertex] > dist[minvert] + e.cost) {
            dist[e.vertex] = dist[minvert] + e.cost;
            prev[e.vertex] = minvert;
          }
        }
      }
    }

    int u = d;
    while (u != -1) {
      rpath.push_back(u);
      u = prev[u];
    }

    std::vector<int> path;
    for (size_t i = 0; i < rpath.size(); ++i) {
      path.push_back(rpath[rpath.size() - i - 1]);
    }

    return SourceTargetReturn(path, dist);
  }

  AllTerminalReturn Floyd_Warshall_Sequential() {
    std::vector<std::vector<int>> dist(V, std::vector<int>(V, INT_MAX));

    // Initialize the distance of points to themselves to 0
    for (int i = 0; i < V; i++) {
      dist[i][i] = 0;
    }

    // Initialize the distance of points to their neighbors
    for (int i = 0; i < V; i++) {
      for (const Edge e : adj[i]) {
        dist[i][e.vertex] = e.cost;
      }
    }

    // Iterate over all intermediate points
    for (int k = 0; k < V; k++) {
      for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
          if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]) {
            dist[i][j] = dist[i][k] + dist[k][j];
          }
        }
      }
    }
    return AllTerminalReturn(dist);
  }

  AllTerminalReturn Floyd_Warshall_Parallel() {
    std::vector<std::vector<int>> dist(V, std::vector<int>(V, INT_MAX));

    // Initialize the distance of points to themselves to 0
    for (int i = 0; i < V; i++) {
      dist[i][i] = 0;
    }

    // Initialize the distance of points to their neighbors
    for (int i = 0; i < V; i++) {
      for (const Edge e : adj[i]) {
        dist[i][e.vertex] = e.cost;
      }
    }

    // Initialize n_threads barriers for synchronization
    std::barrier barrier(n_threads);
    std::vector<std::thread> threads(n_threads - 1);
    int block_size = V / n_threads;

    for (int block = 0; block < n_threads - 1; block++) {
      threads[block] = std::thread([this, &dist, block, block_size, &barrier]() {
        for (int k = 0; k < V; k++) {
          for (int i = block * block_size; i < (block + 1) * block_size; i++) {
            if (i == k) {
              continue;
            }
            for (int j = 0; j < V; j++) {
              if (j == k || j == i) {
                continue;
              }
              if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]) {
                dist[i][j] = dist[i][k] + dist[k][j];
              }
            }
          }
          barrier.arrive_and_wait();
        }
      });
    }
    // Last block
    for (int k = 0; k < V; k++) {
      for (int i = (n_threads - 1) * block_size; i < V; i++) {
        if (i == k) {
          continue;
        }
        for (int j = 0; j < V; j++) {
          if (j == k || j == i) {
            continue;
          }
          if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]) {
            dist[i][j] = dist[i][k] + dist[k][j];
          }
        }
      }
      barrier.arrive_and_wait();
    }

    // Join threads
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }

    // Check the diagonal
    for (int i = 0; i < V; i++) {
      if (dist[i][i] < 0) {
        throw "Negative cycle detected";
      }
    }
    return AllTerminalReturn(dist);
  }

  // single thread delta-stepping
  // TODO: relax edges of a bucket in parallel
  // TODO: any other optimization that we can find in the data structures we use
  SourceTargetReturn deltaStepping(int source, int destination) {
    std::vector<int> dist(this->V, std::numeric_limits<int>::max());
    std::vector<int> prev(this->V, -1);
    std::unordered_map<int, std::list<int>> buckets;

    dist[source] = 0;
    buckets[0].push_back(source);

    while (!buckets.empty()) {
      int i = 0;
      while (buckets.find(i) == buckets.end()) {
        ++i;
      }

      std::list<int> bucket = buckets[i];
      buckets.erase(i);
      std::list<Edge> heavy_edges;

      // Process light edges
      while (!bucket.empty()) {
        int u = bucket.front();
        bucket.pop_front();
        for (const Edge e : adj[u]) {
          if (e.cost <= delta) {
            int new_dist = dist[u] + e.cost;
            if (new_dist < dist[e.vertex]) {
              dist[e.vertex] = new_dist;
              prev[e.vertex] = u;
              int bucket_index = new_dist / delta;
              if (bucket_index == i) {
                bucket.push_back(e.vertex); // should remove it from old bucket as well...
              } else if (buckets.find(bucket_index) == buckets.end()) {
                buckets[bucket_index] = std::list<int>({e.vertex});
              } else {
                buckets[bucket_index].push_back(e.vertex);
              }
            }
          } else {
            heavy_edges.push_back(e);
          }
        }
      }
      // Process heavy edges
      for (const Edge e : heavy_edges) {
        int new_dist = dist[e.from] + e.cost;
        if (new_dist < dist[e.vertex]) {
          dist[e.vertex] = new_dist;
          prev[e.vertex] = e.from;
          int bucket_index = new_dist / delta;
          if (buckets.find(bucket_index) == buckets.end()) {
            buckets[bucket_index] = std::list<int>({e.vertex});
          } else {
            buckets[bucket_index].push_back(e.vertex);
          }
        }
      }
    }

    // Make the return struct
    std::vector<int> rpath;
    for (int at = destination; at != -1; at = prev[at]) {
      rpath.push_back(at);
    }
    std::vector<int> path;
    for (size_t i = 0; i < rpath.size(); ++i) {
      path.push_back(rpath[rpath.size() - i - 1]);
    }
    return SourceTargetReturn(path, dist);
  }

  ////// parallel delta-stepping

  // helper functions
  bool update_dist(std::vector<int> &dist, std::vector<int> &prev, std::vector<std::mutex> &distlocks, int u, int v, int new_dist) {
    if (new_dist < dist[v]) {
      std::lock_guard<std::mutex> lock(distlocks[v]);
      if (new_dist < dist[v]) {
        dist[v] = new_dist;
        prev[v] = u;
      }
      return true;
    }
    return false;

  }
  void relaxThread(std::unordered_map<int, std::list<int>> &buckets,
                   std::vector<int> &dist,
                   std::vector<int> &prev,
                   std::vector<std::mutex> &distlocks,
                   Edge *start_edge,
                   Edge *end_edge) {
    double duration_operations = 0.;
    int operations = 0;
    while (start_edge != end_edge) {
      auto start = high_resolution_clock::now();
      Edge e = *start_edge;
      int new_dist = dist[e.from] + e.cost;
      if(update_dist(dist, prev, distlocks, e.from, e.vertex, new_dist)){
        int bucket_index = new_dist / delta;
        if (buckets.find(bucket_index) == buckets.end()) {
          buckets[bucket_index] = std::list<int>({e.vertex});
        } else {
          buckets[bucket_index].push_back(e.vertex);
        }
      }
      start_edge++;
      auto stop = high_resolution_clock::now();
      duration_operations += (double) (duration_cast<microseconds>(stop - start)).count() / 1000;
      operations++;
    }
    //std::cout<<"Thread finished with "<<operations<<" operations and "<<duration_operations<<" ms"<<std::endl;
  }
  void customRelaxThread(std::unordered_map<int, std::list<int>> &buckets,
                   std::vector<int> &dist,
                   std::vector<int> &prev,
                   ll_collection<Edge> &edges_collection,
                   int thread_id) {
    //std::cout<<"Inside relaxThread"<<std::endl;
    double duration_operations = 0.;
    int operations = 0;
    for(Edge e : edges_collection.data[thread_id]) {
      auto start = high_resolution_clock::now();
      int new_dist = dist[e.from] + e.cost;
      int v = e.vertex;
      if (new_dist < dist[v]) {
        dist[v] = new_dist;
        prev[v] = e.from;
        int bucket_index = new_dist / delta;
        if (buckets.find(bucket_index) == buckets.end()) {
          buckets[bucket_index] = std::list<int>({e.vertex});
        } else {
          buckets[bucket_index].push_back(e.vertex);
        }
      }
      auto stop = high_resolution_clock::now();
      duration_operations += (double) (duration_cast<microseconds>(stop - start)).count() / 1000;
      operations++;
    }
    //std::cout<<"Thread "<<thread_id<<" finished with "<<operations<<" operations and "<<duration_operations<<" ms"<<std::endl;
  }

  //int new_dist = dist[u] + e.cost;
  //          if (new_dist < dist[e.vertex]) {
  //            dist[e.vertex] = new_dist;
  //            prev[e.vertex] = u;
   //           int bucket_index = new_dist / delta;
   //           if (bucket_index == i) {
   //             bucket.push_back(e.vertex); // should remove it from old bucket as well...
   //           } else if (buckets.find(bucket_index) == buckets.end()) {
   //             buckets[bucket_index] = std::list<int>({e.vertex});
   //           } else {
   //             buckets[bucket_index].push_back(e.vertex);
   //           }
   //         }
  // TODOs:
  // 1. erase the edge from the bucket it was in
  // 2. think about compatibility of the data structures we use (for buckets, dist and prev) with parallelism
  // 3. adapt for light edges
  // Dynamic Partitioning with Global Buckets
  void parallelRelax(std::unordered_map<int, std::list<int>> &buckets,
                     std::vector<int> &dist,
                     std::vector<int> &prev,
                     std::vector<std::mutex> &distlocks,
                     std::vector<Edge> &edges,
                     int n_threads) {
    std::vector<std::thread> threads(n_threads - 1);
    int block_size = edges.size() / n_threads;
    Edge *start_edge = &edges[0];
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread(&Graph::relaxThread, this, std::ref(buckets), std::ref(dist), std::ref(prev), std::ref(distlocks), start_edge, start_edge + block_size);
      start_edge += block_size;
    }
    relaxThread(buckets, dist, prev, distlocks, start_edge, &edges[edges.size()]);
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }
  }

  void customParallelRelax(std::unordered_map<int, std::list<int>> &buckets,
                     std::vector<int> &dist,
                     std::vector<int> &prev,
                     //std::vector<std::mutex> &distlocks, // no need to, now!
                     ll_collection<Edge> &edges_collection) {
    //std::cout<<"Inside customParallelRelax"<<std::endl;
    std::vector<std::thread> threads(n_threads - 1);
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread(&Graph::customRelaxThread, this, std::ref(buckets), std::ref(dist), std::ref(prev), std::ref(edges_collection),i);
      //std::cout<<"Thread "<<i<<" created"<<std::endl;
    }
    customRelaxThread(buckets, dist, prev, edges_collection,n_threads-1);
    //std::cout<<"Last Thread created"<<std::endl;
    for (int i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }
  }

  SourceTargetReturn parallelDeltaStepping(int source, int destination) {
    std::vector<int> dist(this->V, INT_MAX);
    std::vector<int> prev(this->V, -1);
    std::unordered_map<int, std::list<int>> buckets;
    std::vector<std::mutex> distlocks(this->V);

    dist[source] = 0;
    buckets[0].push_back(source);
    double duration_exploration = 0.;
    double duration_heavy = 0.;
    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();


    while (!buckets.empty()) {
      start = high_resolution_clock::now();
      
      int i = 0;
      while (buckets.find(i) == buckets.end()) {
        ++i;
      }

      std::list<int> bucket = buckets[i];
      buckets.erase(i);
      std::vector<Edge> heavy_edges;

      // Process light edges
      while (!bucket.empty()) {
        // parallelize this loop
        int u = bucket.front();
        bucket.pop_front();
        for (const Edge e : adj[u]) {
          if (e.cost <= delta) {
            int new_dist = dist[u] + e.cost;
            if (new_dist < dist[e.vertex]) {
              dist[e.vertex] = new_dist;
              prev[e.vertex] = u;
              int bucket_index = new_dist / delta;
              if (bucket_index == i) {
                bucket.push_back(e.vertex); // should remove it from old bucket as well...
              } else if (buckets.find(bucket_index) == buckets.end()) {
                buckets[bucket_index] = std::list<int>({e.vertex});
              } else {
                buckets[bucket_index].push_back(e.vertex);
              }
            }
          } else {
            heavy_edges.push_back(e);
          }
        }
      }
      // Process heavy edges
      stop = high_resolution_clock::now();
      duration_exploration += (double) (duration_cast<microseconds>(stop - start)).count() / 1000;
      start = high_resolution_clock::now();
      parallelRelax(buckets, dist, prev, distlocks, heavy_edges, true);
      stop = high_resolution_clock::now();
      duration_heavy += (double) (duration_cast<microseconds>(stop - start)).count() / 1000;
    }

    std::cout << "Exploration time: " << duration_exploration << " milliseconds. \n";
    std::cout << "Heavy edges time: " << duration_heavy << " milliseconds. \n";

    // Make the return struct
    std::vector<int> rpath;
    for (int at = destination; at != -1; at = prev[at]) {
      rpath.push_back(at);
    }
    std::vector<int> path;
    for (size_t i = 0; i < rpath.size(); ++i) {
      path.push_back(rpath[rpath.size() - i - 1]);
    }
    return SourceTargetReturn(path, dist);
  }

  SourceTargetReturn customParallelDeltaStepping(int source, int destination) {
    std::vector<int> dist(this->V, INT_MAX);
    std::vector<int> prev(this->V, -1);
    std::unordered_map<int, std::list<int>> buckets;
    //std::vector<std::mutex> distlocks(this->V); // no need to anymore
    dist[source] = 0;
    buckets[0].push_back(source);

    ll_collection<Edge> heavy_edges(this->V, this->n_threads);
    ll_collection<Edge> light_edges(this->V, this->n_threads);

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    double duration_exploration = 0.;
    double duration_heavy = 0.;

    int iteration_heavy = 0;
    int iteration_light = 0;
    //std::cout<<"iteration "<<iteration<<std::endl;

    while (!buckets.empty()) {
      start = high_resolution_clock::now();
      int i = 0;
      while (buckets.find(i) == buckets.end()) {
        ++i;
      }

      std::list<int> bucket = buckets[i];

      // Process light edges
      while (!bucket.empty()) {
        buckets[i] = std::list<int>({}); // clear the bucket
        // parallelize this loop
        int u = bucket.front();
        bucket.pop_front();
        for (const Edge e : adj[u]) {
          if (e.cost <= delta) {
            light_edges.replace_if_better(e, e.vertex, iteration_light, dist);
          } else {
            heavy_edges.replace_if_better(e, e.vertex, iteration_light, dist);
          }
        }
        customParallelRelax(buckets, dist, prev, light_edges);
        light_edges.reset();
        bucket = buckets[i];
        iteration_light++;
      }
      buckets.erase(i);
      // Process heavy edges
      //std::cout<<"Heavy Request created "<<iteration<<std::endl;
      //heavy_edges.display();
      stop = high_resolution_clock::now();
      duration_exploration += (double) (duration_cast<microseconds>(stop - start)).count() / 1000;
      start = high_resolution_clock::now();
      customParallelRelax(buckets, dist, prev, heavy_edges);
      stop = high_resolution_clock::now();
      heavy_edges.reset();
      duration_heavy += (double) (duration_cast<microseconds>(stop - start)).count() / 1000;
      iteration_heavy++;
    }

    std::cout << "Exploration time: " << duration_exploration << " milliseconds. \n";
    std::cout << "Heavy edges time: " << duration_heavy << " milliseconds. \n";
    // Make the return struct
    std::vector<int> rpath;
    for (int at = destination; at != -1; at = prev[at]) {
      rpath.push_back(at);
    }
    std::vector<int> path;
    for (size_t i = 0; i < rpath.size(); ++i) {
      path.push_back(rpath[rpath.size() - i - 1]);
    }
    return SourceTargetReturn(path, dist);
  }
};

int main() {
  // test the ll_collection
  std::vector<int> dist(10, 10);
  Edge e(0, 1, 2);
  Edge e3(0, 2, 3);
  Edge e2(0, 2, 1);
  ll_collection<Edge> c(10, 2);
  c.replace_if_better(e, 1, 0, dist);
  c.replace_if_better(e2, 2, 0, dist);
  c.replace_if_better(e3, 2, 0, dist);
  std::cout<<(c.perishable_pointers[1].ptr)->cost<<std::endl;
  std::cout<<(c.perishable_pointers[2].ptr)->cost<<std::endl;
  std::cout<<c.data[0].begin()->vertex<<std::endl;
  std::cout<<c.data[1].begin()->vertex<<std::endl;
  std::cout<<"?"<<std::endl;
  c.reset();
  c.replace_if_better(e, 1, 1, dist);
  c.replace_if_better(e3, 2, 1, dist);
  std::cout<<(c.perishable_pointers[1].ptr)->cost<<std::endl;
  std::cout<<(c.perishable_pointers[2].ptr)->cost<<std::endl;
  std::cout<<c.data[0].begin()->vertex<<std::endl;
  std::cout<<c.data[1].begin()->vertex<<std::endl;
  std::cout<<"?"<<std::endl;




  Graph g = Graph::generate_graph_parallel(100, 0.4, 100, 4, 1);
   std::cout<<" 12 TWELVE THREADS"<<std::endl;
  g.compare_algorithms(0, 3, false);
  g.n_threads = 1;
  std::cout<<" 1 ONE THREAD"<<std::endl;
  g.compare_algorithms(0, 3, false);
   //std::cout<<" 5 FIVE THREADS"<<std::endl;
  //g.n_threads = 5;
  //g.compare_algorithms(0, 3, false);
  return 0;
}