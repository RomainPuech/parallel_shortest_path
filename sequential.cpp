#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std::chrono;
#include <barrier>
#include <limits.h>
#include <mutex>
#include <random>
#include <thread>

#include "utils.hpp"

#pragma GCC diagnostic ignored "-Wvla"

#define DEBUG 0
#define ANALYSIS 1

class Graph {
  // Directed weighted graph
  const size_t V;
  std::unordered_set<Edge> *adj;

public:
  int delta;
  size_t n_threads;

  Graph(size_t V, int delta, size_t n_threads_) : V(V), delta(delta) {
    adj = new std::unordered_set<Edge>[V];
    n_threads = std::min(n_threads_, (size_t)V);
  }

  ~Graph() {
    delete adj;
  }

  void addEdge(size_t v, size_t w, size_t c) {
    if (v < V && w < V) {
      // Any graph generation function in parallel calls this with disjoint set of v from each thread
      // Thus we need no lock here
      adj[v].insert(Edge(v, w, c));
    } else {
      std::cout << "Invalid edge: " << v << " " << w << " " << c << std::endl;
      throw "Invalid vertex";
    }
  }

  void display() {
    for (size_t i = 0; i < V; i++) {
      std::cout << i << ": ";
      for (const Edge e : adj[i]) {
        std::cout << e.vertex << "(" << e.cost << ") ";
      }
      std::cout << "\n";
    }
  }

  void save_to_file(std::string filename) {
    std::ofstream file;
    file.open(filename);
    file << V << " " << delta << "\n";
    for (size_t i = 0; i < V; i++) {
      for (const Edge e : adj[i]) {
        file << i << " " << e.vertex << " " << e.cost << "\n";
      }
    }
    file.close();
  }

  void load_from_file(std::string filename) {
    delete adj;
    adj = new std::unordered_set<Edge>[V];
    std::ifstream file;
    file.open(filename);
    if (!file.is_open()) {
      std::cout << "File not found: " << filename << std::endl;
      throw "File not found";
    }
    size_t V_;
    int delta_;
    file >> V_ >> delta_;
    if (V_ != V || delta_ != delta) {
      std::cout << "Incompatible graph" << std::endl;
      throw "Incompatible graph";
    }
    size_t v, w, c;
    while (file >> v >> w >> c) {
      addEdge(v, w, c);
    }
    file.close();
  }

  static Graph generate_path_parallel(size_t n_vertices, int max_cost, size_t n_threads, int delt) {
    // Creates a path from 0 to 1 with n_vertices vertices
    // addEdge is thread_safe here
    if (n_threads < 1) {
      std::cout << "Invalid number of threads" << std::endl;
      throw "Invalid number of threads";
    }
    Graph g(n_vertices, delt, n_threads);
    std::vector<std::thread> threads(n_threads - 1);
    size_t block_size = (n_vertices - 1) / n_threads;

    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread([&g, i, block_size, n_vertices, max_cost]() {
        std::hash<std::thread::id> hasher;
        static thread_local std::mt19937 generator = std::mt19937(clock() + hasher(std::this_thread::get_id()));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (size_t it = i * block_size; it < (i + 1) * block_size; it++) {
          int next = it + 1;
          if (it == 0) {
            next = 2;
          }
          if (it != 1) {
            g.addEdge(it, next, (int)(distribution(generator) * max_cost));
          } else {
            g.addEdge(n_vertices - 1, it, (int)(distribution(generator) * max_cost));
          }
        }
      });
    }

    std::hash<std::thread::id> hasher;
    static thread_local std::mt19937 generator = std::mt19937(clock() + hasher(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t it = (n_threads - 1) * block_size; it < n_vertices - 1; it++) {
      int next = it + 1;
      if (it == 0) {
        next = 2;
      }
      if (it != 1) {
        g.addEdge(it, next, (int)(distribution(generator) * max_cost));
      } else {
        g.addEdge(n_vertices - 1, it, (int)(distribution(generator) * max_cost));
      }
    }

    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }

    return g;
  }

  static Graph generate_graph_parallel(size_t n_vertices, double edge_density, int max_cost, size_t n_threads, int delt) {
    // Any source/destination pair is interesting
    // addEdge is thread_safe here
    if (n_threads < 1) {
      std::cout << "Invalid number of threads" << std::endl;
      throw "Invalid number of threads";
    }
    Graph g(n_vertices, delt, n_threads);
    std::vector<std::thread> threads(n_threads - 1);
    size_t block_size = n_vertices / n_threads;

    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread([&g, i, block_size, n_vertices, edge_density, max_cost]() {
        std::hash<std::thread::id> hasher;
        static thread_local std::mt19937 generator = std::mt19937(clock() + hasher(std::this_thread::get_id()));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (size_t it = i * block_size; it < (i + 1) * block_size; it++) {
          for (size_t j = 0; j < n_vertices; j++) {
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
    for (size_t it = (n_threads - 1) * block_size; it < n_vertices; it++) {
      for (size_t j = 0; j < n_vertices; j++) {
        if (it != j && distribution(generator) < edge_density) {
          g.addEdge(it, j, (int)(distribution(generator) * max_cost));
        }
      }
    }
    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }
    /* make matrix out of adjacency of g
    std::vector<std::vector<int>> dist(n_vertices, std::vector<int>(n_vertices, -1));
    for (size_t i = 0; i < n_vertices; i++) {
      for (const Edge e : g.adj[i]) {
        dist[i][e.vertex] = e.cost;
      }
    }*/
    return g;
  }

  static Graph generate_network_parallel(size_t n_vertices_components, int n_components, double component_density, double connections_density, int max_cost, int n_threads, int delt) {
    // Each component will have density component_density
    // The graph taking components as vertices will have density connections_density
    // To make interesting paths, pick source 0 and destination > n_vertices_components
    // addEdge is thread_safe here
    if (n_threads < 1) {
      std::cout << "Invalid number of threads" << std::endl;
      throw "Invalid number of threads";
    }
    Graph g(n_vertices_components * n_components, delt, n_threads);
    std::vector<Graph> components;
    for (int i = 0; i < n_components; i++) {
      components.push_back(generate_graph_parallel(n_vertices_components, component_density, max_cost, n_threads, delt));
    }

    // Add the components to the network

    std::vector<std::thread> threads(n_threads - 1);
    int block_size = n_components / n_threads;
    for (int n_th = 0; n_th < n_threads - 1; ++n_th) {
      threads[n_th] = std::thread([&g, &components, n_th, block_size, n_vertices_components]() {
        for (int i = n_th * block_size; i < (n_th + 1) * block_size; i++) {
          for (size_t j = 0; j < n_vertices_components; j++) {
            for (const Edge e : components[i].adj[j]) {
              g.addEdge(i * n_vertices_components + j, i * n_vertices_components + e.vertex, e.cost);
            }
          }
        }
      });
    }

    for (int i = (n_threads - 1) * block_size; i < n_components; i++) {
      for (size_t j = 0; j < n_vertices_components; j++) {
        for (const Edge e : components[i].adj[j]) {
          g.addEdge(i * n_vertices_components + j, i * n_vertices_components + e.vertex, e.cost);
        }
      }
    }

    for (int n_th = 0; n_th < n_threads - 1; ++n_th) {
      threads[n_th].join();
    }

    // Connect the components fully randomly in parallel

    std::vector<std::thread> threads2(n_threads - 1);
    block_size = n_components / n_threads;
    for (int n_th = 0; n_th < n_threads - 1; ++n_th) {
      threads2[n_th] = std::thread([&g, n_th, block_size, n_vertices_components, n_components, connections_density, max_cost]() {
        std::hash<std::thread::id> hasher;
        static thread_local std::mt19937 generator = std::mt19937(clock() + hasher(std::this_thread::get_id()));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int i = n_th * block_size; i < (n_th + 1) * block_size; i++) {
          for (int j = 0; j < n_components; j++) {
            // If we add a connection between two components
            if (i != j && distribution(generator) < connections_density) {
              // Connect two random vertices from these components
              int u = i * n_vertices_components + (int)(distribution(generator) * n_vertices_components);
              int v = j * n_vertices_components + (int)(distribution(generator) * n_vertices_components);
              g.addEdge(u, v, (int)(distribution(generator) * max_cost));
            }
          }
        }
      });
    }

    std::hash<std::thread::id> hasher;
    static thread_local std::mt19937 generator = std::mt19937(clock() + hasher(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = (n_threads - 1) * block_size; i < n_components; i++) {
      for (int j = 0; j < n_components; j++) {
        // If we add a connection between two components
        if (i != j && distribution(generator) < connections_density) {
          // Connect two random vertices from these components
          int u = i * n_vertices_components + (int)(distribution(generator) * n_vertices_components);
          int v = j * n_vertices_components + (int)(distribution(generator) * n_vertices_components);
          g.addEdge(u, v, (int)(distribution(generator) * max_cost));
        }
      }
    }

    for (int n_th = 0; n_th < n_threads - 1; ++n_th) {
      threads2[n_th].join();
    }

    return g;
  }

  void compare_algorithms(int s, int d, bool debug = true) {
    std::vector<std::string> names_ST{"DijkstraSourceTarget", /* "Delta", "DeltaNoPara",*/ "CustomDeltaPara"};                                                                                        //, "UNWEIGHTED BFS_SourceTarget", "UNWEIGHTED DFS_SourceTarget"};
    std::vector<SourceTargetReturn (Graph::*)(int, int)> ST_Funcs{&Graph::DijkstraSourceTarget, /* &Graph::deltaStepping, &Graph::parallelDeltaStepping,*/ &Graph::customParallelDeltaSteppingForce}; //, &Graph::BFS_ST, &Graph::DFS_ST};
    for (size_t i = 0; i < names_ST.size(); i++) {
      std::cout << "   " << names_ST[i] << ": " << std::flush;
      auto start = high_resolution_clock::now();
      SourceTargetReturn r = ((*this).*ST_Funcs[i])(s, d);
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << (double)duration.count() / 1000 << " milliseconds. \n\n";
      if (debug) {
        std::cout << "Distances: \n";
        for (size_t v = 0; v < V; v++) {
          std::cout << v << ": " << r.distance[v] << "\n";
        }

        if (r.distance[d] != INT_MAX) {
          std::cout << "Path: ";
          for (size_t v : r.path) {
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
        for (size_t v = 0; v < V; v++) {
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
    for (size_t i = 0; i < V; i++) {
      SourceAllReturn r = ((*this).*F)(i, i); // could do i, -1 too
      distances.push_back(r.distances);
    }
    return AllTerminalReturn(distances);
  }

  AllTerminalReturn AllTerminalDelta() {
    return SourceAll_To_AllTerminal(customParallelDeltaSteppingForceAll);
  }

  AllTerminalReturn SourceAll_To_AllTerminalParallel(SourceAllReturn (Graph::*F)(int, int)) {
    if (n_threads < 1) {
      std::cout << "Invalid number of threads" << std::endl;
      throw "Invalid number of threads";
    }
    std::vector<std::vector<int>> distances(V);
    std::vector<std::thread> threads(n_threads);
    size_t block_size = V / n_threads;
    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread([this, &distances, F, i, block_size]() {
        for (size_t it = i * block_size; it < (i + 1) * block_size; it++) {
          SourceAllReturn r = ((*this).*F)(it, it); // could do it, -1 too
          distances[it] = r.distances;
        }
      });
    }
    for (size_t it = (n_threads - 1) * block_size; it < V; it++) {
      SourceAllReturn r = ((*this).*F)(it, it); // could do it, -1 too
      distances[it] = r.distances;
    }
    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }
    return AllTerminalReturn(distances);
  }

  // ST = Source-Target, AT = All-Terminal
  SourceTargetReturn BFS_ST(int s, int d) { return FS(s, d, false, true); }
  SourceAllReturn BFS_AT(int s, int d) { return SourceAllReturn(FS(s, d, true, true).distance); }
  SourceTargetReturn DFS_ST(int s, int d) { return FS(s, d, false, false); }
  SourceAllReturn DFS_AT(int s, int d) { return SourceAllReturn(FS(s, d, true, false).distance); }

  // FS = First-Search implementation (BFS, DFS, all_targets or not)
  SourceTargetReturn FS(int s, int d, bool all_targets, bool BFS) {
    std::vector<int> rpath;          // Path reconstruction
    std::unordered_set<int> visited; // Vertices already visited

    std::vector<int> prev(V, -1);      // Previous vertex in path list
    std::vector<int> dist(V, INT_MAX); // Distance list

    for (size_t i = 0; i < V; i++) {
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
    std::unordered_set<int> visited;             // Vertices already visited
    std::unordered_set<int> reachable_unvisited; // Next vertices to visit (should make dijkstra faster)
    std::vector<int> dist(V);                    // Distance list
    std::vector<int> prev(V);                    // Previous vertex in path list

    for (size_t i = 0; i < V; i++) {
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

      if (!all_targets && minvert == d) {
        break;
      }

      // Set it to visited
      reachable_unvisited.erase(minvert);
      visited.insert(minvert);

      // Update distances/predecessors of unvisited neighbors
      for (const Edge e : adj[minvert]) {
        if (!visited.contains(e.vertex)) {
          reachable_unvisited.insert(e.vertex);
          if (dist[e.vertex] > dist[minvert] + e.cost) {
            dist[e.vertex] = dist[minvert] + e.cost;
            prev[e.vertex] = minvert;
          }
        }
      }
    }

    std::vector<int> rpath = {};
    if (!all_targets) {
      // Path reconstruction
      int u = d;
      int len = 0;
      while (u != -1) {
        len++;
        u = prev[u];
      }

      u = d;
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
    return SourceTargetReturn(rpath, dist); // empty path if all_targets
  }

  AllTerminalReturn Floyd_Warshall_Sequential() {
    std::vector<std::vector<int>> dist(V, std::vector<int>(V, INT_MAX));

    // Initialize the distance of points to themselves to 0
    for (size_t i = 0; i < V; i++) {
      dist[i][i] = 0;
    }

    // Initialize the distance of points to their neighbors
    for (size_t i = 0; i < V; i++) {
      for (const Edge e : adj[i]) {
        dist[i][e.vertex] = e.cost;
      }
    }

    // Iterate over all intermediate points
    for (size_t k = 0; k < V; k++) {
      for (size_t i = 0; i < V; i++) {
        for (size_t j = 0; j < V; j++) {
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
    for (size_t i = 0; i < V; i++) {
      dist[i][i] = 0;
    }

    // Initialize the distance of points to their neighbors
    for (size_t i = 0; i < V; i++) {
      for (const Edge e : adj[i]) {
        dist[i][e.vertex] = e.cost;
      }
    }

    // Initialize n_threads barriers for synchronization
    std::barrier barrier(n_threads);
    std::vector<std::thread> threads(n_threads - 1);
    size_t block_size = V / n_threads;

    for (size_t block = 0; block < n_threads - 1; block++) {
      threads[block] = std::thread([this, &dist, block, block_size, &barrier]() {
        for (size_t k = 0; k < V; k++) {
          for (size_t i = block * block_size; i < (block + 1) * block_size; i++) {
            if (i == k) {
              continue;
            }
            for (size_t j = 0; j < V; j++) {
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
    for (size_t k = 0; k < V; k++) {
      for (size_t i = (n_threads - 1) * block_size; i < V; i++) {
        if (i == k) {
          continue;
        }
        for (size_t j = 0; j < V; j++) {
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
    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }

    // Check the diagonal
    for (size_t i = 0; i < V; i++) {
      if (dist[i][i] < 0) {
        std::cout << "Negative cycle detected" << std::endl;
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
    std::unordered_map<int, std::unordered_set<int>> buckets;

    dist[source] = 0;
    buckets[0].insert(source);

    while (!buckets.empty()) {
      int i = 0;
      while (buckets.find(i) == buckets.end()) {
        ++i;
      }

      std::unordered_set<int> bucket = buckets[i];
      buckets.erase(i);
      std::list<Edge> heavy_edges;

      // Process light edges
      while (!bucket.empty()) {
        auto it = bucket.begin();
        int u = *it;
        bucket.erase(it);
        for (const Edge e : adj[u]) {
          if (e.cost <= delta) {
            int new_dist = dist[u] + e.cost;
            if (new_dist < dist[e.vertex]) {
              if (dist[e.vertex] != std::numeric_limits<int>::max()) {
                int old_bucket_index = dist[e.vertex] / delta;
                if (old_bucket_index != i) { // really?
                  buckets[old_bucket_index].erase(e.vertex);
                  // buckets[old_bucket_index][e.vertex] = false;
                }
              }
              dist[e.vertex] = new_dist;
              prev[e.vertex] = u;
              int bucket_index = new_dist / delta;
              if (buckets.find(bucket_index) == buckets.end()) {
                buckets[bucket_index] = std::unordered_set<int>({e.vertex});
              } else {
                buckets[bucket_index].insert(e.vertex);
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
          if (dist[e.vertex] != std::numeric_limits<int>::max()) {
            int old_bucket_index = dist[e.vertex] / delta;
            if (old_bucket_index != i) { // really?
              buckets[old_bucket_index].erase(e.vertex);
            }
          }
          dist[e.vertex] = new_dist;
          prev[e.vertex] = e.from;
          int bucket_index = new_dist / delta;
          if (buckets.find(bucket_index) == buckets.end()) {
            buckets[bucket_index] = std::unordered_set<int>({e.vertex});
          } else {
            buckets[bucket_index].insert(e.vertex);
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
#if DEBUG
    double duration_operations = 0.;
#endif
    int operations = 0;
    while (start_edge != end_edge) {
#if DEBUG
      auto start = high_resolution_clock::now();
#endif
      Edge e = *start_edge;
      int new_dist = dist[e.from] + e.cost;
      if (update_dist(dist, prev, distlocks, e.from, e.vertex, new_dist)) {
        int bucket_index = new_dist / delta;
        if (buckets.find(bucket_index) == buckets.end()) {
          buckets[bucket_index] = std::list<int>({e.vertex});
        } else {
          buckets[bucket_index].push_back(e.vertex);
        }
      }
      start_edge++;
#if DEBUG
      auto stop = high_resolution_clock::now();
      duration_operations += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
      operations++;
#endif
    }
    // std::cout<<"Thread finished with "<<operations<<" operations and "<<duration_operations<<" ms"<<std::endl;
  }

  void customRelaxThread(std::unordered_map<int, std::unordered_map<int, bool>> &buckets,
                         std::vector<int> &dist,
                         std::vector<int> &prev,
                         ll_collection<Edge> &edges_collection,
                         std::map<int, std::mutex *> &bucket_locks,
                         std::mutex &general_mutex,
                         int thread_id,
                         double &duration) {

    int operations = 0;
    auto start = high_resolution_clock::now();
    for (Edge e : edges_collection.data[thread_id]) {
      int new_dist = dist[e.from] + e.cost;
      int v = e.vertex;
      if (new_dist < dist[v]) {
        int bucket_index = new_dist / delta;
        int old_bucket_index = dist[v] / delta;
        // remove from old bucket
        if (prev[v] != -1 and (old_bucket_index != bucket_index or old_bucket_index)) {
          /// if(buckets[old_bucket_index][v]){ //impossible that some other thread removed it
          general_mutex.lock();
          std::lock_guard<std::mutex> lock(*bucket_locks[old_bucket_index]);
          general_mutex.unlock();
          buckets[old_bucket_index].erase(v); // or turn to false?
          //}
        }
        dist[v] = new_dist;
        prev[v] = e.from;
        if (buckets.find(bucket_index) == buckets.end()) {
          // we need to create a new bucket
          // we lock the general mutex
          general_mutex.lock();
          // here is the part where we need map instead of unsorted map
          for (auto &b : bucket_locks) {
            b.second->lock();
          }
          if (bucket_locks.find(bucket_index) == bucket_locks.end()) {
            // we need to create its mutex.
            std::mutex *m = new std::mutex();
            m->lock();
            bucket_locks[bucket_index] = m;
          }
          std::unordered_map<int, bool> bkt = std::unordered_map<int, bool>({{e.vertex, true}});
          buckets[bucket_index] = bkt;
          // unlock all mutices in order
          general_mutex.unlock();
          for (auto &b : bucket_locks) {
            b.second->unlock();
          }
        } else {
          general_mutex.lock();
          std::lock_guard<std::mutex> lock(*bucket_locks[bucket_index]);
          general_mutex.unlock();
          buckets[bucket_index][e.vertex] = true;
        }
      }
      operations++;
    }
    auto stop = high_resolution_clock::now();
    duration = (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
    // std::cout<<"Thread "<<thread_id<<" finished with "<<operations<<" operations and "<<duration_operations<<" ms"<<std::endl;
  }

  // int new_dist = dist[u] + e.cost;
  //           if (new_dist < dist[e.vertex]) {
  //             dist[e.vertex] = new_dist;
  //             prev[e.vertex] = u;
  //            int bucket_index = new_dist / delta;
  //            if (bucket_index == i) {
  //              bucket.push_back(e.vertex); // should remove it from old bucket as well...
  //            } else if (buckets.find(bucket_index) == buckets.end()) {
  //              buckets[bucket_index] = std::list<int>({e.vertex});
  //            } else {
  //              buckets[bucket_index].push_back(e.vertex);
  //            }
  //          }
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
                     size_t n_threads) {
    std::vector<std::thread> threads(n_threads - 1);
    int block_size = edges.size() / n_threads;
    Edge *start_edge = &edges[0];
    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i] = std::thread(&Graph::relaxThread, this, std::ref(buckets), std::ref(dist), std::ref(prev), std::ref(distlocks), start_edge, start_edge + block_size);
      start_edge += block_size;
    }
    relaxThread(buckets, dist, prev, distlocks, start_edge, &edges[edges.size()]);
    for (size_t i = 0; i < n_threads - 1; i++) {
      threads[i].join();
    }
  }

  void customParallelRelax(std::unordered_map<int, std::unordered_map<int, bool>> &buckets,
                           std::vector<int> &dist,
                           std::vector<int> &prev,
                           // std::vector<std::mutex> &distlocks, // no need to, now!
                           ll_collection<Edge> &edges_collection,
                           std::map<int, std::mutex *> &bucket_locks,
                           std::mutex &general_mutex,
                           bool force_parallelization = false) {

    // std::cout<<"Inside customParallelRelax"<<std::endl;
    std::vector<std::thread> threads(n_threads - 1);
    if (force_parallelization and edges_collection.data[0].size() > 10000) {

      double total_duration = 0.;
      double durations[n_threads];
      auto start = high_resolution_clock::now();
      // we parallelize the relax operation
      std::unordered_set<int> ignored;
      for (size_t i = 0; i < n_threads - 1; i++) {
        // threads[i] = std::thread(&do_nothing);
        if (edges_collection.data[i].size() > 0) {
          threads[i] = std::thread(&Graph::customRelaxThread, this, std::ref(buckets), std::ref(dist), std::ref(prev), std::ref(edges_collection), std::ref(bucket_locks), std::ref(general_mutex), i, std::ref(durations[i]));
        } else {
          ignored.insert(i);
        }
      }
      if (edges_collection.data[n_threads - 1].size() > 0) {
        customRelaxThread(buckets, dist, prev, edges_collection, bucket_locks, general_mutex, n_threads - 1, durations[n_threads - 1]);
      } else {
        ignored.insert(n_threads - 1);
      }
      // std::cout<<"Last Thread created"<<std::endl;
      for (size_t i = 0; i < n_threads - 1; i++) {
        if (ignored.find(i) == ignored.end()) {
          threads[i].join();
        }
      }
      auto stop = high_resolution_clock::now();
      total_duration = (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
      double duration_operations = 0.;
      for (size_t i = 0; i < n_threads; i++) {
        if (ignored.find(i) == ignored.end()) {
          duration_operations += durations[i];
        }
      }
#if DEBUG
      if (duration_operations > total_duration) {
        std::cout << "worth it" << std::endl;
      } else {
        std::cout << "!NOT WORTH IT!" << std::endl;
      }
#endif
      // std::cout<<"sublist length: "<<edges_collection.data[0].size()<<std::endl;
    } else {
      for (size_t i = 0; i < n_threads; i++) {
        // we don t parallelize the relax operation
        double duration = 0.;
        if (edges_collection.data[i].size() > 0) {
          customRelaxThread(buckets, dist, prev, edges_collection, bucket_locks, general_mutex, i, duration);
        }
      }
    }
#if DEBUG
    std::cout << "Total duration: " << total_duration << std::endl;
    std::cout << "Operations duration: " << duration_operations << std::endl;
#endif

#if DEBUG
    int total_length = 0;
    for (size_t i = 0; i < n_threads; i++) {
      total_length += edges_collection.data[i].size();
    }
// std::cout << "Total length: " << total_length << std::endl;
#endif
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
      duration_exploration += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
      start = high_resolution_clock::now();
      parallelRelax(buckets, dist, prev, distlocks, heavy_edges, true);
      stop = high_resolution_clock::now();
      duration_heavy += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
    }

#if DEBUG
    std::cout << "Exploration time: " << duration_exploration << " milliseconds. \n";
    std::cout << "Heavy edges time: " << duration_heavy << " milliseconds. \n";
#endif

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

  void exploreNodesThread(std::unordered_map<int, bool>::iterator start, std::unordered_map<int, bool>::iterator end, std::vector<int> &dist, ll_collection<Edge> &light_edges, ll_collection<Edge> &heavy_edges, int iteration_light, int iteration_heavy, int delta) {
    int counter = 0;
    while (start != end) {
      counter++;
      if (counter % 100 == 0) {
      }
      int u = start->first;
      bool flag = start->second;
      if (flag) {
        for (const Edge e : adj[u]) {
          if (e.cost < delta) {
            light_edges.replace_if_better(e, e.vertex, iteration_light, dist);
          } else {
            heavy_edges.replace_if_better(e, e.vertex, iteration_heavy, dist);
          }
        }
      }
      start++;
    }
  }

  SourceTargetReturn customParallelDeltaStepping(int source, int destination, double force_parallelization) {
    std::vector<int> dist(this->V, INT_MAX);
    std::vector<int> prev(this->V, -1);
    std::unordered_map<int, std::unordered_map<int, bool>> buckets;
    std::map<int, std::mutex *> bucket_locks({{0, new std::mutex()}});
    std::mutex general_mutex;
    dist[source] = 0;
    buckets[0][source] = true;

    ll_collection<Edge> heavy_edges(this->V, this->n_threads);
    ll_collection<Edge> light_edges(this->V, this->n_threads);

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    double duration_exploration = 0.;
    double duration_heavy = 0.;

    int iteration_heavy = 0;
    int iteration_light = 0;

    while (!buckets.empty()) {
      start = high_resolution_clock::now();
      int i = 0;
      while (buckets.find(i) == buckets.end()) {
        ++i;
      }

      std::unordered_map<int, bool> bucket = buckets[i]; // this copy might be unecessary - not too long but optimizable

      // Process light edges
      while (!bucket.empty()) {
        buckets[i] = std::unordered_map<int, bool>(); // clear the bucket
        // parallelize this loop
        // parallelize according to the nodes (by edges would be more robust to unbalanced graphs)
        size_t n_threads = this->n_threads;
        if (bucket.size() < 500)
          n_threads = 1;

        // compute the workload per thread
        int workload = bucket.size() / n_threads;
        //
        std::vector<std::thread> workers(n_threads - 1);

        // prepare workload
        std::unordered_map<int, bool>::iterator start = bucket.begin();
        std::unordered_map<int, bool>::iterator end = bucket.begin();
        for (size_t j = 0; j < n_threads - 1; j++) {
          for (int i = 0; i < workload; i++) {
            end++;
          }
          workers[j] = std::thread(&Graph::exploreNodesThread, this, start, end, std::ref(dist), std::ref(light_edges), std::ref(heavy_edges), iteration_light, iteration_heavy, delta);
          start = end;
        }
        exploreNodesThread(start, bucket.end(), dist, light_edges, heavy_edges, iteration_light, iteration_heavy, delta);
        // join
        for (size_t j = 0; j < n_threads - 1; j++) {
          workers[j].join();
        }
        customParallelRelax(buckets, dist, prev, light_edges, bucket_locks, general_mutex, force_parallelization);
        light_edges.reset();
        bucket = buckets[i];
        iteration_light++;
      }
      buckets.erase(i);
      // Process heavy edges
      stop = high_resolution_clock::now();
      duration_exploration += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
      start = high_resolution_clock::now();
      customParallelRelax(buckets, dist, prev, heavy_edges, bucket_locks, general_mutex, force_parallelization);
      stop = high_resolution_clock::now();
      heavy_edges.reset();
      duration_heavy += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
      iteration_heavy++;
    }

#if DEBUG
    std::cout << "Exploration time: " << duration_exploration << " milliseconds. \n";
    std::cout << "Heavy edges time: " << duration_heavy << " milliseconds. \n";
#endif

    // free bucket locks
    for (auto &b : bucket_locks) {
      delete b.second;
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

  SourceTargetReturn customParallelDeltaSteppingForce(int source, int destination) {
    return customParallelDeltaStepping(source, destination, true);
  }
  SourceAllReturn customParallelDeltaSteppingForceAll(int source, int destination) {
    return SourceAllReturn(customParallelDeltaSteppingForce(source, destination).distance);
  }
  SourceTargetReturn customParallelDeltaSteppingNoForce(int source, int destination) {
    return customParallelDeltaStepping(source, destination, false);
  }
};

int main(int argc, char *argv[]) {
#if not ANALYSIS
  // process input
  int nodes = 500;
  double density = 0.5;
  int max_cost = 100;
  int n_threads = 1;
  int delta = 2;
  bool load_previous = false;
  bool save = true;

  if (argc > 1) {
    nodes = std::stoi(argv[1]);
  }
  if (argc > 2) {
    density = std::stod(argv[2]);
  }
  if (argc > 3) {
    max_cost = std::stoi(argv[3]);
  }
  if (argc > 4) {
    n_threads = std::stoi(argv[4]);
  }
  if (argc > 5) {
    delta = std::stoi(argv[5]);
  }
  if (argc > 6) {
    load_previous = std::stoi(argv[7]);
  }
  if (argc > 7) {
    save = std::stoi(argv[8]);
  }

  // "V, density, max_cost, n_threads, delta"
  int res1 = 0;
  int res2 = 0;
  int res3 = 0;
  while (res1 == res2 and res2 == res3) {
    Graph g = Graph::generate_graph_parallel(nodes, density, max_cost, n_threads, delta); // Graph::generate_network_parallel(20, 1, 0.15, 0.1, 10, 1, 3);
    g.load_from_file("graph.txt");

    auto start = high_resolution_clock::now();
    res2 = g.DijkstraSourceTarget(0, nodes - 1).distance[nodes - 1];
    auto stop = high_resolution_clock::now();
    double time = (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
    std::cout << "Dijkstra: " << time << " ms" << std::endl;

    std::cout << "\n";

    res1 = g.customParallelDeltaStepping(0, nodes - 1, true).distance[nodes - 1];

    std::cout << "\n";

    g.n_threads = 2;
    res3 = g.customParallelDeltaStepping(0, nodes - 1, true).distance[nodes - 1];

    std::cout << "\n";

    g.n_threads = 4;
    res3 = g.customParallelDeltaStepping(0, nodes - 1, true).distance[nodes - 1];

    std::cout << "\n";

    std::cout << "-------------------------------\n\n";
    // g.compare_algorithms(0, nodes - 1, false);
  }
  std::cout << "Done: " << res2 << res1 << res3 << std::endl;

  Graph g = Graph(nodes, delta, n_threads); // Graph::generate_network_parallel(20, 1, 0.15, 0.1, 10, 1, 3);
  g.load_from_file("graph.txt");
  g.compare_algorithms(0, nodes - 1, false);

#endif
#if ANALYSIS
  std::cout << "Yes" << std::endl;
  // number of times to repeat each experiment
  const size_t n_repeat = 3;

  // array of graph sizes and density
  std::vector<size_t> graph_sizes = {100, 200, 500, 1000, 1500};

  std::vector<double> densities = {0.3, 0.5, 0.7};

  // array of thread_numners
  std::vector<size_t> n_threads_vect = {4, 8, 12, 24};

  // array of deltas
  std::vector<int> deltas = {1, 2, 4};

  std::cout << "Starting algiorithms ..." << std::endl;

  std::cout << "Repeating each experiment " << n_repeat << " times" << std::endl;
  std::cout << "Graph sizes: ";
  for (size_t i = 0; i < graph_sizes.size(); i++) {
    std::cout << graph_sizes[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Densities: ";
  for (size_t i = 0; i < densities.size(); i++) {
    std::cout << densities[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Total number of graphs: " << graph_sizes.size() * densities.size() << std::endl;
  std::cout << "Number of threads for parallel algorithms: ";
  for (size_t i = 0; i < n_threads_vect.size(); i++) {
    std::cout << n_threads_vect[i] << " ";
  }
  std::cout << std::endl;

  // create a file to store the results
  std::ofstream file;
  // Add the current date to the file name and save it to a specific folder
  std::time_t t = std::time(0);
  std::tm *now = std::localtime(&t);
  std::string filename = "results_" + std::to_string(now->tm_year + 1900) + "-" + std::to_string(now->tm_mon + 1) + "-" + std::to_string(now->tm_mday) + "-" + std::to_string(now->tm_hour) + "-" + std::to_string(now->tm_min) + "-" + std::to_string(now->tm_sec) + ".csv";
  filename = "results_25000_0.7.csv";
  file.open(filename);

  std::cout << "Storing results in " << filename << std::endl;

  // write the header
  file << "algorithm,parallel,delta,graph_size,graph_density,n_threads,time,n_repeat" << std::endl;

  size_t count = 1;
  for (size_t i = 0; i < n_repeat; i++) {
    std::cout << "Repetition " << i + 1 << " of " << n_repeat << std::endl;
    // Loop over the graph sizes
    for (size_t graph_size : graph_sizes) {
      for (double density : densities) {

        double time = 0;
        size_t n_threads = 4;
        int delta = 1;
        size_t max_cost = 100;

        Graph g = Graph::generate_graph_parallel(graph_size, density, max_cost, n_threads, delta);

        // Sequential ALGOS
        std::cout << "Starting sequential algorithms for graph size " << graph_size << " and density " << density << std::endl;
        // Delta Stepping
        time = 0;
        g.delta = delta;
        auto start = high_resolution_clock::now();
        g.Floyd_Warshall_Sequential();
        auto stop = high_resolution_clock::now();
        time += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
        file << "FloydSequential," << false << "," << -1 << "," << graph_size << "," << density << "," << 1 << "," << time << "," << i << std::endl;

        // Dijkstra

        // Parallel ALGOS
        std::cout << "Starting parallel algorithms for graph size " << graph_size << " and density " << density << std::endl;
        for (size_t n_threads : n_threads_vect) {
          g.n_threads = n_threads;

          // floyd parallel
          time = 0;
          auto start = high_resolution_clock::now();
          g.Floyd_Warshall_Parallel();
          auto stop = high_resolution_clock::now();
          time += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
          file << "FloydParallel," << true << "," << -1 << "," << graph_size << "," << density << "," << 1 << "," << time << "," << i << std::endl;

          // Delta Stepping
          for (int delta : deltas) {
            time = 0;
            g.delta = delta;
            auto start = high_resolution_clock::now();
            int source = 0;
            int target = 101;
            bool force_parallel = true;
            g.AllTerminalDelta();
            auto stop = high_resolution_clock::now();
            time += (double)(duration_cast<microseconds>(stop - start)).count() / 1000;
            file << "ParallelDeltaSteppingAllTerminal," << (n_threads > 1) << "," << delta << "," << graph_size << "," << density << "," << n_threads << "," << time << "," << i << std::endl;
          }
        }
      }
    }
  }
#endif

  exit(1); // Success
}
