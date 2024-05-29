#include <iostream>
#include <list>
#include <unordered_set>

#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <limits>
#include <set>
#include <queue>


#include <chrono>
using namespace std::chrono;

#include <thread>
#include <mutex>

class Graph {
    int V;
    std::unordered_set<int>* adj;
    std::unordered_map<int, int>* weights;

public:
    Graph(int V): V(V) {
        adj = new std::unordered_set<int>[V];
        weights = new std::unordered_map<int, int>[V];
    }

    void addEdge(int u, int v, int w) {
        if (u < V && v < V) {
            adj[u].insert(v);
            adj[v].insert(u);
            weights[u][v] = w;
            weights[v][u] = w;
        } else {
            throw "Invalid vertex";
        }
    }

    //shortest path using dijkstra
    void shortestPath(int s, int d) {
        std::list<int> path;
        std::unordered_set<int> visited;
        std::unordered_set<int> unvisited;
        int dist[V];
        int prev[V];

        for (int i = 0; i < V; i++) {
            dist[i] = INT_MAX;
            prev[i] = -1;
            unvisited.insert(i);
        }

        dist[s] = 0;

        while (!unvisited.empty()) {
            int u = -1;
            int min = INT_MAX;
            for (int i = 0; i < V; i++) {
                if (unvisited.find(i) != unvisited.end() && dist[i] < min) {
                    u = i;
                    min = dist[i];
                }
            }

            if (u == -1) {
                break;
            }

            unvisited.erase(u);
            visited.insert(u);

            for (int v : adj[u]) {
                if (visited.find(v) == visited.end()) {
                    int alt = dist[u] + 1;
                    if (alt < dist[v]) {
                        dist[v] = alt;
                        prev[v] = u;
                    }
                }
            }
        }

        int u = d;
        while (u != -1) {
            path.push_front(u);
            u = prev[u];
        }

        for (int v : path) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    // single thread delta-stepping
    // TODO: currently the algo doesn't terminate, the loop condition is probably wrong
    // TODO: test that the output agrees with Dijkstra
    // TODO: destination is not used
    // TODO: relax edges of a bucket in parallel
    // TODO: idea: order the edges by weight for a single pass instead of 2.
    // TODO: any other optimization that we can find in the data structures we use

    void deltaStepping(int source, int destination, int delta) {
    std::vector<int> dist(this->V, std::numeric_limits<int>::max());
    std::vector<int> prev(this->V, -1);
    std::unordered_map<int, std::list<int>> buckets;
    std::set<int> unvisited;

    dist[source] = 0;
    buckets[0].push_back(source);

    for (int i = 0; !buckets.empty(); ++i) {
        if (buckets.find(i) == buckets.end()) continue;

        std::list<int> bucket = buckets[i];
        buckets.erase(i);

        // Process light edges
        while (!bucket.empty()) {
            int u = bucket.front();
            bucket.pop_front();

            for (auto &v : this->adj[u]) {
                int weight = this->weights[u][v];
                if (weight <= delta) {
                    int new_dist = dist[u] + weight;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        prev[v] = u;
                        int bucket_index = new_dist / delta;
                        buckets[bucket_index].push_back(v);
                    }
                }
            }
        }

        // Process heavy edges
        for (int u : bucket) {
            for (auto &v : this->adj[u]) {
                int weight = this->weights[u][v];
                if (weight > delta) {
                    int new_dist = dist[u] + weight;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        prev[v] = u;
                        int bucket_index = new_dist / delta;
                        buckets[bucket_index].push_back(v);
                    }
                }
            }
        }
    }

    // Output distances
    std::cout << "Vertex distances from source:" << std::endl;
    for (int i = 0; i < this->V; ++i) {
        std::cout << "Vertex " << i << ": " << dist[i] << std::endl;
    }

    // Output shortest paths
    std::cout << "\nShortest paths from source:" << std::endl;
    for (int i = 0; i < this->V; ++i) {
        if (i != source) {
            std::cout << "Path to " << i << ": ";
            if (dist[i] == std::numeric_limits<int>::max()) {
                std::cout << "No path" << std::endl;
            } else {
                std::vector<int> path;
                for (int at = i; at != -1; at = prev[at]) {
                    path.push_back(at);
                }
                for (int j = path.size() - 1; j >= 0; --j) {
                    std::cout << path[j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}



};

void compare_algorithms(Graph& g, int s, int d) {
    auto start = high_resolution_clock::now();
    g.shortestPath(s, d);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << (int) duration.count()/1000 << " miliseconds for Dijkstra"<< std::endl;

    start = high_resolution_clock::now();
    g.deltaStepping(s, d, 3);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << (int) duration.count()/1000 << " miliseconds for Delta-Stepping"<< std::endl;
}

int main() {
    Graph g(6);
    g.addEdge(0, 1, 2);
    g.addEdge(0, 2, 5);
    g.addEdge(1, 3, 6);
    g.addEdge(2, 3, 1);
    g.addEdge(3, 4, 1);
    g.addEdge(4, 5, 3);

    compare_algorithms(g, 0, 5);
    return 0;
}