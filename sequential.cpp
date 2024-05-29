#include <iostream>
#include <list>
#include <unordered_set>
#include <algorithm>
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

#pragma GCC diagnostic ignored "-Wvla"

struct Edge{
    int vertex;
    int cost;
    Edge(int vertex_, int cost_) {vertex = vertex_; cost = cost_;}
    bool operator==(const Edge &other) const {return vertex == other.vertex && cost == other.cost;}
};

// Define a hash function
template <>
struct std::hash<Edge> {
    std::size_t operator()(const Edge &e) const {
        // Combine the hashes of the individual members
        std::size_t h1 = std::hash<int>()(e.vertex);
        std::size_t h2 = std::hash<int>()(e.cost);
        return h1 ^ (h2 << 1);
    }
};

class Graph {
    // Directed weighted graph
    const int V;
    std::unordered_set<Edge>* adj;

public:
    int delta;
    Graph(int V, int delta): V(V), delta(delta){
        adj = new std::unordered_set<Edge>[V];
    }
    
    void addEdge(int v, int w, int c) {
        if (v < V && w < V) {
            adj[v].insert(Edge(w, c));
        } else {
            throw "Invalid vertex";
        }
    }

    void compare_algorithms(int s, int d) {
        std::vector<std::string> names{"DijkastraSourceAll", "DijkstraSourceTarget", "Delta"};
        std::vector<void(Graph::*)(int, int)> functions {
            &Graph::DijkstraSourceAll,
            &Graph::DijkstraSourceTarget,
            &Graph::deltaStepping
        };
        for (size_t i = 0; i < names.size(); i++) {
            std::cout << "   " << names[i] << std::endl;
            auto start = high_resolution_clock::now();
            ((*this).*functions[i])(s, d);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << (double) duration.count()/1000 << " milliseconds." << std::endl;
        }
    }

    void DijkstraSourceAll(int s, int d){Dijkstra(s, d, true);}
    void DijkstraSourceTarget(int s, int d){Dijkstra(s, d, false);}

    // Dijkstra implementation for (positively) weighted graphs
    void Dijkstra(int s, int d, bool all_targets) {
        std::vector<int> path; // Path reconstruction
        std::unordered_set<int> visited; // Vertices already visited
        std::unordered_set<int> reachable_unvisited; // Next vertices to visit (should make dijkstra faster)
        std::vector<int> dist(V); // Distance list
        std::vector<int> prev(V); // Previous vertex in path list

        for (int i = 0; i < V; i++) {
            dist[i] = INT_MAX;
            prev[i] = -1;
        }

        dist[s] = 0;
        reachable_unvisited.insert(s);
        while ((dist[d] == INT_MAX || all_targets) && !reachable_unvisited.empty()) {
            // Pick closest element in reachables
            int mindist = INT_MAX;
            int minvert = -1;
            for (const int &unvisited_vertex : reachable_unvisited){
                if (dist[unvisited_vertex] < mindist){minvert = unvisited_vertex; mindist = dist[minvert];}
            }

            // Set it to visited
            reachable_unvisited.erase(minvert);
            visited.insert(minvert);

            // Update distances/predecessors of unvisited neighbors
            for (const Edge e : adj[minvert]){
                if (visited.find(e.vertex) == visited.end()){
                    reachable_unvisited.insert(e.vertex);
                    if (dist[e.vertex] > dist[minvert] + e.cost){
                        dist[e.vertex] = dist[minvert] + e.cost;
                        prev[e.vertex] = minvert;
                    }
                }
            }

        }

        int u = d;
        while (u != -1) {
            path.push_back(u);
            u = prev[u];
        }

        std::cout << "Distance (weighted): " << dist[d] << std::endl;

        for (size_t i = 0; i < path.size(); ++i) {
            std::cout << path[path.size() - i - 1] << " ";
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
    void deltaStepping(int source, int destination) {
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

                for (const Edge e : adj[u]) {
                    if (e.cost <= delta) {
                        int new_dist = dist[u] + e.cost;
                        if (new_dist < dist[e.vertex]) {
                            dist[e.vertex] = new_dist;
                            prev[e.vertex] = u;
                            int bucket_index = new_dist / delta;
                            buckets[bucket_index].push_back(e.vertex);
                        }
                    }
                }
            }

            // Process heavy edges
            for (int u : bucket) {
                for (const Edge e : adj[u]) {
                    if (e.cost > delta) {
                        int new_dist = dist[u] + e.cost;
                        if (new_dist < dist[e.vertex]) {
                            dist[e.vertex] = new_dist;
                            prev[e.vertex] = u;
                            int bucket_index = new_dist / delta;
                            buckets[bucket_index].push_back(e.vertex);
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


int main() {
    Graph g(6, 1);
    g.addEdge(0, 1, 1);
    g.addEdge(0, 2, 1);
    g.addEdge(1, 3, 2);
    g.addEdge(2, 3, 1);
    g.addEdge(3, 4, 1);
    g.addEdge(4, 5, 1);

    g.compare_algorithms(0, 5);
    return 0;
}