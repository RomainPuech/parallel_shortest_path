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

struct ShortestPathReturn{
    std::vector<int> path;
    std::vector<int> distance;
    ShortestPathReturn(std::vector<int> path_, std::vector<int> distance_): path(path_), distance(distance_){}
};

struct AllTerminalReturn{
    std::vector<std::vector<int>> distances;
    AllTerminalReturn(std::vector<std::vector<int>> distances_): distances(distances_){}
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

    void compare_algorithms(int s, int d, bool debug = true) {
        std::vector<std::string> names_shortest{"DijkastraSourceAll", "DijkstraSourceTarget", "Delta", "UNWEIGHTED BFS_SourceTarget", "UNWEIGHTED BFS_AllTargets", "UNWEIGHTED DFS_SourceTarget", "UNWEIGHTED DFS_AllTargets"};
        std::vector<ShortestPathReturn(Graph::*)(int, int)> shortest_paths {&Graph::DijkstraSourceAll, &Graph::DijkstraSourceTarget, &Graph::deltaStepping, &Graph::BFS_ST, &Graph::BFS_AT, &Graph::DFS_ST, &Graph::DFS_AT
        };
        for (size_t i = 0; i < names_shortest.size(); i++) {
            std::cout << "   " << names_shortest[i] << ": " << std::flush;
            auto start = high_resolution_clock::now();
            ShortestPathReturn r = ((*this).*shortest_paths[i])(s, d);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << (double) duration.count()/1000 << " milliseconds. \n\n";
            if (debug) {
                std::cout << "Distances: \n";
                for (int v = 0; v < V; v++) {std::cout << v << ": " << r.distance[v] << "\n";}

                if (r.distance[d] != INT_MAX){
                    std::cout << "Path: ";
                    for (int v : r.path) {std::cout << v << " ";}
                } else{std::cout << "No path found.";}
                std::cout << "\n\n";
            }
            std::cout << std::flush;
        }
    }

    ShortestPathReturn BFS_ST(int s, int d){return FS(s, d, false, true);}
    ShortestPathReturn BFS_AT(int s, int d){return FS(s, d, true, true);}
    ShortestPathReturn DFS_ST(int s, int d){return FS(s, d, false, false);}
    ShortestPathReturn DFS_AT(int s, int d){return FS(s, d, true, false);}

    // FS implementation (BFS, DFS, all_targets or not)
    ShortestPathReturn FS(int s, int d, bool all_targets, bool BFS){
        std::vector<int> rpath; // Path reconstruction
        std::unordered_set<int> visited; // Vertices already visited
        
        std::vector<int> prev(V, -1); // Previous vertex in path list
        std::vector<int> dist(V, INT_MAX); // Distance list

        for (int i = 0; i < V; i++) {prev[i] = -1;}

        std::queue<int> queue;
        std::vector<int> pile;
        if (BFS){queue.push(s);}
        else{pile.push_back(s);}
        visited.insert(s);
        dist[s] = 0;

        while ((BFS && !queue.empty()) || (!BFS && !pile.empty())) {
            int act;
            if (BFS){act = queue.front();queue.pop();}
            else{act = pile.back();pile.pop_back();}

            if (!all_targets && act == d) {break;}
            for (const Edge e : adj[act]){
                if (visited.find(e.vertex) == visited.end()){
                    if (BFS){queue.push(e.vertex);}
                    else{pile.push_back(e.vertex);}
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

        return ShortestPathReturn(path, dist);
    }

    ShortestPathReturn DijkstraSourceAll(int s, int d){return Dijkstra(s, d, true);}
    ShortestPathReturn DijkstraSourceTarget(int s, int d){return Dijkstra(s, d, false);}

    // Dijkstra implementation for (positively) weighted graphs
    ShortestPathReturn Dijkstra(int s, int d, bool all_targets) {
        std::vector<int> rpath; // Path reconstruction
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
            rpath.push_back(u);
            u = prev[u];
        }

        std::vector<int> path;
        for (size_t i = 0; i < rpath.size(); ++i) {path.push_back(rpath[rpath.size() - i - 1]);}

        return ShortestPathReturn(path, dist);
    }

    // single thread delta-stepping
    // TODO: currently the algo doesn't terminate, the loop condition is probably wrong
    // TODO: test that the output agrees with Dijkstra
    // TODO: destination is not used
    // TODO: relax edges of a bucket in parallel
    // TODO: idea: order the edges by weight for a single pass instead of 2.
    // TODO: any other optimization that we can find in the data structures we use
    ShortestPathReturn deltaStepping(int source, int destination) {
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

        // Make the return struct
        std::vector<int> rpath;
        for (int at = destination; at != -1; at = prev[at]) {
            rpath.push_back(at);
        }
        std::vector<int> path;
        for (size_t i = 0; i < rpath.size(); ++i) {path.push_back(rpath[rpath.size() - i - 1]);}
        return ShortestPathReturn(path, dist);
    }
};


int main() {
    Graph g(7, 1);
    g.addEdge(0, 1, 1);
    g.addEdge(0, 2, 1);
    g.addEdge(1, 3, 2);
    g.addEdge(2, 3, 1);
    g.addEdge(3, 4, 1);
    g.addEdge(4, 5, 1);
    g.addEdge(5, 6, 3);

    g.compare_algorithms(0, 5, false);
    return 0;
}