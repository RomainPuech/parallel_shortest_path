#include <iostream>
#include <list>
#include <unordered_set>

#include <chrono>
using namespace std::chrono;

#include <thread>
#include <mutex>

class Graph {
    int V;
    std::unordered_set<int>* adj;

public:
    Graph(int V): V(V) {
        adj = new std::unordered_set<int>[V];
    }

    void addEdge(int v, int w) {
        if (v < V && w < V) {
            adj[v].insert(w);
            adj[w].insert(v);
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
    void deltaStepping(int s, int d) {
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

    // multi-thread delta-stepping
    // TODO


};

void compare_algorithms(Graph& g, int s, int d) {
    auto start = high_resolution_clock::now();
    g.shortestPath(s, d);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << (int) duration.count()/1000 << " miliseconds for Dijkstra"<< std::endl;

    start = high_resolution_clock::now();
    g.deltaStepping(s, d);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << (int) duration.count()/1000 << " miliseconds for Delta-Stepping"<< std::endl;
}

int main() {
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 3);
    g.addEdge(3, 4);
    g.addEdge(4, 5);

    compare_algorithms(g, 0, 5);
    return 0;
}