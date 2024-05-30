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
#include <barrier>

#pragma GCC diagnostic ignored "-Wvla"

struct Edge{
    int from;
    int vertex;
    int cost;
    Edge(int vertex_, int cost_) {vertex = vertex_; cost = cost_;}
    Edge(int from_, int vertex_, int cost_) {from = from_; vertex = vertex_; cost = cost_;}
    bool operator==(const Edge &other) const {return from == other.from && vertex == other.vertex && cost == other.cost;}
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

struct SourceTargetReturn{
    std::vector<int> path;
    std::vector<int> distance;
    SourceTargetReturn(std::vector<int> path_, std::vector<int> distance_): path(path_), distance(distance_){}
};

struct SourceAllReturn{
    std::vector<int> distances;
    SourceAllReturn(std::vector<int> distances_): distances(distances_){}
};

struct AllTerminalReturn{
    std::vector<std::vector<int>> distances;
    AllTerminalReturn(std::vector<std::vector<int>> distances_): distances(distances_){}
};

int n_digits(int n){
    if (n == 0) {return 1;}
    int digits = 0;
    while (n) {n /= 10; digits++;}
    return digits;
}

void print_spaced(int x, int n){
    int digits = n_digits(x);
    std::cout << x;
    for (int i = 0; i < n-digits; i++) {std::cout << " ";}
}

void printDistMatrix(std::vector<std::vector<int>> distances, int V){
    std::vector<int> max_dist(V+1, -1);
    for (int X = 0; X < V; X++) {
        max_dist[X] = std::max(max_dist[X], X);
        for (int Y = 0; Y < V; Y++) {
            max_dist[Y+1] = std::max(max_dist[Y], distances[X][Y]);   
        }
    }
    max_dist[0] = V-1;
    for (int X = 0; X < V+1; X++) {
        max_dist[X] = n_digits(max_dist[X]) + 1;
    }
    // max_dist[i] is now the maximum number of digits in the i-th column (0 is index)

    std::cout << "Distances: \n";
    for (int X = 0; X < V+1; X++) {
        if (X == 0) {std::cout << "TO "; for (int i=0; i<max_dist[0]+5-3; ++i) {std::cout << " ";}}
        else {std::cout << "FROM "; print_spaced(X-1, max_dist[0]);}
        std::cout << " ";

        for (int Y = 1; Y < V+1; Y++) {
            if (X == 0) {print_spaced(Y-1, max_dist[Y]);}
            else {print_spaced(distances[X-1][Y-1], max_dist[Y]);}
        }
        std::cout << "\n";
    }
    std::cout << "\n\n" << std::flush;
}

class Graph {
    // Directed weighted graph
    const int V;
    std::unordered_set<Edge>* adj;

public:
    int delta;
    int n_threads;
    Graph(int V, int delta, int n_threads_): V(V), delta(delta) {
        adj = new std::unordered_set<Edge>[V];
        n_threads = std::min(n_threads_, V);
    }
    
    void addEdge(int v, int w, int c) {
        if (v < V && w < V) {
            adj[v].insert(Edge(v, w, c));
        } else {
            throw "Invalid vertex";
        }
    }

    void compare_algorithms(int s, int d, bool debug = true) {
        std::vector<std::string> names_ST{"DijkstraSourceTarget", "Delta", "UNWEIGHTED BFS_SourceTarget", "UNWEIGHTED DFS_SourceTarget"};
        std::vector<SourceTargetReturn(Graph::*)(int, int)> ST_Funcs {&Graph::DijkstraSourceTarget, &Graph::parallelDeltaStepping, &Graph::BFS_ST, &Graph::DFS_ST};
        for (size_t i = 0; i < names_ST.size(); i++) {
            std::cout << "   " << names_ST[i] << ": " << std::flush;
            auto start = high_resolution_clock::now();
            SourceTargetReturn r = ((*this).*ST_Funcs[i])(s, d);
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

        // Define similar things for all terminal from the source-all terminal problem
        std::vector<std::string> names_SA{"DijkastraSourceAll", "UNWEIGHTED BFS_SourceAll", "UNWEIGHTED DFS_SourceAll"};
        std::vector<SourceAllReturn(Graph::*)(int, int)> SA_Funcs {&Graph::DijkstraSourceAll, &Graph::BFS_AT, &Graph::DFS_AT};
        for (size_t i = 0; i < names_SA.size(); i++){
            std::cout << "   " << names_SA[i] << ": " << std::flush;
            auto start = high_resolution_clock::now();
            SourceAllReturn r = ((*this).*SA_Funcs[i])(s, d);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << (double) duration.count()/1000 << " milliseconds. \n\n";
            if (debug) {
                std::cout << "Distances: \n";
                for (int v = 0; v < V; v++) {std::cout << v << ": " << r.distances[v] << "\n";}
                std::cout << "\n\n";
            }
            std::cout << std::flush;
        }

        // Generate All-Terminal from Source-All-Terminal (SEQUENTIAL)
        for (size_t i = 0; i < names_SA.size(); i++){
            std::cout << "   Sequential All-Terminal " << names_SA[i] << ": " << std::flush;
            auto start = high_resolution_clock::now();
            AllTerminalReturn r = SourceAll_To_AllTerminal(SA_Funcs[i]);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << (double) duration.count()/1000 << " milliseconds. \n\n";
            if (debug) {
                printDistMatrix(r.distances, V);
            }
            std::cout << std::flush;
        }

        // Generate All-Terminal from Source-All-Terminal (PARALLEL)
        for (size_t i = 0; i < names_SA.size(); i++){
            std::cout << "   Parallel All-Terminal " << names_SA[i] << ": " << std::flush;
            auto start = high_resolution_clock::now();
            AllTerminalReturn r = SourceAll_To_AllTerminalParallel(SA_Funcs[i]);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << (double) duration.count()/1000 << " milliseconds. \n\n";
            if (debug) {
                printDistMatrix(r.distances, V);
            }
            std::cout << std::flush;
        }

        // Do All-Terminal algorithms
        std::vector<std::string> names_AT{"Floyd_Warshall_Sequential", "Floyd_Warshall_Parallel"};
        std::vector<AllTerminalReturn(Graph::*)()> AT_Funcs {&Graph::Floyd_Warshall_Sequential, &Graph::Floyd_Warshall_Parallel};
        for (size_t i = 0; i < names_AT.size(); i++){
            std::cout << "   " << names_AT[i] << ": " << std::flush;
            auto start = high_resolution_clock::now();
            AllTerminalReturn r = ((*this).*AT_Funcs[i])();
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            std::cout << (double) duration.count()/1000 << " milliseconds. \n\n";
            if (debug) {
                printDistMatrix(r.distances, V);
            }
            std::cout << std::flush;
        }

    }

    AllTerminalReturn SourceAll_To_AllTerminal(SourceAllReturn(Graph::*F)(int, int)){
        std::vector<std::vector<int>> distances;
        for (int i = 0; i < V; i++) {
            SourceAllReturn r = ((*this).*F)(i, i); // could do i, -1 too
            distances.push_back(r.distances);
        }
        return AllTerminalReturn(distances); 
    }

    AllTerminalReturn SourceAll_To_AllTerminalParallel(SourceAllReturn(Graph::*F)(int, int)){
        std::vector<std::vector<int>> distances(V);
        std::vector<std::thread> threads(n_threads);
        int block_size = V/n_threads;
        for (int i = 0; i < n_threads-1; i++) {
            threads[i] = std::thread([this, &distances, F, i, block_size]() {
                for (int it = i*block_size; it < (i+1)*block_size; it++) {
                    SourceAllReturn r = ((*this).*F)(it, it); // could do it, -1 too
                    distances[it] = r.distances;
                }
            });
        }
        threads[n_threads-1] = std::thread([this, &distances, F, block_size]() {
            for (int it = (n_threads-1)*block_size; it < V; it++) {
                SourceAllReturn r = ((*this).*F)(it, it); // could do it, -1 too
                distances[it] = r.distances;
            }
        });
        for (int i = 0; i < V; i++) {threads[i].join();}
        return AllTerminalReturn(distances); 
    }

    SourceTargetReturn BFS_ST(int s, int d){return FS(s, d, false, true);}
    SourceAllReturn BFS_AT(int s, int d){return SourceAllReturn(FS(s, d, true, true).distance);}
    SourceTargetReturn DFS_ST(int s, int d){return FS(s, d, false, false);}
    SourceAllReturn DFS_AT(int s, int d){return SourceAllReturn(FS(s, d, true, false).distance);}

    // FS implementation (BFS, DFS, all_targets or not)
    SourceTargetReturn FS(int s, int d, bool all_targets, bool BFS){
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

        return SourceTargetReturn(path, dist);
    }

    SourceAllReturn DijkstraSourceAll(int s, int d){return SourceAllReturn(Dijkstra(s, d, true).distance);}
    SourceTargetReturn DijkstraSourceTarget(int s, int d){return Dijkstra(s, d, false);}

    // Dijkstra implementation for (positively) weighted graphs
    SourceTargetReturn Dijkstra(int s, int d, bool all_targets) {
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

        return SourceTargetReturn(path, dist);
    }

    AllTerminalReturn Floyd_Warshall_Sequential(){
        std::vector<std::vector<int>> dist(V, std::vector<int>(V, INT_MAX));

        // Initialize the distance of points to themselves to 0
        for (int i = 0; i < V; i++) {dist[i][i] = 0;}

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
                    if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]){
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        return AllTerminalReturn(dist);
    }

    AllTerminalReturn Floyd_Warshall_Parallel(){
        std::vector<std::vector<int>> dist(V, std::vector<int>(V, INT_MAX));

        // Initialize the distance of points to themselves to 0
        for (int i = 0; i < V; i++) {dist[i][i] = 0;}

        // Initialize the distance of points to their neighbors
        for (int i = 0; i < V; i++) {
            for (const Edge e : adj[i]) {
                dist[i][e.vertex] = e.cost;
            }
        }

        // Initialize n_threads barriers for synchronization
        std::barrier barrier(V);
        std::vector<std::thread> threads(n_threads);
        int block_size = V/n_threads;

        for (int block = 0; block < n_threads-1; block++) {
            threads[block] = std::thread([this, &dist, block, block_size, &barrier]() {
                for (int k = 0; k < V; k++) {
                    for (int i = block*block_size; i < (block+1)*block_size; i++) {
                        if (i == k) {continue;}
                        for (int j = 0; j < V; j++) {
                            if (j == k || j == i) {continue;}
                            if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]){
                                dist[i][j] = dist[i][k] + dist[k][j];
                            }
                        }
                    }
                    barrier.arrive_and_wait();
                }
            });
        }
        // Last block
        threads[n_threads-1] = std::thread([this, &dist, block_size, &barrier]() {
            for (int k = 0; k < V; k++) {
                for (int i = (n_threads-1)*block_size; i < V; i++) {
                    if (i == k) {continue;}
                    for (int j = 0; j < V; j++) {
                        if (j == k || j == i) {continue;}
                        if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]){
                            dist[i][j] = dist[i][k] + dist[k][j];
                        }
                    }
                }
                barrier.arrive_and_wait();
            }
        });

        // Join threads
        for (int i = 0; i < n_threads; i++) {threads[i].join();}
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

        while(!buckets.empty()) {
            int i=0;
            while(buckets.find(i)==buckets.end()) {
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
                            if(bucket_index == i){
                                bucket.push_back(e.vertex);//should remove it from old bucket as well...
                            }
                            else if(buckets.find(bucket_index) == buckets.end()) {
                                buckets[bucket_index] = std::list<int>({e.vertex});
                            } else {
                                buckets[bucket_index].push_back(e.vertex);
                            }
                        }
                    }else{
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
                    if(buckets.find(bucket_index) == buckets.end()) {
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
        for (size_t i = 0; i < rpath.size(); ++i) {path.push_back(rpath[rpath.size() - i - 1]);}
        return SourceTargetReturn(path, dist);
    }

    ////// parallel delta-stepping

    // helper functions
    void relaxThread(std::unordered_map<int, std::list<int>> &buckets,
    std::vector<int> &dist, 
    std::vector<int> &prev,
    Edge* start_edge,
    Edge* end_edge){
        while(start_edge!=end_edge) {
            Edge e = *start_edge;
            int new_dist = dist[e.from] + e.cost;
            if (new_dist < dist[e.vertex]) {
                dist[e.vertex] = new_dist;
                prev[e.vertex] = e.from;
                int bucket_index = new_dist / delta;
                if(buckets.find(bucket_index) == buckets.end()) {
                    buckets[bucket_index] = std::list<int>({e.vertex});
                } else {
                    buckets[bucket_index].push_back(e.vertex);
                }
            }
            start_edge++;
        }
    }
    // TODOs:
    // 1. erase the edge from the bucket it was in
    // 2. think about compatibility of the data structures we use (for buckets, dist and prev) with parallelism
    // 3. adapt for light edges
    void parallelRelax(std::unordered_map<int, std::list<int>> &buckets,
    std::vector<int> &dist,
    std::vector<int> &prev,
    std::vector<Edge> &edges,
    int n_threads){
        std::vector<std::thread> threads(n_threads);
        int block_size = edges.size()/n_threads;
        Edge* start_edge = &edges[0];
        for (int i = 0; i < n_threads-1; i++) {
            threads[i] = std::thread(&Graph::relaxThread, this, std::ref(buckets), std::ref(dist), std::ref(prev), start_edge, start_edge+block_size);
            start_edge += block_size;
        }
        threads[n_threads-1] = std::thread(&Graph::relaxThread, this, std::ref(buckets), std::ref(dist), std::ref(prev), start_edge, &edges[edges.size()]);
        for (int i = 0; i < n_threads; i++) {threads[i].join();}
    }


    SourceTargetReturn parallelDeltaStepping(int source, int destination) {
        std::vector<int> dist(this->V, std::numeric_limits<int>::max());
        std::vector<int> prev(this->V, -1);
        std::unordered_map<int, std::list<int>> buckets;
        

        dist[source] = 0;
        buckets[0].push_back(source);

        while(!buckets.empty()) {
            int i=0;
            while(buckets.find(i)==buckets.end()) {
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
                            if(bucket_index == i){
                                bucket.push_back(e.vertex);//should remove it from old bucket as well...
                            }
                            else if(buckets.find(bucket_index) == buckets.end()) {
                                buckets[bucket_index] = std::list<int>({e.vertex});
                            } else {
                                buckets[bucket_index].push_back(e.vertex);
                            }
                        }
                    }else{
                        heavy_edges.push_back(e);
                    }
                }
            }
            // Process heavy edges
            parallelRelax(buckets, dist, prev, heavy_edges, true);
        }

        // Make the return struct
        std::vector<int> rpath;
        for (int at = destination; at != -1; at = prev[at]) {
            rpath.push_back(at);
        }
        std::vector<int> path;
        for (size_t i = 0; i < rpath.size(); ++i) {path.push_back(rpath[rpath.size() - i - 1]);}
        return SourceTargetReturn(path, dist);
    }
};



int main() {
    Graph g(7, 2, 32);
    g.addEdge(0, 1, 1);
    g.addEdge(0, 2, 1);
    g.addEdge(1, 3, 2);
    g.addEdge(2, 3, 1);
    g.addEdge(3, 4, 1);
    g.addEdge(4, 5, 1);
    g.addEdge(5, 6, 3);

    g.compare_algorithms(0, 5);
    return 0;
}