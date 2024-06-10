# CSE305 - Parallel and Distributed Computing Project
## Parallel Shortest Paths

Finding shortest distances in a graph is one the fundamental problems in computer science with
numerous applications (route planning by CityMapper and Google/Yandex-maps, for example).
You have probably learned classical algorithms used for this task in your algorithms course (BFS,
Dijkstra, etc). However, these algorithms are inherently sequential and hard to parallelize (although
parallele versions exist). In this project, you will be asked to implement and benchmark one of the
most standard shortest path algorithms, $\Delta$-stepping algorithm.

[Original Delta Stepping Paper](https://www.sciencedirect.com/science/article/pii/S0196677403000762?via%3Dihub)

[Recent Developments](https://ieeexplore.ieee.org/abstract/document/9006237)

The file [data](./data/) contains the different runtimes obtained over multiple runs for different graphs of different sizes. Here are the values we studied:

Parameter | Values
--- | ---
Graph Size | 200, 500, 1000, 2500, 7500, 10000
Graph Density | 0.3, 0.5, 0.7
N Threads | 1, 2, 4, 8, 12, 16, 20, 24, 25
Delta | 1, 2, 4, 6, 8

