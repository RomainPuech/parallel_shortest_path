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

