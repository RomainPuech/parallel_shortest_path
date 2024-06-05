/*

TODO: Split the Graph Class



*/

int main() {
  return 1;
}

// #include <vector>
// #include <thread>
// #include "utils.hpp"
// #include <iostream>

//   AllTerminalReturn Floyd_Warshall_Parallel() {
//     std::vector<std::vector<int>> dist(V, std::vector<int>(V, INT_MAX));

//     // Initialize the distance of points to themselves to 0
//     for (int i = 0; i < V; i++) {
//       dist[i][i] = 0;
//     }

//     // Initialize the distance of points to their neighbors
//     for (int i = 0; i < V; i++) {
//       for (const Edge e : adj[i]) {
//         dist[i][e.vertex] = e.cost;
//       }
//     }

//     // Initialize n_threads barriers for synchronization
//     std::barrier barrier(n_threads);
//     std::vector<std::thread> threads(n_threads - 1);
//     int block_size = V / n_threads;

//     for (int block = 0; block < n_threads - 1; block++) {
//       threads[block] = std::thread([this, &dist, block, block_size, &barrier]() {
//         for (int k = 0; k < V; k++) {
//           for (int i = block * block_size; i < (block + 1) * block_size; i++) {
//             if (i == k) {
//               continue;
//             }
//             for (int j = 0; j < V; j++) {
//               if (j == k || j == i) {
//                 continue;
//               }
//               if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]) {
//                 dist[i][j] = dist[i][k] + dist[k][j];
//               }
//             }
//           }
//           barrier.arrive_and_wait();
//         }
//       });
//     }
//     // Last block
//     for (int k = 0; k < V; k++) {
//       for (int i = (n_threads - 1) * block_size; i < V; i++) {
//         if (i == k) {
//           continue;
//         }
//         for (int j = 0; j < V; j++) {
//           if (j == k || j == i) {
//             continue;
//           }
//           if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX && dist[i][j] > dist[i][k] + dist[k][j]) {
//             dist[i][j] = dist[i][k] + dist[k][j];
//           }
//         }
//       }
//       barrier.arrive_and_wait();
//     }

//     // Join threads
//     for (int i = 0; i < n_threads - 1; i++) {
//       threads[i].join();
//     }

//     // Check the diagonal
//     for (int i = 0; i < V; i++) {
//       if (dist[i][i] < 0) {
//         throw "Negative cycle detected";
//       }
//     }
//     return AllTerminalReturn(dist);
//   }