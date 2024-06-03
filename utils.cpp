/*
File for utils

Incldues:

---- DATA STRUCTURES ----
struct perishable_pointer

class ll_collection


---- FUCNTIONS ----

*/

#include <limits>
#include <list>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <mutex>
#include <vector>
#include <random>
#include <thread>

#include "utils.hpp"


template <typename T>
ll_collection<T>::ll_collection(int V_, int p_): V(V_), p(p_) {
  data = std::vector<std::list<T>>(p);
  perishable_pointers = std::vector<perishable_pointer<T>>(V, perishable_pointer<T>()); // might be long? but initialize it only once and change tag
  p_locks = std::vector<std::mutex>(p);
  V_locks = std::vector<std::mutex>(V); 
  counter = 0;
}

template <typename T> 
bool ll_collection<T>::replace_if_better(T element, int index, int current_tag, std::vector<int> &dist) {
  if ( (perishable_pointers[index].tag != current_tag) || (  (perishable_pointers[index].ptr)->cost + dist[(perishable_pointers[index].ptr)->from] < dist[(perishable_pointers[index].ptr)->vertex]    ) ){
    // take a lock.
    // no need to lock distance as it is not modified in this phase
    V_locks[index].lock();
    if ( (perishable_pointers[index].tag != current_tag) || (  (perishable_pointers[index].ptr)->cost + dist[(perishable_pointers[index].ptr)->from] < dist[(perishable_pointers[index].ptr)->vertex]    ) ){
      int ll_index = counter++;
      p_locks[ll_index % p].lock();
      data[ll_index % p].push_back(element); // TODO: define non-blocking pushback, that returns pointer to it
      perishable_pointers[index] = perishable_pointer(&data[ll_index % p].back(), current_tag);
      p_locks[ll_index % p].unlock();
      V_locks[index].unlock();
      return true;
    }
  }
  return false;
}

template <typename T> 
bool ll_collection<T>::contains(int index,int current_tag) {
  return perishable_pointers[index].tag == current_tag;
}

template <typename T> 
void ll_collection<T>::reset(){
  // delete all content of linkedlists
  for (int i = 0; i < p; i++){
    data[i].clear();
  }
}

template <typename T> 
void ll_collection<T>::display(){
  for(int i=0;i<p;i++){
    std::cout << "List " << i << ": ";
    for (T element : data[i]){
      std::cout << "(" << element.from << ","<< element.vertex << "," << element.cost << "),  ";
    }
    std::cout << "\n";
  }
}


int n_digits(int n) {
  if (n == 0) {
    return 1;
  } else if (n == -1) {
    return 2;
  }
  int digits = 0;
  while (n) {
    n /= 10;
    digits++;
  }
  return digits;
}


void print_spaced(int x, int n) {
  int digits = n_digits(x);
  std::cout << x;
  for (int i = 0; i < n - digits; i++) {
    std::cout << " ";
  }
}


void printDistMatrix(std::vector<std::vector<int>> distances, int V) {
  std::vector<int> max_dist(V + 1, -1);
  for (int X = 0; X < V; X++) {
    max_dist[X] = std::max(max_dist[X], X);
    for (int Y = 0; Y < V; Y++) {
      max_dist[Y + 1] = std::max(max_dist[Y + 1], distances[X][Y]);
      if (distances[X][Y] == -1) {
        max_dist[Y + 1] = std::max(max_dist[Y + 1], 10);
      }
    }
  }
  max_dist[0] = V - 1;
  for (int X = 0; X < V + 1; X++) {
    max_dist[X] = n_digits(max_dist[X]) + 1;
  }
  // max_dist[i] is now the maximum number of digits in the i-th column (0 is index)

  std::cout << "Distances: \n";
  for (int X = 0; X < V + 1; X++) {
    if (X == 0) {
      std::cout << "TO ";
      for (int i = 0; i < max_dist[0] + 5 - 3; ++i) {
        std::cout << " ";
      }
    } else {
      std::cout << "FROM ";
      print_spaced(X - 1, max_dist[0]);
    }
    std::cout << " ";

    for (int Y = 1; Y < V + 1; Y++) {
      if (X == 0) {
        print_spaced(Y - 1, max_dist[Y]);
      } else {
        print_spaced(distances[X - 1][Y - 1], max_dist[Y]);
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n\n"
            << std::flush;
}

