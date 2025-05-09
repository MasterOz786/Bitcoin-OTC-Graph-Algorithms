
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#define DATASET_FILENAME "soc-sign-bitcoinotc.csv"
#define DIJKSTRA_DATASET_FILENAME "truncated-for-dijkstra.csv"
#define BELLMANFORD_DATASET_FILENAME "truncated-for-bellmanFord.csv"

struct Node {
    int source; // from which user
    int dest; // to which user
    int weight; // rating assigned with the destination user
    long long int timestamp; // when was the transaction performed
};

using Nodes = std::vector<Node>;

class Loader {
public:
    Loader(bool bi = false) { }
    static Nodes load(const char*);
};