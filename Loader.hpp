
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#define DATASET_FILENAME "soc-sign-bitcoinotc.csv"
#define SHORT_DATASET_FILENAME "soc.csv"

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