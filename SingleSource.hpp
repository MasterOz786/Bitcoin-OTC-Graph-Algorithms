
#include "Loader.hpp"

using EdgeWeight = std::pair<int, int>;
using Neighbours = std::vector<std::pair<int, int>>;
using AdjacencyList = std::unordered_map<int, std::vector<EdgeWeight>>;

class SingleSource {
    AdjacencyList adjList;

public:
    void addEdge(const Node&, bool);
    void printAdjList() const;
    void buildAdjacencyList();
    const AdjacencyList& getAdjacencyList() const;
};
