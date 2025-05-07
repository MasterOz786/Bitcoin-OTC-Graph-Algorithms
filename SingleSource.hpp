
#include "Loader.hpp"

using EdgeWeight = std::pair<int, int>;
using Neighbours = std::vector<std::pair<int, int>>;
using AdjacencyList = std::unordered_map<int, std::vector<EdgeWeight>>;

class SingleSource {
    AdjacencyList adjList;
    bool biDirectional = false;

public:
    void addEdge(const Node&);
    void printAdjList() const;
    void buildAdjacencyList(bool);
    const AdjacencyList& getAdjacencyList() const;
    void dijkstra(int);
    void bellmanFord(int);

    double averageDegree() const;
    double diameter() const;
};
