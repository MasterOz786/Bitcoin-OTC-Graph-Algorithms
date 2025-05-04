
#include "SingleSource.hpp"

void SingleSource::addEdge(const Node& node, bool bidirectional) {
    adjList[node.source].emplace_back(node.dest, node.weight);
    if (bidirectional) {
        adjList[node.dest].emplace_back(node.source, node.weight);
    }
}

void SingleSource::buildAdjacencyList() {
    Nodes nodes = Loader::load(DATASET_FILENAME);

    for (auto node: nodes) {
        addEdge(node, true);
    }
    printAdjList();
}

// Print the adjacency list
void SingleSource::printAdjList() const {
    for (const std::pair<const int, std::vector<EdgeWeight>>& node : adjList) {
        std::cout << node.first << " -> ";
        const Neighbours& neighbours = node.second;
        
        for (int i = 0; i < neighbours.size(); i++) {
            int dest = neighbours[i].first;
            int weight = neighbours[i].second;
            std::cout << "(" << dest << ", " << weight << ") ";
        }
    }
}

// Getter for adjacency list (if needed externally)
const AdjacencyList& SingleSource::getAdjacencyList() const {
    return adjList;
}

int main() {
    SingleSource ss;
    ss.buildAdjacencyList();

    return 0;
}