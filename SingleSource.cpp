
#include "SingleSource.hpp"

#include <limits>
#include <queue>

void SingleSource::addEdge(const Node& node, bool bidirectional) {
    adjList[node.source].emplace_back(node.dest, node.weight);
    if (bidirectional) {
        adjList[node.dest].emplace_back(node.source, node.weight);
    }
}

void SingleSource::buildAdjacencyList() {
    Nodes nodes = Loader::load(SHORT_DATASET_FILENAME);

    for (auto node: nodes) {
        addEdge(node, true);
    }
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

void SingleSource::dijkstra(int source) {
    std::unordered_map<int, int> distance;
    for (const std::pair<const int, std::vector<std::pair<int, int>>>& node : adjList) {
        distance[node.first] = std::numeric_limits<int>::max();
    }
    distance[source] = 0;

    using pii = std::pair<int, int>; // distance, node
    std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;
    pq.push(std::make_pair(0, source));

    while (!pq.empty()) {
        pii current = pq.top();
        pq.pop();
        int dist = current.first;
        int node = current.second;

        if (dist > distance[node]) continue;

        const std::vector<std::pair<int, int>>& neighbors = adjList.at(node);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int neighbor = neighbors[i].first;
            int weight = neighbors[i].second;
            if (distance[node] + weight < distance[neighbor]) {
                distance[neighbor] = distance[node] + weight;
                pq.push(std::make_pair(distance[neighbor], neighbor));
            }
        }
    }

    std::cout << "Dijkstra distances from node " << source << ":\n";
    for (const std::pair<const int, int>& pair : distance) {
        std::cout << "Node " << pair.first << " -> Distance: " << pair.second << '\n';
    }
}


int main() {
    SingleSource ss;
    ss.buildAdjacencyList();
    ss.dijkstra(6);
    return 0;
}