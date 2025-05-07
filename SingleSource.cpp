
#include "SingleSource.hpp"

#include <limits>
#include <queue>

void SingleSource::addEdge(const Node& node) {
    adjList[node.source].emplace_back(node.dest, node.weight);
    if (this->biDirectional) {
        adjList[node.dest].emplace_back(node.source, node.weight);
    }
}

void SingleSource::buildAdjacencyList(bool biDirectional) {
    Nodes nodes = Loader::load(BELLMANFORD_DATASET_FILENAME);
    this->biDirectional = biDirectional;

    for (auto node: nodes) {
        addEdge(node);
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


void SingleSource::bellmanFord(int source) {
    std::unordered_map<int, int> distance;
    for (const std::pair<const int, std::vector<std::pair<int, int>>>& node : adjList) {
        distance[node.first] = std::numeric_limits<int>::max();
    }
    distance[source] = 0;

    int V = adjList.size();
    for (int i = 0; i < V - 1; ++i) {
        for (const std::pair<const int, std::vector<std::pair<int, int>>>& node : adjList) {
            int u = node.first;
            const std::vector<std::pair<int, int>>& neighbors = node.second;
            for (size_t j = 0; j < neighbors.size(); ++j) {
                int v = neighbors[j].first;
                int weight = neighbors[j].second;
                if (distance[u] != std::numeric_limits<int>::max() &&
                    distance[u] + weight < distance[v]) {
                    distance[v] = distance[u] + weight;
                }
            }
        }
    }

    // Check for negative-weight cycles
    for (const std::pair<const int, std::vector<std::pair<int, int>>>& node : adjList) {
        int u = node.first;
        const std::vector<std::pair<int, int>>& neighbors = node.second;
        for (size_t j = 0; j < neighbors.size(); ++j) {
            int v = neighbors[j].first;
            int weight = neighbors[j].second;
            if (distance[u] != std::numeric_limits<int>::max() &&
                distance[u] + weight < distance[v]) {
                std::cerr << "Graph contains negative weight cycle\n";
                return;
            }
        }
    }

    std::cout << "Bellman-Ford distances from node " << source << ":\n";
    for (const std::pair<const int, int>& pair : distance) {
        std::cout << "Node " << pair.first << " -> Distance: " << pair.second << '\n';
    }
}

// Calculates the average degree
double SingleSource::averageDegree() const {
    int edgeCount = 0;
    int nodeCount = adjList.size();

    for (const auto& pair : adjList) {
        edgeCount += pair.second.size();
    }

    // For undirected graph, divide edgeCount by 2 to avoid double counting
    double avgDegree = this->biDirectional? static_cast<double>(edgeCount) / nodeCount : edgeCount; 
    return avgDegree;
}

// Calculates the graph diameter using Dijkstra (weighted) from every node
double SingleSource::diameter() const {
    int maxDistance = 0;

    for (const auto& [start, _] : adjList) {
        std::unordered_map<int, int> distance;
        for (const auto& [node, _] : adjList) {
            distance[node] = std::numeric_limits<int>::max();
        }
        distance[start] = 0;

        using pii = std::pair<int, int>; // (distance, node)
        std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;
        pq.push({0, start});

        while (!pq.empty()) {
            auto [dist, node] = pq.top();
            pq.pop();

            if (dist > distance[node]) continue;

            for (const auto& [neighbor, weight] : adjList.at(node)) {
                if (distance[node] + weight < distance[neighbor]) {
                    distance[neighbor] = distance[node] + weight;
                    pq.push({distance[neighbor], neighbor});
                }
            }
        }

        for (const auto& [node, dist] : distance) {
            if (dist != std::numeric_limits<int>::max()) {
                maxDistance = std::max(maxDistance, dist);
            }
        }
    }

    return maxDistance;
}


int main() {
    SingleSource ss;

    ss.buildAdjacencyList(true);

    int sourceNode;

    std::cout << "Source Node for Dijkstra: ";
    std::cin >> sourceNode;
    ss.dijkstra(sourceNode);

    std::cout << "Source Node for Bellman Ford: ";
    std::cin >> sourceNode;
    ss.bellmanFord(sourceNode);
    
    std::cout << "Average Degree => " << ss.averageDegree() << '\n';
    std::cout << "Diameter of the Graph => " << ss.diameter() << '\n';

    return 0;
}
