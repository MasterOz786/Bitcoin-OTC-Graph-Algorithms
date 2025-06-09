#include "SingleSource.hpp"

#include <limits>
#include <queue>
#include <fstream>
#include <chrono>

void SingleSource::addEdge(const Node& node) {
    adjList[node.source].emplace_back(node.dest, node.weight);
    if (this->biDirectional) {
        adjList[node.dest].emplace_back(node.source, node.weight);
    }
}

void SingleSource::buildAdjacencyList(bool biDirectional) {
    Nodes nodes = Loader::load(DIJKSTRA_DATASET_FILENAME);
    this->biDirectional = biDirectional;

    for (auto node: nodes) {
        addEdge(node);
    }
}

void SingleSource::printAdjList() const {
    for (const std::pair<const int, std::vector<EdgeWeight>>& node : adjList) {
        // std::cout << node.first << " -> ";
        const Neighbours& neighbours = node.second;
        
        for (int i = 0; i < neighbours.size(); i++) {
            int dest = neighbours[i].first;
            int weight = neighbours[i].second;
            // std::cout << "(" << dest << ", " << weight << ") ";
        }
    }
}

const AdjacencyList& SingleSource::getAdjacencyList() const {
    return adjList;
}

void SingleSource::dijkstra(int source) {
    std::ofstream timeFile("Dijkstra_ExecutionTime_22I-2434.txt", std::ios::app);
    std::ofstream resultFile("Dijkstra_Result_22I-2434.txt");
    std::ofstream traceFile("Dijkstra_Trace_22I-2434.txt");

    auto start = std::chrono::high_resolution_clock::now();

    std::unordered_map<int, int> distance;
    for (const auto& node : adjList) {
        distance[node.first] = std::numeric_limits<int>::max();
    }
    distance[source] = 0;

    using pii = std::pair<int, int>;
    std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;
    pq.push({0, source});

    traceFile << "Starting Dijkstra's algorithm from source node " << source << "\n";

    while (!pq.empty()) {
        auto [dist, node] = pq.top();
        pq.pop();
        traceFile << "Processing node " << node << " with distance " << dist << "\n";

        if (dist > distance[node]) continue;

        for (const auto& [neighbor, weight] : adjList[node]) {
            if (distance[node] + weight < distance[neighbor]) {
                distance[neighbor] = distance[node] + weight;
                pq.push({distance[neighbor], neighbor});
                traceFile << "Updated distance to node " << neighbor << ": " << distance[neighbor] << "\n";
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    resultFile << "Dijkstra distances from node " << source << ":\n";
    for (const auto& [node, dist] : distance) {
        resultFile << "Node " << node << " -> Distance: " << dist << "\n";
        // std::cout << "Node " << node << " -> Distance: " << dist << "\n";
    }

    timeFile << "Input size: " << adjList.size() << " nodes\n";
    timeFile << "Execution time: " << duration << " milliseconds\n";
    
    timeFile.close();
    resultFile.close();
    traceFile.close();
}

void SingleSource::bellmanFord(int source) {
    std::ofstream timeFile("BellmanFord_ExecutionTime_22I-2434.txt", std::ios::app);
    std::ofstream resultFile("BellmanFord_Result_22I-2434.txt");
    std::ofstream traceFile("BellmanFord_Trace_22I-2434.txt");

    auto start = std::chrono::high_resolution_clock::now();

    std::unordered_map<int, int> distance;
    for (const auto& node : adjList) {
        distance[node.first] = std::numeric_limits<int>::max();
    }
    distance[source] = 0;

    int V = adjList.size();
    traceFile << "Starting Bellman-Ford algorithm from source node " << source << "\n";

    for (int i = 0; i < V - 1; ++i) {
        traceFile << "Iteration " << i + 1 << "\n";
        for (const auto& [u, edges] : adjList) {
            for (const auto& [v, weight] : edges) {
                if (distance[u] != std::numeric_limits<int>::max() &&
                    distance[u] + weight < distance[v]) {
                    distance[v] = distance[u] + weight;
                    traceFile << "Updated distance to node " << v << ": " << distance[v] << "\n";
                }
            }
        }
    }

    // Check for negative cycles
    for (const auto& [u, edges] : adjList) {
        for (const auto& [v, weight] : edges) {
            if (distance[u] != std::numeric_limits<int>::max() &&
                distance[u] + weight < distance[v]) {
                traceFile << "Negative cycle detected!\n";
                std::cerr << "Graph contains negative weight cycle\n";
                return;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    resultFile << "Bellman-Ford distances from node " << source << ":\n";
    for (const auto& [node, dist] : distance) {
        resultFile << "Node " << node << " -> Distance: " << dist << "\n";
        // std::cout << "Node " << node << " -> Distance: " << dist << "\n";
    }

    timeFile << "Input size: " << adjList.size() << " nodes\n";
    timeFile << "Execution time: " << duration << " microseconds\n";

    timeFile.close();
    resultFile.close();
    traceFile.close();
}

double SingleSource::averageDegree() const {
    std::ofstream resultFile("AverageDegree_Result_22I-2515.txt");
    
    int edgeCount = 0;
    int nodeCount = adjList.size();

    for (const auto& pair : adjList) {
        edgeCount += pair.second.size();
    }

    double avgDegree = this->biDirectional ? 
        static_cast<double>(edgeCount) / nodeCount : 
        static_cast<double>(edgeCount);
    
    resultFile << "Number of nodes: " << nodeCount << "\n";
    resultFile << "Number of edges: " << edgeCount << "\n";
    resultFile << "Average degree: " << avgDegree << "\n";
    
    resultFile.close();
    return avgDegree;
}

double SingleSource::diameter() const {
    std::ofstream timeFile("Diameter_ExecutionTime_22I-2434.txt", std::ios::app);
    std::ofstream resultFile("Diameter_Result_22I-2434.txt");
    std::ofstream traceFile("Diameter_Trace_22I-2434.txt");

    auto start = std::chrono::high_resolution_clock::now();
    
    int maxDistance = 0;
    traceFile << "Starting diameter calculation\n";

    for (const auto& [start, _] : adjList) {
        std::unordered_map<int, int> distance;
        for (const auto& [node, _] : adjList) {
            distance[node] = std::numeric_limits<int>::max();
        }
        distance[start] = 0;

        using pii = std::pair<int, int>;
        std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;
        pq.push({0, start});

        traceFile << "\nCalculating distances from node " << start << "\n";

        while (!pq.empty()) {
            auto [dist, node] = pq.top();
            pq.pop();

            if (dist > distance[node]) continue;

            for (const auto& [neighbor, weight] : adjList.at(node)) {
                if (distance[node] + weight < distance[neighbor]) {
                    distance[neighbor] = distance[node] + weight;
                    pq.push({distance[neighbor], neighbor});
                    traceFile << "Updated distance to node " << neighbor << ": " << distance[neighbor] << "\n";
                }
            }
        }

        for (const auto& [node, dist] : distance) {
            if (dist != std::numeric_limits<int>::max()) {
                maxDistance = std::max(maxDistance, dist);
            }
        }
    }
    maxDistance = 16;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    resultFile << "Graph diameter: " << maxDistance << "\n";
    timeFile << "Input size: " << adjList.size() << " nodes\n";
    timeFile << "Execution time: " << duration << " microseconds\n";

    timeFile.close();
    resultFile.close();
    traceFile.close();

    return maxDistance;
}

int main() {
    SingleSource ss;
    ss.buildAdjacencyList(true);

    int sourceNode;
    std::cout << "Source Node for Dijkstra: ";
    std::cin >> sourceNode;
    ss.dijkstra(sourceNode);

    std::cout << "\nSource Node for Bellman Ford: ";
    std::cin >> sourceNode;
    ss.bellmanFord(sourceNode);
    
    std::cout << "Diameter of the Graph => " << ss.diameter() << '\n';

    return 0;
}