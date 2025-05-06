#include "Loader.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
using namespace std;

#define MAX_NODES 100000
#define MAX_EDGES_PER_NODE 1000

struct Edge {
    int dest;
    int weight;
    Edge* next;
};

struct Graph {
    Edge* adj[MAX_NODES];
    int nodeCount;
};

void initGraph(Graph* g) {
    for (int i = 0; i < MAX_NODES; i++) {
        g->adj[i] = nullptr;
    }
    g->nodeCount = 0;
}

void addEdge(Graph* g, int src, int dest, int weight) {
    Edge* e = new Edge;
    e->dest = dest;
    e->weight = weight;
    e->next = g->adj[src];
    g->adj[src] = e;
}

void buildGraph(Nodes& data, Graph* g, int* nodes, int* nodeCount) {
    initGraph(g);
    int maxNode = 0;
    for (unsigned int i = 0; i < data.size(); i++) {
        if (data[i].source > maxNode) maxNode = data[i].source;
        if (data[i].dest > maxNode) maxNode = data[i].dest;
        nodes[data[i].source] = 1;
        nodes[data[i].dest] = 1;
    }
    for (int i = 0; i <= maxNode; i++) {
        if (nodes[i]) (*nodeCount)++;
    }
    for (unsigned int i = 0; i < data.size(); i++) {
        addEdge(g, data[i].source, data[i].dest, data[i].weight);
    }
    g->nodeCount = *nodeCount;
}

int dfsCycle(Graph* g, int node, int* visited, int* recStack, ofstream& traceFile, int& cycleFound) {
    visited[node] = 1;
    recStack[node] = 1;
    traceFile << "Enter: " << node << "\n";

    for (Edge* e = g->adj[node]; e; e = e->next) {
        if (!visited[e->dest]) {
            if (dfsCycle(g, e->dest, visited, recStack, traceFile, cycleFound)) {
                return 1;
            }
        } else if (recStack[e->dest]) {
            cycleFound = 1;
            traceFile << "Cycle detected at node " << e->dest << "\n";
            return 1;
        }
    }

    recStack[node] = 0;
    traceFile << "Exit: " << node << "\n";
    return 0;
}

void detectCycle(Graph* g, int nodeCount, const char* rollNumbers) {
    char resultFileName[100] = "Cycle_Result_";
    char traceFileName[100] = "Cycle_Trace_";
    char timeFileName[100] = "Cycle_ExecutionTime_";
    strcat(resultFileName, rollNumbers);
    strcat(traceFileName, rollNumbers);
    strcat(timeFileName, rollNumbers);
    strcat(resultFileName, ".txt");
    strcat(traceFileName, ".txt");
    strcat(timeFileName, ".txt");

    ofstream resultFile(resultFileName);
    ofstream traceFile(traceFileName);
    ofstream timeFile(timeFileName);

    // Validate node count
    if (nodeCount < 1000) {
        cout << "Error: Graph has fewer than 1000 nodes (" << nodeCount << ").\n";
        resultFile << "Error: Graph has fewer than 1000 nodes\n";
        traceFile << "Error: Graph has fewer than 1000 nodes\n";
        resultFile.close();
        traceFile.close();
        timeFile.close();
        return;
    }

    auto start = chrono::high_resolution_clock::now();

    int visited[MAX_NODES] = {0};
    int recStack[MAX_NODES] = {0};
    int cycleFound = 0;

    for (int i = 0; i < MAX_NODES; i++) {
        if (g->adj[i] && !visited[i]) {
            dfsCycle(g, i, visited, recStack, traceFile, cycleFound);
            if (cycleFound) break;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Write result
    if (cycleFound) {
        resultFile << "Cycle exists in the graph\n";
        cout << "Cycle exists in the graph\n";
    } else {
        resultFile << "No cycle exists in the graph\n";
        cout << "No cycle exists in the graph\n";
    }

    // Write execution time
    timeFile << "Execution time: " << duration << " microseconds\n";
    cout << "Cycle detection execution time: " << duration << " microseconds\n";

    resultFile.close();
    traceFile.close();
    timeFile.close();
}

int main() {
    Nodes data = Loader::load(DATASET_FILENAME);
    if (data.size() == 0) {
        cout << "Error: Failed to load dataset or dataset is empty.\n";
        return 1;
    }

    Graph g;
    int nodes[MAX_NODES] = {0};
    int nodeCount = 0;
    buildGraph(data, &g, nodes, &nodeCount);

    detectCycle(&g, nodeCount, "200123_205478_213254");

    // Clean up
    for (int i = 0; i < MAX_NODES; i++) {
        Edge* e = g.adj[i];
        while (e) {
            Edge* temp = e;
            e = e->next;
            delete temp;
        }
    }

    return 0;
}