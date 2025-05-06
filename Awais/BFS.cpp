#include "Loader.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
using namespace std;


#define MAX_NODES 100000
#define MAX_EDGES_PER_NODE 1000
#define MAX_QUEUE_SIZE 100000

struct Edge {
    int dest;
    int weight;
    Edge* next;
};

struct Graph {
    Edge* adj[MAX_NODES];
    int nodeCount;
};

struct Queue {
    int data[MAX_QUEUE_SIZE];
    int front;
    int rear;
    int size;
};

void initQueue(Queue* q) {
    q->front = 0;
    q->rear = -1;
    q->size = 0;
}

int isQueueEmpty(Queue* q) {
    return q->size == 0;
}

void enqueue(Queue* q, int value) {
    if (q->size < MAX_QUEUE_SIZE) {
        q->rear = (q->rear + 1) % MAX_QUEUE_SIZE;
        q->data[q->rear] = value;
        q->size++;
    }
}

int dequeue(Queue* q) {
    if (isQueueEmpty(q)) return -1;
    int value = q->data[q->front];
    q->front = (q->front + 1) % MAX_QUEUE_SIZE;
    q->size--;
    return value;
}

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


void bfs(Graph* g, int source, int nodeCount, const char* rollNumbers) {
    char traversalFileName[100] = "BFS_Traversal_";
    char traceFileName[100] = "BFS_Trace_";
    char timeFileName[100] = "BFS_ExecutionTime_";
    ::strcat(traversalFileName, rollNumbers);
    ::strcat(traceFileName, rollNumbers);
    ::strcat(timeFileName, rollNumbers);
    ::strcat(traversalFileName, ".txt");
    ::strcat(traceFileName, ".txt");
    ::strcat(timeFileName, ".txt");

    ::ofstream traversalFile(traversalFileName);
    ::ofstream traceFile(traceFileName);
    ::ofstream timeFile(timeFileName);

    int visited[MAX_NODES] = {0};
    int traversal[MAX_NODES];
    int traversalSize = 0;

    // Validate source node
    int sourceExists = 0;
    for (int i = 0; i < MAX_NODES; i++) {
        if (g->adj[i]) {
            if (i == source) sourceExists = 1;
            for (Edge* e = g->adj[i]; e; e = e->next) {
                if (e->dest == source) sourceExists = 1;
            }
        }
    }
    if (!sourceExists) {
        cout << "Error: Source node " << source << " does not exist in the graph.\n";
        traversalFile << "Error: Invalid source node\n";
        traceFile << "Error: Invalid source node\n";
        traversalFile.close();
        traceFile.close();
        timeFile.close();
        return;
    }

    // Validate node count
    if (nodeCount < 1000) {
        cout << "Error: Graph has fewer than 1000 nodes (" << nodeCount << ").\n";
        traversalFile << "Error: Graph has fewer than 1000 nodes\n";
        traceFile << "Error: Graph has fewer than 1000 nodes\n";
        traversalFile.close();
        traceFile.close();
        timeFile.close();
        return;
    }

    auto start = ::chrono::high_resolution_clock::now();

    Queue q;
    initQueue(&q);
    enqueue(&q, source);
    visited[source] = 1;
    traceFile << "Enqueue: " << source << "\n";

    while (!isQueueEmpty(&q)) {
        int current = dequeue(&q);
        traceFile << "Dequeue: " << current << "\n";
        traversal[traversalSize++] = current;

        for (Edge* e = g->adj[current]; e; e = e->next) {
            if (!visited[e->dest]) {
                enqueue(&q, e->dest);
                visited[e->dest] = 1;
                traceFile << "Enqueue: " << e->dest << "\n";
            }
        }
    }

    auto end = ::chrono::high_resolution_clock::now();
    auto duration = ::chrono::duration_cast<::chrono::microseconds>(end - start).count();

    // Write traversal
    for (int i = 0; i < traversalSize; i++) {
        traversalFile << traversal[i] << "\n";
        ::cout << traversal[i] << " ";
    }
    ::cout << "\n";

    // Write execution time
    timeFile << "Execution time: " << duration << " microseconds\n";
    ::cout << "BFS execution time: " << duration << " microseconds\n";

    traversalFile.close();
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

    int source;
    ::cout << "Enter source node for BFS: ";
    cin >> source;

    bfs(&g, source, nodeCount, "22i-2511");

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