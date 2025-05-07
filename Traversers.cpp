#include "Traversers.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>

using namespace std;

// Queue related methods
void Traversers::initQueue(Queue* q) {
    q->front = 0;
    q->rear = -1;
    q->size = 0;
}

int Traversers::isQueueEmpty(Queue* q) {
    return q->size == 0;
}

void Traversers::enqueue(Queue* q, int value) {
    if (q->size < MAX_QUEUE_SIZE) {
        q->rear = (q->rear + 1) % MAX_QUEUE_SIZE;
        q->data[q->rear] = value;
        q->size++;
    }
}

int Traversers::dequeue(Queue* q) {
    if (isQueueEmpty(q)) return -1;
    int value = q->data[q->front];
    q->front = (q->front + 1) % MAX_QUEUE_SIZE;
    q->size--;
    return value;
}

// Stack related functions
void Traversers::initStack(Stack* s) {
    s->top = -1;
}

int Traversers::isStackEmpty(Stack* s) {
    return s->top == -1;
}

void Traversers::push(Stack* s, int value) {
    if (s->top < MAX_STACK_SIZE - 1) {
        s->data[++s->top] = value;
    }
}

int Traversers::pop(Stack* s) {
    if (isStackEmpty(s)) return -1;
    return s->data[s->top--];
}

// Graph related methods
void Traversers::initGraph(Graph* g) {
    for (int i = 0; i < MAX_NODES; i++) {
        g->adj[i] = nullptr;
    }
    g->nodeCount = 0;
}

void Traversers::addEdge(Graph* g, int src, int dest, int weight) {
    Edge* e = new Edge;
    e->dest = dest;
    e->weight = weight;
    e->next = g->adj[src];
    g->adj[src] = e;
}

void Traversers::buildGraph(Nodes& data, Graph* g, int* nodes, int* nodeCount) {
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

// BFS related functions
void Traversers::bfs(Graph* g, int source, int nodeCount, const char* rollNumbers) {
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
    ::cout << "BFS execution time: " << duration << " microseconds or " << duration / 1000.0 << " seconds\n";

    traversalFile.close();
    traceFile.close();
    timeFile.close();
}

// DFS related functions
void Traversers::dfs(Graph* g, int source, int nodeCount, const char* rollNumbers) {
    char traversalFileName[100] = "DFS_Traversal_";
    char traceFileName[100] = "DFS_Trace_";
    char timeFileName[100] = "DFS_ExecutionTime_";
    strcat(traversalFileName, rollNumbers);
    strcat(traceFileName, rollNumbers);
    strcat(timeFileName, rollNumbers);
    strcat(traversalFileName, ".txt");
    strcat(traceFileName, ".txt");
    strcat(timeFileName, ".txt");

    ofstream traversalFile(traversalFileName);
    ofstream traceFile(traceFileName);
    ofstream timeFile(timeFileName);

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

    auto start = chrono::high_resolution_clock::now();

    Stack s;
    initStack(&s);
    push(&s, source);
    traceFile << "Push: " << source << "\n";

    while (!isStackEmpty(&s)) {
        int current = pop(&s);
        traceFile << "Pop: " << current << "\n";

        if (!visited[current]) {
            visited[current] = 1;
            traversal[traversalSize++] = current;

            // Collect neighbors to mimic recursive DFS order
            int neighbors[MAX_EDGES_PER_NODE];
            int neighborCount = 0;
            for (Edge* e = g->adj[current]; e; e = e->next) {
                if (!visited[e->dest]) {
                    neighbors[neighborCount++] = e->dest;
                }
            }
            // Push in reverse order
            for (int i = neighborCount - 1; i >= 0; i--) {
                push(&s, neighbors[i]);
                traceFile << "Push: " << neighbors[i] << "\n";
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Write traversal
    for (int i = 0; i < traversalSize; i++) {
        traversalFile << traversal[i] << "\n";
        cout << traversal[i] << " ";
    }
    cout << "\n";

    // Write execution time
    timeFile << "Execution time: " << duration << " microseconds\n";
    cout << "DFS execution time: " << duration << " microseconds or " << duration / 1000.0 << " seconds!\n";

    traversalFile.close();
    traceFile.close();
    timeFile.close();
}

int Traversers::dfsCycle(Graph* g, int node, int* visited, int* recStack, ofstream& traceFile, int& cycleFound) {
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

void Traversers::detectCycle(Graph* g, int nodeCount, const char* rollNumbers) {
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

    Traversers t;

    t.buildGraph(data, &g, nodes, &nodeCount);

    int source;
    cout << "Enter source node for DFS: ";
    cin >> source;

    t.dfs(&g, source, nodeCount, "22I-2511");

    cout << '\n';
    cout << "Enter source node for BFS: ";
    cin >> source;

    t.bfs(&g, source, nodeCount, "22I-2511");

    // detect cycle
    t.detectCycle(&g, nodeCount, "22I-2511");

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