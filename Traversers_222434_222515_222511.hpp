
#pragma once

#include "Loader.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>

using namespace std;

#define MAX_NODES 100000
#define MAX_EDGES_PER_NODE 1000
#define MAX_QUEUE_SIZE 100000
#define MAX_STACK_SIZE 100000

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

struct Stack {
    int data[MAX_STACK_SIZE];
    int top;
};

class Traversers {
    public:
        // Queue related methods
        void initQueue(Queue* q);
        int isQueueEmpty(Queue* q);
        void enqueue(Queue* q, int value);
        int dequeue(Queue* q);

        // Stack related methods
        void initStack(Stack*);
        int isStackEmpty(Stack*);
        void push(Stack*, int);
        int pop(Stack*);

        // Graph related methods
        void initGraph(Graph* g);
        void addEdge(Graph* g, int src, int dest, int weight);
        void buildGraph(Nodes& data, Graph* g, int* nodes, int* nodeCount);

        void bfs(Graph* g, int source, int nodeCount, const char* rollNumbers);
        void dfs(Graph* g, int source, int nodeCount, const char* rollNumbers);
        int dfsCycle(Graph* g, int node, int* visited, int* recStack, ofstream& traceFile, int& cycleFound);
        void detectCycle(Graph* g, int nodeCount, const char* rollNumbers);
};