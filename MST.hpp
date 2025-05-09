#pragma once
#include "Graph.hpp"
#include <vector>
#include <tuple>
#include <set>

// MST class, yahan dono algo hain, prim aur kruskal
class MST {
public:
    // Prim's algo, mst nikalta, startNode say shuru hota, trace file me likhta
    static std::pair<std::vector<Kinara>, int> prim(const Graph& graph, int shuruNode, const std::string& traceFile);

    // Kruskal's algo, mst nikalta, trace file me likhta
    static std::pair<std::vector<Kinara>, int> kruskal(const Graph& graph, const std::string& traceFile);
};