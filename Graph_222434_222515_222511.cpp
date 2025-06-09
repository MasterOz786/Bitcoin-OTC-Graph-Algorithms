#include "Graph.hpp"

// graph banao, donoTaraf set karo, default true hai
Graph::Graph(bool bi) : donoTaraf(bi) {}

// nodes ki list say graph banao, har node ko kinara samjho
void Graph::nodesSayBanao(const Nodes& nodes) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        kinaraDalo(nodes[i].source, nodes[i].dest, nodes[i].weight); // har node say kinara dalo
    }
}

// aik kinara dalo, u say v, wazan w
void Graph::kinaraDalo(int u, int v, int w) {
    kinarey.push_back(std::make_tuple(u, v, w)); // kinara list me dalo
    humsayaList[u].push_back(std::make_pair(v, w)); // humsaya list me bhi dalo
    if (donoTaraf) { // agar undirected hai to ulta bhi dalo
        humsayaList[v].push_back(std::make_pair(u, w));
        kinarey.push_back(std::make_tuple(v, u, w)); // reverse bhi
    }
}

// sab kinarey wapis do
const std::vector<Kinara>& Graph::sabKinareyLo() const {
    return kinarey;
}

// humsaya list do
const std::unordered_map<int, std::vector<std::pair<int, int>>>& Graph::humsayaListLo() const {
    return humsayaList;
}

// sab nodes ka set do, repeat na ho warna masla hoga
std::set<int> Graph::sabNodesLo() const {
    std::set<int> nodesSet;
    for (std::unordered_map<int, std::vector<std::pair<int, int>>>::const_iterator it = humsayaList.begin(); it != humsayaList.end(); ++it) {
        nodesSet.insert(it->first); // node dalo
    }
    return nodesSet;
}

// average degree nikal lo, sirf unique undirected kinarey ginno, self-loop ignore karo
double Graph::averageDegreeNikalo() const {
    std::set<std::pair<int, int> > uniqueKinarey;
    for (std::unordered_map<int, std::vector<std::pair<int, int>>>::const_iterator it = humsayaList.begin(); it != humsayaList.end(); ++it) {
        int u = it->first;
        for (size_t i = 0; i < it->second.size(); ++i) {
            int v = it->second[i].first;
            if (u == v) continue; // self-loop ignore karo
            int chota = std::min(u, v);
            int bara = std::max(u, v);
            uniqueKinarey.insert(std::make_pair(chota, bara)); // undirected edge ek hi dafa
        }
    }
    int nodesCount = humsayaList.size();
    if (nodesCount == 0) return 0.0;
    // har edge 2 nodes ko touch karta, degree = 2*edges/nodes
    return (2.0 * uniqueKinarey.size()) / nodesCount;
}

// kitne nodes hain, bas humsaya list ka size
int Graph::nodesKiTadaad() const {
    return humsayaList.size();
}

// kinarey kitne hain, agar dono taraf hai to divide by 2, warna jitne hain
int Graph::kinareyKiTadaad() const {
    if (donoTaraf) return kinarey.size() / 2;
    else return kinarey.size();
}