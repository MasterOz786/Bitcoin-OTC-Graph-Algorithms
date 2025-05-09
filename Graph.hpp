#pragma once
#include "Loader.hpp"
#include <vector>
#include <unordered_map>
#include <set>

// edge ka matlab, teen cheezen, source, dest, wazan
using Kinara = std::tuple<int, int, int>; // source, dest, wazan

// Graph class, graph k liye sab kuch
class Graph {
    std::vector<Kinara> kinarey; // sab kinarey yahan ajate hain
    std::unordered_map<int, std::vector<std::pair<int, int>>> humsayaList; // humsaya list, jaldi dhoondnay k liye
    bool donoTaraf; // agar true hai to edge dono taraf jata, warna aik hi

public:
    Graph(bool bi = true); // graph banao, by default dono taraf

    // file se ya kahin se nodes lao, graph me dalo
    void nodesSayBanao(const Nodes& nodes);

    // aik kinara dalo, u say v, wazan w
    void kinaraDalo(int u, int v, int w);

    // sab kinarey wapis do
    const std::vector<Kinara>& sabKinareyLo() const;

    // humsaya list do, bfs dfs k liye
    const std::unordered_map<int, std::vector<std::pair<int, int>>>& humsayaListLo() const;

    // sab nodes ka set do, repeat na ho
    std::set<int> sabNodesLo() const;

    // average degree nikal lo, matlab har node k kitne kinarey
    double averageDegreeNikalo() const;

    // kitne nodes hain
    int nodesKiTadaad() const;

    // kitne kinarey hain, agar dono taraf hai to divide by 2
    int kinareyKiTadaad() const;
};