#include "Loader.hpp"
#include "Graph.hpp"
#include "MST.hpp"
#include "Timer.hpp"
#include <iostream>
#include <fstream>

int main() {
    // Data load karo, file se, bohat zaroori hai warna sab fail
    Nodes dataNodes = Loader::load(DATASET_FILENAME);

    // Graph banao, undirected hai, mst k liye
    Graph meraGraph(true);
    meraGraph.nodesSayBanao(dataNodes);

    // Average degree nikal lo, phir file me daal do
    double avDegree = meraGraph.averageDegreeNikalo();
    std::cout << "Average Degree: " << avDegree << std::endl;
    std::ofstream avgFile("average_degree.txt");
    avgFile << "Average Degree: " << avDegree << std::endl;
    avgFile.close();

    // Prim's mst chalao, time bhi dekho
    Timer ghadi;
    ghadi.shuruKaro();
    int shuruNode = 0;
    std::set<int> sabNodes = meraGraph.sabNodesLo();
    if (!sabNodes.empty()) {
        shuruNode = *sabNodes.begin(); // pehla node le lo, warna 0
    }
    std::pair<std::vector<Kinara>, int> primNatija = MST::prim(meraGraph, shuruNode, "mst_trace_prim.txt");
    ghadi.bandKaro();
    std::vector<Kinara> primMST = primNatija.first;
    int primWazan = primNatija.second;
    std::cout << "Prim's MST total weight: " << primWazan << std::endl;
    std::cout << "Prim's execution time: " << ghadi.guzraMilliseconds() << " ms" << std::endl;

    std::ofstream primOut("mst_result_prim.txt");
    primOut << "Prim's MST Edges:\n";
    for (size_t i = 0; i < primMST.size(); ++i) {
        int u = std::get<0>(primMST[i]);
        int v = std::get<1>(primMST[i]);
        int w = std::get<2>(primMST[i]);
        primOut << u << " - " << v << " (weight " << w << ")\n";
    }
    primOut << "Total weight: " << primWazan << "\n";
    primOut << "Execution time: " << ghadi.guzraMilliseconds() << " ms\n";
    primOut.close();

    // Kruskal bhi chalao, warna teacher bolega kyu nahi kiya
    ghadi.shuruKaro();
    std::pair<std::vector<Kinara>, int> kruskalNatija = MST::kruskal(meraGraph, "mst_trace_kruskal.txt");
    ghadi.bandKaro();
    std::vector<Kinara> kruskalMST = kruskalNatija.first;
    int kruskalWazan = kruskalNatija.second;
    std::cout << "Kruskal's MST total weight: " << kruskalWazan << std::endl;
    std::cout << "Kruskal's execution time: " << ghadi.guzraMilliseconds() << " ms" << std::endl;

    std::ofstream kruskalOut("mst_result_kruskal.txt");
    kruskalOut << "Kruskal's MST Edges:\n";
    for (size_t i = 0; i < kruskalMST.size(); ++i) {
        int u = std::get<0>(kruskalMST[i]);
        int v = std::get<1>(kruskalMST[i]);
        int w = std::get<2>(kruskalMST[i]);
        kruskalOut << u << " - " << v << " (weight " << w << ")\n";
    }
    kruskalOut << "Total weight: " << kruskalWazan << "\n";
    kruskalOut << "Execution time: " << ghadi.guzraMilliseconds() << " ms\n";
    kruskalOut.close();

    return 0;
}