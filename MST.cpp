#include "MST.hpp"
#include "DisjointSet.hpp"
#include <queue>
#include <fstream>
#include <algorithm>

// prim ka algo, mst nikalnay wala, bohat kaam ka
std::pair<std::vector<Kinara>, int> MST::prim(const Graph& graph, int shuruNode, const std::string& traceFile) {
    std::set<int> hoGaye; // jo nodes ho chuke
    std::vector<Kinara> mstKinarey; // mst ke kinarey
    int totalWazan = 0; // sab ka wazan

    auto humsaya = graph.humsayaListLo(); // sab humsaya le lo
    std::ofstream trace(traceFile); // file khol lo

    typedef std::tuple<int, int, int> T; // wazan, u, v
    std::priority_queue<T, std::vector<T>, std::greater<T> > pq; // chota wazan pehle

    hoGaye.insert(shuruNode); // pehla node ho gaya
    const std::vector<std::pair<int, int> >& shuruAdj = humsaya.at(shuruNode);
    for (size_t i = 0; i < shuruAdj.size(); ++i) {
        int v = shuruAdj[i].first;
        int w = shuruAdj[i].second;
        pq.push(std::make_tuple(w, shuruNode, v)); // sab kinarey dal do
    }

    while (!pq.empty() && mstKinarey.size() < graph.nodesKiTadaad() - 1) {
        T top = pq.top(); pq.pop();
        int w = std::get<0>(top);
        int u = std::get<1>(top);
        int v = std::get<2>(top);
        if (hoGaye.count(v)) continue; // already ho gaya
        hoGaye.insert(v);
        mstKinarey.push_back(std::make_tuple(u, v, w)); // mst me dalo
        totalWazan += w;
        // English trace
        trace << "Selected edge: " << u << " - " << v << " (weight " << w << ")\n";
        const std::vector<std::pair<int, int> >& vAdj = humsaya.at(v);
        for (size_t i = 0; i < vAdj.size(); ++i) {
            int to = vAdj[i].first;
            int w2 = vAdj[i].second;
            if (!hoGaye.count(to)) {
                pq.push(std::make_tuple(w2, v, to));
            }
        }
    }
    trace << "Total MST weight: " << totalWazan << "\n";
    trace.close();
    return std::make_pair(mstKinarey, totalWazan);
}

// kruskal ka algo, mst nikalta, union find use karta
std::pair<std::vector<Kinara>, int> MST::kruskal(const Graph& graph, const std::string& traceFile) {
    std::vector<Kinara> kinarey = graph.sabKinareyLo(); // sab kinarey le lo
    std::sort(kinarey.begin(), kinarey.end(), [](const Kinara& a, const Kinara& b) {
        return std::get<2>(a) < std::get<2>(b); // wazan pe sort karo
    });

    DisjointSet ds;
    std::set<int> nodes = graph.sabNodesLo();
    for (std::set<int>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
        ds.setBanao(*it); // har node ka set banao
    }

    std::vector<Kinara> mstKinarey;
    int totalWazan = 0;
    std::ofstream trace(traceFile);

    for (size_t i = 0; i < kinarey.size(); ++i) {
        int u = std::get<0>(kinarey[i]);
        int v = std::get<1>(kinarey[i]);
        int w = std::get<2>(kinarey[i]);
        if (ds.baapDhoondo(u) != ds.baapDhoondo(v)) {
            ds.milao(u, v); // dono ko mila do
            mstKinarey.push_back(std::make_tuple(u, v, w));
            totalWazan += w;
            // English trace
            trace << "Selected edge: " << u << " - " << v << " (weight " << w << ")\n";
            if (mstKinarey.size() == graph.nodesKiTadaad() - 1) break;
        }
    }
    trace << "Total MST weight: " << totalWazan << "\n";
    trace.close();
    return std::make_pair(mstKinarey, totalWazan);
}