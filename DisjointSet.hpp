#pragma once
#include <unordered_map>

// yeh class set banata, milata, union karta, mst k liye zaroori hai
class DisjointSet {
    std::unordered_map<int, int> baap; // har node ka baap kon hai
    std::unordered_map<int, int> rutba; // rank ya rutba, union k liye

public:
    // naya set banao, har node apna baap
    void setBanao(int x);

    // baap dhoondo, path compress bhi karo warna slow ho jata
    int baapDhoondo(int x);

    // do set milao, rutba dekh k
    void milao(int x, int y);
};