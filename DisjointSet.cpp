#include "DisjointSet.hpp"

// naya set banao, har node apna baap, rutba zero
void DisjointSet::setBanao(int x) {
    baap[x] = x; // khud ka baap
    rutba[x] = 0; // rutba zero, naya banda
}

// baap dhoondo, agar baap khud nahi to baap ka baap dhoondo, path compress bhi
int DisjointSet::baapDhoondo(int x) {
    if (baap[x] != x)
        baap[x] = baapDhoondo(baap[x]); // shortcut bana do
    return baap[x];
}

// do set milao, rutba dekh k, jis ka rutba zyada uska baap
void DisjointSet::milao(int x, int y) {
    int x_baap = baapDhoondo(x);
    int y_baap = baapDhoondo(y);
    if (x_baap == y_baap) return; // already milay huay
    if (rutba[x_baap] < rutba[y_baap]) {
        baap[x_baap] = y_baap; // chota wale ka baap bara wala
    } else if (rutba[x_baap] > rutba[y_baap]) {
        baap[y_baap] = x_baap; // dusra chota
    } else {
        baap[y_baap] = x_baap; // koi bhi, rutba barha do
        rutba[x_baap]++;
    }
}