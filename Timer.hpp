#pragma once
#include <chrono>

// yeh timer class hai, time nikalnay k liye, bohat zaroori hai jab speed dekhni ho
class Timer {
    std::chrono::high_resolution_clock::time_point shuruWaqt, khatamWaqt; // time ka hisab
public:
    // timer chalu karo, abhi say time lo
    void shuruKaro();

    // timer band karo, abhi ka time lo
    void bandKaro();

    // kitna time guzra, milliseconds me, chota time
    double guzraMilliseconds() const;

    // kitna time guzra, seconds me, bara time
    double guzraSeconds() const;
};