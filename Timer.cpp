#include "Timer.hpp"

// timer chalu karo, abhi ka waqt pakro
void Timer::shuruKaro() {
    shuruWaqt = std::chrono::high_resolution_clock::now(); // abhi ka time
}

// timer band karo, abhi ka waqt pakro
void Timer::bandKaro() {
    khatamWaqt = std::chrono::high_resolution_clock::now(); // abhi ka time
}

// kitna time guzra, milliseconds me, chota time
double Timer::guzraMilliseconds() const {
    return std::chrono::duration<double, std::milli>(khatamWaqt - shuruWaqt).count(); // farq nikal lo
}

// kitna time guzra, seconds me, bara time
double Timer::guzraSeconds() const {
    return std::chrono::duration<double>(khatamWaqt - shuruWaqt).count(); // seconds me farq
}