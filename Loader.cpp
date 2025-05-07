
#include "Loader.hpp"

Nodes Loader::load(const char* filename) {
    std::ifstream file(filename);
    Nodes data;
    
    if (!file.is_open()) {
        std::cerr << "Could not open the file.\n";
        return data;
    }

    std::string line;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string source, dest, weight, timestamp;

        getline(ss, source, ',');
        getline(ss, dest, ',');
        getline(ss, weight, ',');
        getline(ss, timestamp, ',');

        // std::cout << "From: " << source << ", To: " << dest
        //           << ", Rating: " << weight << ", Timestamp: " << timestamp << '\n';
        
        
        Node n;
        n.source = std::stoi(source);
        n.dest = std::stoi(dest);
        n.weight = std::stoi(weight);
        n.timestamp = std::stoi(timestamp);

        data.push_back(n);
    }

    file.close();
    return data;
}