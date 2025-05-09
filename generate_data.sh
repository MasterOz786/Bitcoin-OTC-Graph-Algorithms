
#!/bin/bash

# Compile programs
g++ -o traversers Traversers.cpp Loader.cpp -std=c++11
g++ -o singlesource SingleSource.cpp Loader.cpp -std=c++11
g++ -o m main.cpp Timer.cpp DisjointSet.cpp Graph.cpp MST.cpp Loader.cpp

# Run traversal algorithms (BFS, DFS, Cycle detection) with different source nodes
echo "Generating data for graph traversal algorithms..."
for source in 5 10 20 50 100 200 500 1000
do
  echo "Running with source node $source"
  echo $source | ./traversers
done

# Run shortest path algorithms (Dijkstra, Bellman-Ford) with different source nodes
echo "Generating data for shortest path algorithms..."
for source in 5 10 20 50 100 200 500
do
  echo "Running with source node $source"
  echo $source | ./singlesource
done

# Run Prims and Kruskal
./m

echo "Data generation complete!"

# clean up 
rm traversers singlesource m
