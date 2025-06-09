
#!/bin/bash

# Compile programs
g++ -o traversers Traversers.cpp Loader_222434_222515_222511.cpp -std=c++11
g++ -o singlesource SingleSource_222434_222515_222511.cpp Loader_222434_222515_222511.cpp -std=c++11
g++ -o m main_222434_222515_222511.cpp Timer_222434_222515_222511.cpp DisjointSet_222434_222515_222511.cpp Graph_222434_222515_222511.cpp MST_222434_222515_222511.cpp Loader_222434_222515_222511.cpp

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

# Run visualizations
python visualize_222434_222515_222511.py

echo "Data generation complete!"

# clean up 
rm traversers singlesource m
