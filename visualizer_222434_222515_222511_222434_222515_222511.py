import networkx as nx
import matplotlib.pyplot as plt
import time
import heapq
import os
from typing import Dict, List, Tuple

class GraphAnalyzer:
    def __init__(self, graph_file):
        """
        Initialize graph from file or create graph
        """
        # Load graph from file or create graph
        self.G = self.load_graph(graph_file)
        
        # Create output directory
        os.makedirs('output', exist_ok=True)

    def load_graph(self, graph_file):
        """
        Load graph from various formats
        """
        # Implement graph loading logic
        # For this example, we'll create a sample graph
        G = nx.read_edgelist(graph_file, nodetype=int, create_using=nx.Graph())
        return G

    def dijkstra_shortest_path(self, source):
        """
        Dijkstra's Shortest Path Algorithm
        """
        start_time = time.time()
        
        # Dijkstra implementation
        distances = {node: float('inf') for node in self.G.nodes()}
        distances[source] = 0
        predecessors = {node: None for node in self.G.nodes()}
        
        pq = [(0, source)]
        trace_log = []
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # Log trace
            trace_log.append(f"Processing node {current_node} with distance {current_distance}")
            
            # If we've found a longer path, skip
            if current_distance > distances[current_node]:
                continue
            
            for neighbor in self.G.neighbors(current_node):
                weight = self.G[current_node][neighbor].get('weight', 1)
                distance = current_distance + weight
                
                # If we've found a shorter path, update
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
                    
                    trace_log.append(f"Updated path to {neighbor}: distance = {distance}")
        
        end_time = time.time()
        
        # Write results to file
        with open('output/dijkstra_results.txt', 'w') as f:
            f.write("Dijkstra Shortest Path Results:\n")
            f.write(f"Source Node: {source}\n")
            f.write("Distances:\n")
            for node, dist in distances.items():
                f.write(f"Node {node}: {dist}\n")
        
        # Write trace log
        with open('output/dijkstra_trace.txt', 'w') as f:
            f.writelines('\n'.join(trace_log))
        
        return distances, predecessors, end_time - start_time

    def bellman_ford_shortest_path(self, source):
        """
        Bellman-Ford Shortest Path Algorithm
        """
        start_time = time.time()
        
        # Initialize distances and predecessors
        distances = {node: float('inf') for node in self.G.nodes()}
        distances[source] = 0
        predecessors = {node: None for node in self.G.nodes()}
        trace_log = []
        
        # Relax edges |V| - 1 times
        for _ in range(len(self.G.nodes()) - 1):
            for u, v in self.G.edges():
                weight = self.G[u][v].get('weight', 1)
                
                # If we can improve the distance
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    trace_log.append(f"Relaxed edge {u}->{v}: new distance = {distances[v]}")
        
        # Check for negative weight cycles
        for u, v in self.G.edges():
            weight = self.G[u][v].get('weight', 1)
            if distances[u] + weight < distances[v]:
                raise ValueError("Graph contains a negative weight cycle")
        
        end_time = time.time()
        
        # Write results to file
        with open('output/bellman_ford_results.txt', 'w') as f:
            f.write("Bellman-Ford Shortest Path Results:\n")
            f.write(f"Source Node: {source}\n")
            f.write("Distances:\n")
            for node, dist in distances.items():
                f.write(f"Node {node}: {dist}\n")
        
        # Write trace log
        with open('output/bellman_ford_trace.txt', 'w') as f:
            f.writelines('\n'.join(trace_log))
        
        return distances, predecessors, end_time - start_time

    def prims_mst(self):
        """
        Prim's Minimum Spanning Tree Algorithm
        """
        start_time = time.time()
        
        # Start from an arbitrary node
        start_node = list(self.G.nodes())[0]
        
        # Initialize
        mst = nx.Graph()
        visited = set([start_node])
        edges = [(self.G[start_node][to].get('weight', 1), start_node, to) 
                 for to in self.G[start_node]]
        heapq.heapify(edges)
        
        trace_log = [f"Starting MST from node {start_node}"]
        
        while edges:
            weight, frm, to = heapq.heappop(edges)
            
            if to not in visited:
                visited.add(to)
                mst.add_edge(frm, to, weight=weight)
                trace_log.append(f"Added edge {frm}-{to} with weight {weight}")
                
                # Add new edges
                for next_node in self.G[to]:
                    if next_node not in visited:
                        heapq.heappush(
                            edges, 
                            (self.G[to][next_node].get('weight', 1), to, next_node)
                        )
        
        end_time = time.time()
        
        # Write MST to file
        with open('output/prims_mst.txt', 'w') as f:
            f.write("Prim's Minimum Spanning Tree Edges:\n")
            for edge in mst.edges(data=True):
                f.write(f"{edge[0]} - {edge[1]}: {edge[2]['weight']}\n")
        
        # Write trace log
        with open('output/prims_trace.txt', 'w') as f:
            f.writelines('\n'.join(trace_log))
        
        return mst, end_time - start_time

    def kruskals_mst(self):
        """
        Kruskal's Minimum Spanning Tree Algorithm
        """
        start_time = time.time()
        
        # Create a disjoint set data structure
        parent = {node: node for node in self.G.nodes()}
        
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_u] = root_v
                return True
            return False
        
        # Sort edges by weight
        edges = sorted(
            self.G.edges(data=True), 
            key=lambda x: x[2].get('weight', 1)
        )
        
        # Minimum Spanning Tree
        mst = nx.Graph()
        trace_log = ["Starting Kruskal's MST algorithm"]
        
        for u, v, data in edges:
            if union(u, v):
                mst.add_edge(u, v, weight=data.get('weight', 1))
                trace_log.append(f"Added edge {u}-{v} with weight {data.get('weight', 1)}")
        
        end_time = time.time()
        
        # Write MST to file
        with open('output/kruskals_mst.txt', 'w') as f:
            f.write("Kruskal's Minimum Spanning Tree Edges:\n")
            for edge in mst.edges(data=True):
                f.write(f"{edge[0]} - {edge[1]}: {edge[2]['weight']}\n")
        
        # Write trace log
        with open('output/kruskals_trace.txt', 'w') as f:
            f.writelines('\n'.join(trace_log))
        
        return mst, end_time - start_time

    def bfs_traversal(self, source):
        """
        Breadth-First Search Traversal
        """
        start_time = time.time()
        
        visited = set()
        queue = [source]
        visited.add(source)
        traversal_order = []
        trace_log = [f"Starting BFS from node {source}"]
        
        while queue:
            current_node = queue.pop(0)
            traversal_order.append(current_node)
            trace_log.append(f"Visited node {current_node}")
            
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    trace_log.append(f"Added {neighbor} to queue")
        
        end_time = time.time()
        
        # Write results to file
        with open('output/bfs_results.txt', 'w') as f:
            f.write("BFS Traversal Order:\n")
            f.write(" -> ".join(map(str, traversal_order)))
        
        # Write trace log
        with open('output/bfs_trace.txt', 'w') as f:
            f.writelines('\n'.join(trace_log))
        
        return traversal_order, end_time - start_time

    def dfs_traversal(self, source):
        """
        Depth-First Search Traversal
        """
        start_time = time.time()
        
        visited = set()
        traversal_order = []
        trace_log = [f"Starting DFS from node {source}"]
        
        def dfs_recursive(node):
            visited.add(node)
            traversal_order.append(node)
            trace_log.append(f"Visited node {node}")
            
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    trace_log.append(f"Exploring neighbor {neighbor}")
                    dfs_recursive(neighbor)
        
        dfs_recursive(source)
        
        end_time = time.time()
        
        # Write results to file
        with open('output/dfs_results.txt', 'w') as f:
            f.write("DFS Traversal Order:\n")
            f.write(" -> ".join(map(str, traversal_order)))
        
        # Write trace log
        with open('output/dfs_trace.txt', 'w') as f:
            f.writelines('\n'.join(trace_log))
        
        return traversal_order, end_time - start_time

    def find_graph_diameter(self):
        """
        Find the diameter of the graph
        """
        start_time = time.time()
        
        # Compute all-pairs shortest paths
        diameter = 0
        trace_log = ["Computing graph diameter"]
        
        # Use Floyd-Warshall for all-pairs shortest paths
        shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))
        
        for source in self.G.nodes():
            for target in self.G.nodes():
                if source != target:
                    path_length = shortest_paths[source][target]
                    diameter = max(diameter, path_length)
                    trace_log.append(f"Path {source}->{target}: length {path_length}")
        
        end_time = time.time()
        
        # Write results to file
        with open('output/diameter_results.txt', 'w') as f:
            f.write(f"Graph Diameter: {diameter}\n")
        
        # Write trace log
        with open('output/diameter_trace.txt', 'w') as f:
            f.writelines('\n'.join(trace_log))
        
        return diameter, end_time - start_time

    def detect_cycle(self):
        """
        Detect cycle in the graph
        """
        start_time = time.time()
        
        # Use NetworkX cycle detection
        cycles = list(nx.cycle_basis(self.G))
        
        end_time = time.time()
        
        # Write results to file
        with open('output/cycle_detection.txt', 'w') as f:
            if cycles:
                f.write("Cycles Detected:\n")
                for i, cycle in enumerate(cycles, 1):
                    f.write(f"Cycle {i}: {cycle}\n")
            else:
                f.write("No cycles found in the graph.\n")
        
        return cycles, end_time - start_time

def main():
    # Specify graph file path
    graph_file = 'soc.csv'
    
    # Create graph analyzer
    analyzer = GraphAnalyzer(graph_file)
    
    # Select source node (first node in the graph)
    source_node = list(analyzer.G.nodes())[0]
    
    # Run all algorithms
    print("Running Dijkstra's Shortest Path...")
    dijkstra_distances, _, dijkstra_time = analyzer.dijkstra_shortest_path(source_node)
    
    print("Running Bellman-Ford Shortest Path...")
    bellman_ford_distances, _, bellman_ford_time = analyzer.bellman_ford_shortest_path(source_node)
    
    print("Running Prim's MST...")
    prims_mst, prims_time = analyzer.prims_mst()
    
    print("Running Kruskal's MST...")
    kruskals_mst, kruskals_time = analyzer.kruskals_mst()
    
    print("Running BFS Traversal...")
    bfs_order, bfs_time = analyzer.bfs_traversal(source_node)
    
    print("Running DFS Traversal...")
    dfs_order, dfs_time = analyzer.dfs_traversal(source_node)
    
    print("Finding Graph Diameter...")
    diameter, diameter_time = analyzer.find_graph_diameter()
    
    print("Detecting Cycles...")
    cycles, cycle_time = analyzer.detect_cycle()

if __name__ == '__main__':
    main()