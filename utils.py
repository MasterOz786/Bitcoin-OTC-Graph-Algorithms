"""
Utility functions for file I/O and analysis.
"""
import time
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def ensure_directory(directory):
    """
    Ensure that a directory exists.
    
    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_shortest_path_results(distances, predecessors, source, filename):
    """
    Save shortest path results to a file.
    
    Args:
        distances (dict): Shortest distances.
        predecessors (dict): Predecessors.
        source (int): Source node.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        f.write(f"Shortest paths from source node {source}:\n\n")
        
        for node in sorted(distances.keys()):
            if node == source:
                f.write(f"Node {node}: Distance = 0 (source)\n")
            elif distances[node] == float('inf'):
                f.write(f"Node {node}: Unreachable\n")
            else:
                # Reconstruct path
                path = []
                current = node
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                
                f.write(f"Node {node}: Distance = {distances[node]}, Path = {' -> '.join(map(str, path))}\n")

def save_mst_results(mst_edges, total_weight, filename):
    """
    Save minimum spanning tree results to a file.
    
    Args:
        mst_edges (list): Edges in the MST.
        total_weight (float): Total weight of MST.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        f.write(f"Minimum Spanning Tree:\n\n")
        f.write(f"Total weight: {total_weight}\n\n")
        f.write("Edges in MST:\n")
        
        for u, v, weight in mst_edges:
            f.write(f"({u}, {v}): {weight}\n")

def save_traversal_results(traversal_order, times, start_node, filename):
    """
    Save traversal results to a file.
    
    Args:
        traversal_order (list): Traversal order.
        times (dict or tuple): Times for DFS, distances for BFS.
        start_node (int): Starting node.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        f.write(f"Traversal from start node {start_node}:\n\n")
        f.write(f"Traversal order: {' -> '.join(map(str, traversal_order))}\n\n")
        
        if isinstance(times, tuple):  # DFS
            discovery_time, finish_time = times
            f.write("Discovery and finish times:\n")
            for node in sorted(discovery_time.keys()):
                if discovery_time[node] == -1:
                    f.write(f"Node {node}: Unreachable\n")
                else:
                    f.write(f"Node {node}: Discovery time = {discovery_time[node]}, Finish time = {finish_time[node]}\n")
        else:  # BFS
            distances = times
            f.write("Distances from start node:\n")
            for node in sorted(distances.keys()):
                if node == start_node:
                    f.write(f"Node {node}: Distance = 0 (start)\n")
                elif distances[node] == float('inf'):
                    f.write(f"Node {node}: Unreachable\n")
                else:
                    f.write(f"Node {node}: Distance = {distances[node]}\n")

def save_diameter_results(diameter, nodes, filename):
    """
    Save diameter results to a file.
    
    Args:
        diameter (float): Diameter of the graph.
        nodes (tuple): Nodes that achieve the diameter.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        f.write(f"Graph Diameter:\n\n")
        f.write(f"Diameter: {diameter}\n")
        f.write(f"Achieved between nodes {nodes[0]} and {nodes[1]}\n")

def save_cycle_results(cycle_exists, cycle_path, filename):
    """
    Save cycle detection results to a file.
    
    Args:
        cycle_exists (bool): Whether a cycle exists.
        cycle_path (list): A cycle, if one exists.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        f.write(f"Cycle Detection:\n\n")
        
        if cycle_exists:
            f.write(f"Cycle detected: {' -> '.join(map(str, cycle_path))}\n")
        else:
            f.write("No cycle detected\n")

def save_average_degree_results(avg_degree, filename):
    """
    Save average degree results to a file.
    
    Args:
        avg_degree (float): Average degree of the graph.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        f.write(f"Average Degree:\n\n")
        f.write(f"Average degree: {avg_degree}\n")

def save_execution_times(algorithm_times, filename):
    """
    Save execution times to a file.
    
    Args:
        algorithm_times (dict): Execution times for algorithms.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        f.write("Execution Times:\n\n")
        
        for algorithm, time in algorithm_times.items():
            f.write(f"{algorithm}: {time:.6f} seconds\n")

def create_performance_chart(algorithm_names, execution_times, filename):
    """
    Create a performance comparison chart.
    
    Args:
        algorithm_names (list): Names of algorithms.
        execution_times (list): Execution times.
        filename (str): Output filename.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(algorithm_names, execution_times)
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Algorithm Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

