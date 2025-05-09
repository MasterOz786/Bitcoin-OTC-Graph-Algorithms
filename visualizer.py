
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
from performance_visualizer import (
    parse_execution_time_file,
    generate_performance_graph,
    compare_algorithms
)

def process_execution_time_data():
    """Process all execution time files and organize data for comparison"""
    algorithm_data = {}
    
    # Find all execution time files
    time_files = glob.glob("*_ExecutionTime_*.txt")
    
    for file_path in time_files:
        algorithm_name = file_path.split('_')[0]
        
        if algorithm_name not in algorithm_data:
            algorithm_data[algorithm_name] = {"input_sizes": [], "execution_times": []}
        
        input_sizes, execution_times = parse_execution_time_file(file_path)
        algorithm_data[algorithm_name]["input_sizes"].extend(input_sizes)
        algorithm_data[algorithm_name]["execution_times"].extend(execution_times)
    
    # Sort data for each algorithm
    for algo in algorithm_data:
        sorted_data = sorted(zip(algorithm_data[algo]["input_sizes"], 
                                algorithm_data[algo]["execution_times"]))
        if sorted_data:
            algorithm_data[algo]["input_sizes"], algorithm_data[algo]["execution_times"] = zip(*sorted_data)
    
    return algorithm_data

def compare_algorithm_scaling(algorithm_data):
    """Compare how different algorithms scale with input size"""
    plt.figure(figsize=(14, 10))
    
    for algo, data in algorithm_data.items():
        if not data["input_sizes"]:
            continue
        
        plt.plot(data["input_sizes"], data["execution_times"], 
                 marker='o', linestyle='-', label=algo)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Input Size (number of nodes)')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Comparison of Algorithm Performance Scaling')
    plt.legend()
    plt.tight_layout()
    plt.savefig("algorithm_scaling_comparison.png")
    plt.close()
    
    print(f"Generated algorithm scaling comparison: algorithm_scaling_comparison.png")

def compare_traversal_vs_singlesource(algorithm_data):
    """Compare traversal algorithms vs. single source algorithms"""
    plt.figure(figsize=(14, 10))
    
    traversal_algos = ["BFS", "DFS"]
    singlesource_algos = ["Dijkstra", "BellmanFord"]
    
    # Plot traversal algorithms
    for algo in traversal_algos:
        if algo in algorithm_data and algorithm_data[algo]["input_sizes"]:
            plt.plot(algorithm_data[algo]["input_sizes"], 
                     algorithm_data[algo]["execution_times"],
                     marker='o', linestyle='-', label=algo)
    
    # Plot single source algorithms
    for algo in singlesource_algos:
        if algo in algorithm_data and algorithm_data[algo]["input_sizes"]:
            plt.plot(algorithm_data[algo]["input_sizes"], 
                     algorithm_data[algo]["execution_times"], 
                     marker='s', linestyle='--', label=algo)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Input Size (number of nodes)')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Traversal Algorithms vs. Single Source Shortest Path Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig("traversal_vs_singlesource.png")
    plt.close()
    
    print(f"Generated comparison: traversal_vs_singlesource.png")

def compare_order_of_growth(algorithm_data):
    """Compare actual performance with theoretical complexity models"""
    plt.figure(figsize=(14, 10))
    
    # For each algorithm, plot actual data
    for algo, data in algorithm_data.items():
        if not data["input_sizes"]:
            continue
        
        x = np.array(data["input_sizes"])
        y = np.array(data["execution_times"])
        
        plt.scatter(x, y, label=f"{algo} (Actual)", alpha=0.7)
        
        # Find best fit line for logarithmic plotting
        if len(x) > 1:
            # Try to fit different curves
            # For O(n) - linear model
            if algo in ["BFS", "DFS"]:
                # For traversal algorithms, expect closer to linear
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), linestyle='--', 
                         label=f"{algo} (O(n) fit)")
            
            # For O(n log n) or O(n²) - for weighted graph algorithms
            elif algo in ["Dijkstra", "BellmanFord"]:
                try:
                    # Try O(n log n) fit for Dijkstra
                    if algo == "Dijkstra":
                        z = np.polyfit(x * np.log(x), y, 1)
                        p = np.poly1d(z)
                        plt.plot(x, p(x * np.log(x)), linestyle=':', 
                                label=f"{algo} (O(n log n) fit)")
                    
                    # Try O(n²) fit for Bellman-Ford
                    if algo == "BellmanFord":
                        z = np.polyfit(x**2, y, 1)
                        p = np.poly1d(z)
                        plt.plot(x, p(x**2), linestyle='-.', 
                                label=f"{algo} (O(n²) fit)")
                except:
                    # If fitting fails, just continue
                    pass
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Input Size (number of nodes)')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Algorithm Performance Compared to Theoretical Complexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_comparison.png")
    plt.close()
    
    print(f"Generated complexity comparison: complexity_comparison.png")

def visualize_node_density_comparison(algorithm_data):
    """Compare algorithm performance based on graph density"""
    # We can approximate density by looking at average degree from AverageDegree_Result file
    avg_degree = None
    try:
        with open("AverageDegree_Result_22I-2515.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Average degree:" in line:
                    avg_degree = float(line.split(":")[1].strip())
    except:
        print("Could not read average degree file")
    
    if avg_degree is None:
        print("Average degree data not available, skipping density comparison")
        return
    
    # Create a plot that includes average degree information
    plt.figure(figsize=(14, 10))
    
    for algo, data in algorithm_data.items():
        if not data["input_sizes"]:
            continue
        
        plt.plot(data["input_sizes"], data["execution_times"], 
                 marker='o', linestyle='-', label=f"{algo}")
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Input Size (number of nodes)')
    plt.ylabel('Execution Time (microseconds)')
    plt.title(f'Algorithm Performance (Graph Density: Avg Degree = {avg_degree:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig("node_density_comparison.png")
    plt.close()
    
    print(f"Generated node density comparison: node_density_comparison.png")

def main():
    print("Starting algorithm performance analysis...")
    
    # Check which algorithm execution time files exist
    algorithm_data = process_execution_time_data()
    available_algorithms = list(algorithm_data.keys())
    
    print(f"Available algorithm data: {available_algorithms}")
    
    # If we have actual data, visualize it
    if available_algorithms:
        # Generate individual algorithm graphs
        for algorithm in available_algorithms:
            if algorithm_data[algorithm]["input_sizes"]:
                generate_performance_graph(algorithm, f"{algorithm}_performance.png")
        
        # Generate comparison visualizations
        if len(available_algorithms) > 1:
            # Compare all algorithms together
            compare_algorithms(available_algorithms, "all_algorithm_comparison.png")
            
            # Compare how algorithms scale with input size
            compare_algorithm_scaling(algorithm_data)
            
            # Compare traversal vs single source algorithms
            compare_traversal_vs_singlesource(algorithm_data)
            
            # Compare actual performance with theoretical complexity
            compare_order_of_growth(algorithm_data)
            
            # Compare based on graph density
            visualize_node_density_comparison(algorithm_data)
    
    print("Visualization complete! The following image files have been generated:")
    for alg in available_algorithms:
        print(f"- {alg}_performance.png")
    if len(available_algorithms) > 1:
        print("- all_algorithm_comparison.png")
        print("- algorithm_scaling_comparison.png")
        print("- traversal_vs_singlesource.png")
        print("- complexity_comparison.png")
        print("- node_density_comparison.png")

if __name__ == "__main__":
    main()

