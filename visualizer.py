import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from performance_visualizer import (
    parse_execution_time_file,
    generate_performance_graph,
    compare_algorithms
)

def generate_synthetic_data(nodes, algo, edges=None, directed=False, weighted=False):
    """Generate synthetic execution times based on algorithm and graph properties."""
    execution_times = []
    
    for i, n in enumerate(nodes):
        # Determine edge count for this node
        if isinstance(edges, (list, np.ndarray)):
            edge_count = edges[i]  # Use corresponding edge count
        else:
            edge_count = n * 7.30896 if n < 1000 else n * np.log(n)  # Default sparse graph
        
        # Calculate base execution time
        if algo == "BFS":
            time = 6000 * (n / 1000) if not weighted else 6500 * (n / 1000)  # ~6s for 1000 nodes
            time *= 1.1 if directed else 1.0  # Directed graphs slightly slower
        elif algo == "DFS":
            time = 13 * (n / 1000) if not weighted else 15 * (n / 1000)  # ~0.013s for 1000 nodes
            time *= 1.05 if directed else 1.0
        elif algo == "Dijkstra":
            time = 1000 * (n * np.log(n) / (1000 * np.log(1000)))  # O((V+E)logV)
            time *= 1.2 if weighted else 1.0
        elif algo == "BellmanFord":
            time = 5000 * (n * edge_count / (1000 * 1000))  # O(VE)
            time *= 1.3 if weighted else 1.0
        elif algo == "Prims":
            time = 43481.7 * (n**2 / 1000**2)  # O(V^2), scaled from data
            time *= 1.15 if weighted else 1.0
        elif algo == "Kruskal":
            time = 73174.2 * (edge_count * np.log(edge_count) / (1000 * np.log(1000)))  # O(E log E)
            time *= 1.1 if weighted else 1.0
        else:
            time = 1000 * (n / 1000)
        
        # Adjust for dense graphs
        if edge_count > n * np.log(n) * 10:  # Dense graph condition
            time *= 1.5  # Dense graphs increase runtime
        
        execution_times.append(time)
    
    return execution_times

def process_execution_time_data(use_synthetic=True):
    """Process execution time files or generate synthetic data."""
    algorithm_data = {}
    nodes = [100, 200, 500, 1000, 2000, 5000, 10000]
    algorithms = ["BFS", "DFS", "Dijkstra", "BellmanFord", "Prims", "Kruskal"]

    if use_synthetic:
        for algo in algorithms:
            algorithm_data[algo] = {
                "input_sizes": nodes,
                "execution_times": generate_synthetic_data(nodes, algo)
            }
    else:
        time_files = glob.glob("*_ExecutionTime_*.txt")
        for file_path in time_files:
            algorithm_name = file_path.split('_')[0]
            if algorithm_name not in algorithm_data:
                algorithm_data[algorithm_name] = {"input_sizes": [], "execution_times": []}
            input_sizes, execution_times = parse_execution_time_file(file_path)
            algorithm_data[algorithm_name]["input_sizes"].extend(input_sizes)
            algorithm_data[algorithm_name]["execution_times"].extend(execution_times)
        
        for algo in algorithm_data:
            sorted_data = sorted(zip(algorithm_data[algo]["input_sizes"],
                                    algorithm_data[algo]["execution_times"]))
            if bounded_data:
                algorithm_data[algo]["input_sizes"], algorithm_data[algo]["execution_times"] = zip(*sorted_data)

    return algorithm_data

def compare_algorithm_scaling(algorithm_data):
    """Compare how different algorithms scale with input size."""
    plt.figure(figsize=(14, 10))
    for algo, data in algorithm_data.items():
        if not data["input_sizes"]:
            continue
        plt.plot(data["input_sizes"], data["execution_times"],
                 marker='o', linestyle='-', label=algo)
    
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Algorithm Performance Scaling')
    plt.legend()
    plt.tight_layout()
    plt.savefig("algorithm_scaling_comparison.png")
    plt.close()
    print("Generated: algorithm_scaling_comparison.png")

def compare_traversal_vs_singlesource(algorithm_data):
    """Compare traversal algorithms vs. single source algorithms."""
    plt.figure(figsize=(14, 10))
    traversal_algos = ["BFS", "DFS"]
    singlesource_algos = ["Dijkstra", "BellmanFord"]
    
    for algo in traversal_algos + singlesource_algos:
        if algo in algorithm_data and algorithm_data[algo]["input_sizes"]:
            marker = 'o' if algo in traversal_algos else 's'
            linestyle = '-' if algo in traversal_algos else '--'
            plt.plot(algorithm_data[algo]["input_sizes"],
                     algorithm_data[algo]["execution_times"],
                     marker=marker, linestyle=linestyle, label=algo)
    
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Traversal vs. Single Source Shortest Path Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig("traversal_vs_singlesource.png")
    plt.close()
    print("Generated: traversal_vs_singlesource.png")

def compare_order_of_growth(algorithm_data):
    """Compare actual performance with theoretical complexity models."""
    plt.figure(figsize=(14, 10))
    for algo, data in algorithm_data.items():
        if not data["input_sizes"]:
            continue
        x = np.array(data["input_sizes"])
        y = np.array(data["execution_times"])
        plt.scatter(x, y, label=f"{algo} (Actual)", alpha=0.7)
        
        if len(x) > 1:
            try:
                if algo in ["BFS", "DFS"]:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x), linestyle='--', label=f"{algo} (O(n) fit)")
                elif algo == "Dijkstra":
                    z = np.polyfit(x * np.log(x), y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x * np.log(x)), linestyle=':', label=f"{algo} (O(n log n) fit)")
                elif algo == "BellmanFord":
                    z = np.polyfit(x**2, y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x**2), linestyle='-.', label=f"{algo} (O(n²) fit)")
                elif algo == "Kruskal":
                    z = np.polyfit(x * np.log(x), y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x * np.log(x)), linestyle='--', label=f"{algo} (O(E log E) fit)")
                elif algo == "Prims":
                    z = np.polyfit(x**2, y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x**2), linestyle=':', label=f"{algo} (O(n²) fit)")
            except Exception as e:
                print(f"Skipping curve fit for {algo}: {e}")
    
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Algorithm Performance vs. Theoretical Complexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_comparison.png")
    plt.close()
    print("Generated: complexity_comparison.png")

def visualize_node_density_comparison(algorithm_data):
    """Compare algorithm performance based on graph density."""
    avg_degree = 7.30896  # From provided data
    plt.figure(figsize=(14, 10))
    for algo, data in algorithm_data.items():
        if not data["input_sizes"]:
            continue
        plt.plot(data["input_sizes"], data["execution_times"],
                 marker='o', linestyle='-', label=f"{algo}")
    
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (microseconds)')
    plt.title(f'Algorithm Performance (Graph Density: Avg Degree = {avg_degree:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig("node_density_comparison.png")
    plt.close()
    print("Generated: node_density_comparison.png")

def compare_graph_density():
    """Compare algorithm performance for sparse vs. dense graphs."""
    nodes = [100, 200, 500, 1000, 2000, 5000, 10000]
    algorithms = ["BFS", "DFS", "Dijkstra", "BellmanFord", "Prims", "Kruskal"]
    
    plt.figure(figsize=(14, 10))
    for algo in algorithms:
        # Sparse graph: E ≈ V * 7.30896
        sparse_times = generate_synthetic_data(nodes, algo, edges=None)
        plt.plot(nodes, sparse_times, marker='o', linestyle='-', label=f"{algo} (Sparse)")
        
        # Dense graph: E ≈ V^2
        dense_edges = [n**2 * 0.5 for n in nodes]  # Half for undirected
        dense_times = generate_synthetic_data(nodes, algo, edges=dense_edges)
        plt.plot(nodes, dense_times, marker='s', linestyle='--', label=f"{algo} (Dense)")
    
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Algorithm Performance: Sparse vs. Dense Graphs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("sparse_vs_dense_comparison.png")
    plt.close()
    print("Generated: sparse_vs_dense_comparison.png")

def compare_graph_direction():
    """Compare algorithm performance for directed vs. undirected graphs."""
    nodes = [100, 200, 500, 1000, 2000, 5000, 10000]
    algorithms = ["BFS", "DFS", "Dijkstra", "BellmanFord", "Prims", "Kruskal"]
    
    plt.figure(figsize=(14, 10))
    for algo in algorithms:
        # Undirected graph
        undirected_times = generate_synthetic_data(nodes, algo, directed=False)
        plt.plot(nodes, undirected_times, marker='o', linestyle='-', label=f"{algo} (Undirected)")
        
        # Directed graph
        directed_times = generate_synthetic_data(nodes, algo, directed=True)
        plt.plot(nodes, directed_times, marker='s', linestyle='--', label=f"{algo} (Directed)")
    
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Algorithm Performance: Directed vs. Undirected Graphs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("directed_vs_undirected_comparison.png")
    plt.close()
    print("Generated: directed_vs_undirected_comparison.png")

def compare_graph_weights():
    """Compare algorithm performance for weighted vs. unweighted graphs."""
    nodes = [100, 200, 500, 1000, 2000, 5000, 10000]
    algorithms = ["BFS", "DFS", "Dijkstra", "BellmanFord", "Prims", "Kruskal"]
    
    plt.figure(figsize=(14, 10))
    for algo in algorithms:
        # Unweighted graph
        unweighted_times = generate_synthetic_data(nodes, algo, weighted=False)
        plt.plot(nodes, unweighted_times, marker='o', linestyle='-', label=f"{algo} (Unweighted)")
        
        # Weighted graph
        weighted_times = generate_synthetic_data(nodes, algo, weighted=True)
        plt.plot(nodes, weighted_times, marker='s', linestyle='--', label=f"{algo} (Weighted)")
    
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Algorithm Performance: Weighted vs. Unweighted Graphs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("weighted_vs_unweighted_comparison.png")
    plt.close()
    print("Generated: weighted_vs_unweighted_comparison.png")

def main():
    print("Starting algorithm performance analysis...")
    algorithm_data = process_execution_time_data(use_synthetic=True)
    available_algorithms = list(algorithm_data.keys())
    print(f"Available algorithm data: {available_algorithms}")

    if available_algorithms:
        for algorithm in available_algorithms:
            if algorithm_data[algorithm]["input_sizes"]:
                generate_performance_graph(algorithm, f"{algorithm}_performance.png")
        
        if len(available_algorithms) > 1:
            compare_algorithms(available_algorithms, "all_algorithm_comparison.png")
            compare_algorithm_scaling(algorithm_data)
            compare_traversal_vs_singlesource(algorithm_data)
            compare_order_of_growth(algorithm_data)
            visualize_node_density_comparison(algorithm_data)
            compare_graph_density()
            compare_graph_direction()
            compare_graph_weights()

    print("Visualization complete! The following image files have been generated:")
    for alg in available_algorithms:
        print(f"- {alg}_performance.png")
    if len(available_algorithms) > 1:
        print("- all_algorithm_comparison.png")
        print("- algorithm_scaling_comparison.png")
        print("- traversal_vs_singlesource.png")
        print("- complexity_comparison.png")
        print("- node_density_comparison.png")
        print("- sparse_vs Dense_comparison.png")
        print("- directed_vs_undirected_comparison.png")
        print("- weighted_vs_unweighted_comparison.png")

if __name__ == "__main__":
    main()