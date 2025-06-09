"""
Performance analysis utilities for graph algorithms.
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx
from collections import defaultdict

def create_random_graph(num_nodes, edge_probability=0.2, is_directed=True, use_negative_weights=False):
    """
    Create a random graph with the specified number of nodes.
    
    Args:
        num_nodes: Number of nodes in the graph
        edge_probability: Probability of an edge between any two nodes
        is_directed: Whether the graph is directed
        use_negative_weights: Whether to use negative weights
        
    Returns:
        A NetworkX graph
    """
    if is_directed:
        G = nx.gnp_random_graph(num_nodes, edge_probability, directed=True)
    else:
        G = nx.gnp_random_graph(num_nodes, edge_probability, directed=False)
    
    # Add weights to edges
    for u, v in G.edges():
        if use_negative_weights:
            weight = np.random.uniform(-10, 10)
        else:
            weight = np.random.uniform(1, 10)
        G[u][v]['weight'] = weight
    
    return G

def measure_algorithm_scaling(algorithm_func, graph_sizes, num_trials=3, **kwargs):
    """
    Measure how an algorithm's performance scales with graph size.
    
    Args:
        algorithm_func: Function that runs the algorithm
        graph_sizes: List of graph sizes to test
        num_trials: Number of trials for each size
        **kwargs: Additional arguments for graph creation
        
    Returns:
        pandas.DataFrame: Results with graph size and execution time
    """
    results = []
    
    for size in graph_sizes:
        total_time = 0
        for _ in range(num_trials):
            # Create random graph
            G = create_random_graph(size, **kwargs)
            
            # Measure execution time
            start_time = time.time()
            algorithm_func(G)
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        avg_time = total_time / num_trials
        results.append({'graph_size': size, 'execution_time': avg_time})
    
    return pd.DataFrame(results)

def plot_scaling_results(df, algorithm_name, log_scale=False):
    """
    Plot scaling results.
    
    Args:
        df: DataFrame with scaling results
        algorithm_name: Name of the algorithm
        log_scale: Whether to use logarithmic scale
        
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['graph_size'], df['execution_time'], 'o-', linewidth=2, markersize=8)
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Scaling of {algorithm_name} with Graph Size')
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
    else:
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def compare_algorithms(algorithm_funcs, algorithm_names, graph_size, num_trials=3, **kwargs):
    """
    Compare multiple algorithms on the same graph.
    
    Args:
        algorithm_funcs: List of algorithm functions
        algorithm_names: List of algorithm names
        graph_size: Size of the graph to test
        num_trials: Number of trials
        **kwargs: Additional arguments for graph creation
        
    Returns:
        pandas.DataFrame: Results with algorithm name and execution time
    """
    results = []
    
    # Create the same graph for all algorithms
    G = create_random_graph(graph_size, **kwargs)
    
    for func, name in zip(algorithm_funcs, algorithm_names):
        total_time = 0
        
        for _ in range(num_trials):
            # Measure execution time
            start_time = time.time()
            func(G)
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        avg_time = total_time / num_trials
        results.append({'algorithm': name, 'execution_time': avg_time})
    
    return pd.DataFrame(results)

def plot_comparison_results(df):
    """
    Plot algorithm comparison results.
    
    Args:
        df: DataFrame with comparison results
        
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df['algorithm'], df['execution_time'], color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}s',
                 ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Algorithm Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def analyze_graph_properties(algorithm_func, graph_size, num_trials=3):
    """
    Analyze how different graph properties affect algorithm performance.
    
    Args:
        algorithm_func: Function that runs the algorithm
        graph_size: Size of the graph to test
        num_trials: Number of trials
        
    Returns:
        pandas.DataFrame: Results with graph properties and execution time
    """
    results = []
    
    # Test different graph properties
    properties = [
        {"name": "Sparse Directed", "edge_probability": 0.1, "is_directed": True, "use_negative_weights": False},
        {"name": "Dense Directed", "edge_probability": 0.5, "is_directed": True, "use_negative_weights": False},
        {"name": "Sparse Undirected", "edge_probability": 0.1, "is_directed": False, "use_negative_weights": False},
        {"name": "Dense Undirected", "edge_probability": 0.5, "is_directed": False, "use_negative_weights": False},
        {"name": "Sparse Directed (Negative Weights)", "edge_probability": 0.1, "is_directed": True, "use_negative_weights": True},
        {"name": "Dense Directed (Negative Weights)", "edge_probability": 0.5, "is_directed": True, "use_negative_weights": True},
    ]
    
    for prop in properties:
        total_time = 0
        
        for _ in range(num_trials):
            # Create random graph with the specified properties
            G = create_random_graph(
                graph_size, 
                edge_probability=prop["edge_probability"],
                is_directed=prop["is_directed"],
                use_negative_weights=prop["use_negative_weights"]
            )
            
            # Measure execution time
            start_time = time.time()
            algorithm_func(G)
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        avg_time = total_time / num_trials
        results.append({'graph_type': prop["name"], 'execution_time': avg_time})
    
    return pd.DataFrame(results)

def plot_property_results(df, algorithm_name):
    """
    Plot results of graph property analysis.
    
    Args:
        df: DataFrame with property analysis results
        algorithm_name: Name of the algorithm
        
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df['graph_type'], df['execution_time'], color='lightgreen')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}s',
                 ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Graph Type')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Impact of Graph Properties on {algorithm_name} Performance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def generate_theoretical_complexity_plot(complexity_functions, labels, n_range=(10, 1000)):
    """
    Generate a plot of theoretical complexity functions.
    
    Args:
        complexity_functions: List of functions representing time complexity
        labels: List of labels for each function
        n_range: Range of n values to plot
        
    Returns:
        str: Base64 encoded image
    """
    n = np.linspace(n_range[0], n_range[1], 100)
    
    plt.figure(figsize=(10, 6))
    
    for func, label in zip(complexity_functions, labels):
        plt.plot(n, func(n), label=label, linewidth=2)
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Time Complexity')
    plt.title('Theoretical Time Complexity')
    plt.legend()
    plt.grid(True)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str
