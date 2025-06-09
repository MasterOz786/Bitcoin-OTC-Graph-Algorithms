"""
Graph implementation module for Bitcoin OTC trust network.
Includes classes for representing a graph data structure.
"""
import pandas as pd
import numpy as np
import heapq

class Graph:
    """
    Graph class to represent the Bitcoin OTC trust network.
    The graph is directed and weighted, where weights represent trust ratings.
    """
    
    def __init__(self, is_directed=True):
        """
        Initialize a graph object.
        
        Args:
            is_directed (bool): Whether the graph is directed. Default is True.
        """
        self.is_directed = is_directed
        self.adj_list = {}
        self.nodes = set()
        self.edges = []
        self.weights = {}
        self.edge_count = 0
    
    def add_node(self, node):
        """
        Add a node to the graph.
        
        Args:
            node: The node to add.
        """
        if node not in self.nodes:
            self.nodes.add(node)
            self.adj_list[node] = []
    
    def add_edge(self, source, target, weight):
        """
        Add an edge to the graph.
        
        Args:
            source: The source node.
            target: The target node.
            weight: The edge weight (trust rating).
        """
        # Add nodes if they don't exist
        self.add_node(source)
        self.add_node(target)
        
        # Add edge
        self.adj_list[source].append(target)
        self.weights[(source, target)] = weight
        self.edges.append((source, target, weight))
        self.edge_count += 1
        
        # If undirected, add reverse edge
        if not self.is_directed:
            self.adj_list[target].append(source)
            self.weights[(target, source)] = weight
    
    def get_neighbors(self, node):
        """
        Get all neighbors of a node.
        
        Args:
            node: The node to get neighbors for.
        
        Returns:
            List of neighboring nodes.
        """
        if node in self.adj_list:
            return self.adj_list[node]
        return []
    
    def get_weight(self, source, target):
        """
        Get the weight of an edge.
        
        Args:
            source: The source node.
            target: The target node.
        
        Returns:
            The weight of the edge, or None if the edge doesn't exist.
        """
        return self.weights.get((source, target), None)
    
    def get_all_edges(self):
        """
        Get all edges in the graph.
        
        Returns:
            List of tuples (source, target, weight).
        """
        return self.edges
    
    def get_all_nodes(self):
        """
        Get all nodes in the graph.
        
        Returns:
            Set of all nodes.
        """
        return self.nodes
    
    def node_count(self):
        """
        Get the number of nodes in the graph.
        
        Returns:
            Number of nodes.
        """
        return len(self.nodes)
    
    def edge_count(self):
        """
        Get the number of edges in the graph.
        
        Returns:
            Number of edges.
        """
        return self.edge_count
    
    def get_average_degree(self):
        """
        Calculate the average degree of the graph.
        
        Returns:
            The average degree.
        """
        if len(self.nodes) == 0:
            return 0
        
        total_degree = sum(len(neighbors) for neighbors in self.adj_list.values())
        return total_degree / len(self.nodes)
    
    def to_adjacency_matrix(self):
        """
        Convert the graph to an adjacency matrix.
        
        Returns:
            numpy.ndarray: The adjacency matrix.
            dict: Mapping from node to index.
        """
        # Create node to index mapping
        node_to_index = {node: i for i, node in enumerate(sorted(self.nodes))}
        index_to_node = {i: node for node, i in node_to_index.items()}
        
        # Create adjacency matrix
        n = len(self.nodes)
        adj_matrix = np.full((n, n), float('inf'))
        
        # Set diagonal elements to 0
        for i in range(n):
            adj_matrix[i, i] = 0
        
        # Fill in edge weights
        for source, target, weight in self.edges:
            adj_matrix[node_to_index[source], node_to_index[target]] = weight
        
        return adj_matrix, node_to_index, index_to_node


def load_bitcoin_otc_graph(file_path="soc-sign-bitcoinotc.csv", is_directed=True, use_abs_weights=False):
    """
    Load the Bitcoin OTC trust network dataset.
    
    Args:
        file_path (str): Path to the dataset file.
        is_directed (bool): Whether to create a directed graph.
        use_abs_weights (bool): Whether to use absolute values of ratings.
    
    Returns:
        Graph: The loaded graph.
    """
    # Create a new graph
    graph = Graph(is_directed=is_directed)
    
    # Read the dataset
    try:
        data = pd.read_csv(file_path, header=None, 
                           names=['source', 'target', 'rating', 'time'])
        
        # Add edges to the graph
        for _, row in data.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            rating = abs(row['rating']) if use_abs_weights else row['rating']
            
            # For algorithms like Dijkstra, we need positive weights
            # We need to invert the rating to represent "distance"
            # Higher rating means more trust, which should be a shorter distance
            # So we use the negative of rating or a large number minus the rating
            weight = 10 - rating if rating > 0 else 10 + abs(rating)
            
            graph.add_edge(source, target, weight)
        
        print(f"Loaded graph with {graph.node_count()} nodes and {graph.edge_count} edges")
        return graph
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

