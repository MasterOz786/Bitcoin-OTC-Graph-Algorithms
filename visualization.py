"""
Visualization utilities for graph algorithms.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import io
import base64

def create_graph_visualization(graph, title="Graph Visualization", max_nodes=100):
    """
    Create a visualization of the graph.
    
    Args:
        graph: The graph to visualize
        title: Title of the visualization
        max_nodes: Maximum number of nodes to visualize
    
    Returns:
        str: Base64 encoded image
    """
    # Convert to NetworkX graph for visualization
    G = nx.DiGraph() if graph.is_directed else nx.Graph()
    
    # Get all nodes and edges
    nodes = list(graph.get_all_nodes())
    
    # If there are too many nodes, sample a subset
    if len(nodes) > max_nodes:
        nodes = np.random.choice(nodes, max_nodes, replace=False)
    
    # Add nodes to NetworkX graph
    for node in nodes:
        G.add_node(node)
    
    # Add edges to NetworkX graph
    for u in nodes:
        for v in graph.get_neighbors(u):
            if v in nodes:
                weight = graph.get_weight(u, v)
                G.add_edge(u, v, weight=weight)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    plt.title(title)
    
    # Use spring layout to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
    
    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize weights for better visualization
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 2 + 0.5 if max_weight != min_weight else 1 for w in edge_weights]
    else:
        normalized_weights = []
    
    nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.7, edge_color="gray", arrows=graph.is_directed)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Draw edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Set margins and save to buffer
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def visualize_shortest_path(graph, source, distances, predecessors, title="Shortest Path Visualization", max_nodes=100):
    """
    Create a visualization of the shortest paths from a source node.
    
    Args:
        graph: The graph
        source: The source node
        distances: Shortest distances
        predecessors: Predecessors
        title: Title of the visualization
        max_nodes: Maximum number of nodes to visualize
    
    Returns:
        str: Base64 encoded image
    """
    # Convert to NetworkX graph for visualization
    G = nx.DiGraph() if graph.is_directed else nx.Graph()
    
    # Get all nodes
    nodes = list(graph.get_all_nodes())
    
    # Sort nodes by distance for better selection
    sorted_nodes = sorted([(node, distances.get(node, float('inf'))) for node in nodes], 
                          key=lambda x: x[1] if x[1] != float('inf') else float('inf'))
    
    # If there are too many nodes, take the closest ones
    if len(sorted_nodes) > max_nodes:
        selected_nodes = [source]  # Always include source
        remaining = max_nodes - 1
        
        # Add closest nodes
        for node, dist in sorted_nodes:
            if node != source and dist != float('inf') and remaining > 0:
                selected_nodes.append(node)
                remaining -= 1
    else:
        selected_nodes = [node for node, _ in sorted_nodes]
    
    # Add nodes to NetworkX graph
    for node in selected_nodes:
        G.add_node(node)
    
    # Add all edges between selected nodes
    for u in selected_nodes:
        for v in graph.get_neighbors(u):
            if v in selected_nodes:
                weight = graph.get_weight(u, v)
                G.add_edge(u, v, weight=weight)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    plt.title(title)
    
    # Use spring layout to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare shortest path edges for highlighting
    shortest_path_edges = []
    for node in selected_nodes:
        if node != source and predecessors.get(node) is not None:
            current = node
            while current != source:
                prev = predecessors.get(current)
                if prev is not None and prev in selected_nodes and current in selected_nodes:
                    # Ensure both nodes in the edge are in the graph before adding
                    if prev in G.nodes() and current in G.nodes():
                        shortest_path_edges.append((prev, current))
                current = prev
                if current is None or current == source:
                    break
    
    # Draw nodes with different colors for source
    node_colors = ['red' if node == source else 'skyblue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors)
    
    # Draw all edges with gray color
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=graph.is_directed)
    
    # Draw shortest path edges with red color
    if shortest_path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=shortest_path_edges, edge_color='red', 
                              width=2, alpha=1.0, arrows=graph.is_directed)
    
    # Draw node labels with distances
    labels = {node: f"{node}\n({distances.get(node, 'inf')})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Set margins and save to buffer
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def visualize_mst(graph, mst_edges, title="Minimum Spanning Tree", max_nodes=100):
    """
    Create a visualization of the minimum spanning tree.
    
    Args:
        graph: The graph
        mst_edges: Edges in the MST
        title: Title of the visualization
        max_nodes: Maximum number of nodes to visualize
    
    Returns:
        str: Base64 encoded image
    """
    # Create a NetworkX graph for visualization
    G = nx.Graph()  # MST is undirected
    
    # Get nodes from MST edges
    mst_nodes = set()
    for u, v, _ in mst_edges:
        mst_nodes.add(u)
        mst_nodes.add(v)
    
    # If there are too many nodes, sample a subset
    if len(mst_nodes) > max_nodes:
        mst_nodes = np.random.choice(list(mst_nodes), max_nodes, replace=False)
        # Filter edges to only include selected nodes
        mst_edges = [(u, v, w) for u, v, w in mst_edges if u in mst_nodes and v in mst_nodes]
    
    # Add nodes to NetworkX graph
    for node in mst_nodes:
        G.add_node(node)
    
    # Add MST edges to NetworkX graph
    for u, v, weight in mst_edges:
        if u in mst_nodes and v in mst_nodes:
            G.add_edge(u, v, weight=weight)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    plt.title(title)
    
    # Use spring layout to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
    
    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize weights for better visualization
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 2 + 0.5 if max_weight != min_weight else 1 for w in edge_weights]
    else:
        normalized_weights = []
    
    nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.7, edge_color="green")
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Draw edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Set margins and save to buffer
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def visualize_traversal(graph, traversal_order, title="Graph Traversal", max_nodes=100):
    """
    Create a visualization of a graph traversal.
    
    Args:
        graph: The graph
        traversal_order: Order of nodes in the traversal
        title: Title of the visualization
        max_nodes: Maximum number of nodes to visualize
    
    Returns:
        str: Base64 encoded image
    """
    # Convert to NetworkX graph for visualization
    G = nx.DiGraph() if graph.is_directed else nx.Graph()
    
    # If there are too many nodes in traversal, sample a subset
    if len(traversal_order) > max_nodes:
        selected_indices = np.linspace(0, len(traversal_order) - 1, max_nodes, dtype=int)
        traversal_order = [traversal_order[i] for i in selected_indices]
    
    # Add nodes to NetworkX graph with traversal order
    for i, node in enumerate(traversal_order):
        G.add_node(node, order=i)
    
    # Add edges between nodes in traversal order
    for i in range(len(traversal_order) - 1):
        u = traversal_order[i]
        v = traversal_order[i + 1]
        
        # Check if there's a direct edge in the original graph
        if v in graph.get_neighbors(u):
            weight = graph.get_weight(u, v)
            G.add_edge(u, v, weight=weight)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    plt.title(title)
    
    # Use spring layout to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with color indicating traversal order
    node_colors = [G.nodes[node]['order'] for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, cmap=plt.cm.viridis)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=graph.is_directed)
    
    # Draw node labels with traversal order
    labels = {node: f"{node}\n({G.nodes[node]['order']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Add colorbar to show traversal order
    plt.colorbar(nodes, label='Traversal Order')
    
    # Set margins and save to buffer
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def visualize_cycle(graph, cycle_path, title="Cycle Detection", max_nodes=100):
    """
    Create a visualization of a cycle in the graph.
    
    Args:
        graph: The graph
        cycle_path: Nodes in the cycle
        title: Title of the visualization
        max_nodes: Maximum number of nodes to visualize
    
    Returns:
        str: Base64 encoded image
    """
    # Convert to NetworkX graph for visualization
    G = nx.DiGraph() if graph.is_directed else nx.Graph()
    
    # Get all nodes in the cycle plus some neighbors
    cycle_nodes = set(cycle_path)
    extra_nodes = set()
    
    # Add some neighbors of cycle nodes
    for node in cycle_nodes:
        for neighbor in graph.get_neighbors(node):
            extra_nodes.add(neighbor)
            if len(cycle_nodes) + len(extra_nodes) >= max_nodes:
                break
    
    # If too many nodes, prioritize cycle nodes and add some extra nodes
    all_nodes = list(cycle_nodes)
    extra_nodes = list(extra_nodes - cycle_nodes)
    if len(all_nodes) + len(extra_nodes) > max_nodes:
        extra_nodes = extra_nodes[:max_nodes - len(all_nodes)]
    
    all_nodes.extend(extra_nodes)
    
    # Add nodes to NetworkX graph
    for node in all_nodes:
        G.add_node(node)
    
    # Add edges to NetworkX graph
    for u in all_nodes:
        for v in graph.get_neighbors(u):
            if v in all_nodes:
                weight = graph.get_weight(u, v)
                G.add_edge(u, v, weight=weight)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    plt.title(title)
    
    # Use spring layout to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with different colors for cycle nodes
    node_colors = ['red' if node in cycle_nodes else 'skyblue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors)
    
    # Create cycle edges
    cycle_edges = []
    for i in range(len(cycle_path) - 1):
        cycle_edges.append((cycle_path[i], cycle_path[i + 1]))
    
    # Draw all edges with gray color
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=graph.is_directed)
    
    # Draw cycle edges with red color
    if cycle_edges:
        nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color='red', 
                              width=2, alpha=1.0, arrows=graph.is_directed)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Set margins and save to buffer
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def create_performance_plot(algorithm_names, execution_times):
    """
    Create a performance comparison plot.
    
    Args:
        algorithm_names: List of algorithm names
        execution_times: List of execution times
    
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(10, 6))
    plt.bar(algorithm_names, execution_times)
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

