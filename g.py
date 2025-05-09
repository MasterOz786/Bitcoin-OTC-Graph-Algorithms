import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('soc.csv', header=None, names=['source', 'target', 'rating', 'time'])

# Create an undirected graph for MST algorithms
G = nx.Graph()

# Add edges with rating as an attribute
for _, row in df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['rating'])

# Prim's Algorithm for Minimum Spanning Tree (MST)
def prim_mst(graph):
    mst = nx.minimum_spanning_tree(graph, algorithm='prim')
    return mst

# Kruskal's Algorithm for Minimum Spanning Tree (MST)
def kruskal_mst(graph):
    mst = nx.minimum_spanning_tree(graph, algorithm='kruskal')
    return mst

# Dijkstra's Algorithm (Shortest Path)
def dijkstra_shortest_path(graph, source):
    # Compute shortest paths from the source node
    shortest_paths = nx.single_source_dijkstra_path(graph, source)
    return shortest_paths

# Bellman-Ford Algorithm (Shortest Path with Negative Weights)
def bellman_ford_shortest_path(graph, source):
    # Compute shortest paths from the source node, handles negative weights
    length, path = nx.single_source_bellman_ford(graph, source)
    return length, path

# Perform BFS Algorithm
def bfs(graph, start_node):
    visited = set()  # To keep track of visited nodes
    queue = [start_node]  # Queue for BFS
    bfs_order = []  # List to store the BFS traversal order

    while queue:
        node = queue.pop(0)  # Pop the first node in the queue
        if node not in visited:
            visited.add(node)
            bfs_order.append(node)

            # Add all unvisited neighbors to the queue
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return bfs_order

# Perform DFS Algorithm
def dfs(graph, start_node):
    visited = set()  # To keep track of visited nodes
    dfs_order = []  # List to store the DFS traversal order

    def dfs_recursive(node):
        visited.add(node)
        dfs_order.append(node)
        
        # Visit all unvisited neighbors
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs_recursive(neighbor)

    # Start DFS from the given node
    dfs_recursive(start_node)
    
    return dfs_order

# Perform BFS starting from node 1
bfs_result = bfs(G, 1)
print("BFS Traversal Order:", bfs_result)

# Perform DFS starting from node 1
dfs_result = dfs(G, 1)
print("DFS Traversal Order:", dfs_result)

# Visualize the graph with BFS and DFS order highlighted
plt.figure(figsize=(15, 15))

# Generate layout for visualization
pos = nx.spring_layout(G)

# Draw the graph nodes
node_color = ['skyblue' if node not in dfs_result else 'lightgreen' for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_color, alpha=0.7)

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.5)

# Draw the labels
nx.draw_networkx_labels(G, pos, font_size=8)

# Title for the visualization
plt.title(f'Graph Visualization with DFS Starting from Node 1\nDFS Order: {dfs_result[:20]}...', fontsize=14)

# Label the BFS and DFS
plt.annotate('BFS Traversal', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12, color='red')
plt.annotate('DFS Traversal', xy=(0.1, 0.85), xycoords='axes fraction', fontsize=12, color='green')

# Show the graph
plt.axis('off')
plt.show()

# Apply and visualize MST using Prim's Algorithm
prim_mst_graph = prim_mst(G)
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(prim_mst_graph)
nx.draw(prim_mst_graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=8, edge_color='green')
plt.title("Minimum Spanning Tree (Prim's Algorithm)")

# Label the Prim's MST visualization
plt.annotate('Prim\'s MST', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12, color='blue')

plt.show()

# Apply and visualize MST using Kruskal's Algorithm
kruskal_mst_graph = kruskal_mst(G)
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(kruskal_mst_graph)
nx.draw(kruskal_mst_graph, pos, with_labels=True, node_size=700, node_color='lightcoral', font_size=8, edge_color='red')
plt.title("Minimum Spanning Tree (Kruskal's Algorithm)")

# Label the Kruskal's MST visualization
plt.annotate('Kruskal\'s MST', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12, color='blue')

plt.show()

# Dijkstra's Shortest Path (from node 1)
# dijkstra_paths = dijkstra_shortest_path(G, 1)
# print("Dijkstra's Shortest Path/ from Node 1:", dijkstra_paths)

# Visualize Dijkstra's shortest path
# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, edge_color='grey')
# for target, path in dijkstra_paths.items():
#     path_edges = list(zip(path[:-1], path[1:]))
#     nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=2)

# plt.title("Dijkstra's Shortest Path (From Node 1)")
# plt.annotate("Dijkstra's Shortest Path", xy=(0.1, 0.9), xycoords='axes fraction', fontsize=12, color='orange')

# plt.show()


# Bellman-Ford Algorithm with Negative Cycle Detection
def bellman_ford_with_cycle_detection(graph, start):
    distance = {node: float('inf') for node in graph.nodes}
    predecessor = {node: None for node in graph.nodes}
    distance[start] = 0

    # Relax edges repeatedly
    for _ in range(len(graph.nodes) - 1):
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u

    # Check for negative cycles
    cycle_edges = []
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        if distance[u] + weight < distance[v]:
            cycle_edges.append((u, v))
    
    return distance, predecessor, cycle_edges

# Run Bellman-Ford
start_node = 1
distances, predecessors, cycle_edges = bellman_ford_with_cycle_detection(G, start_node)

# Plot the graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, seed=42)  # Better reproducibility

# Color edges: red for cycle, gray otherwise
edge_colors = ['red' if (u, v) in cycle_edges else 'gray' for u, v in G.edges()]

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300, alpha=0.8)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, alpha=0.6)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=7)

# Edge weight labels
edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

# Final display
title = "⚠️ Negative Cycle Detected" if cycle_edges else "✅ Graph: Bellman-Ford - No Negative Cycle"
plt.title(title, fontsize=14)
plt.axis('off')
plt.show()