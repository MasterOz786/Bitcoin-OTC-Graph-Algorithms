"""
Implementation of graph algorithms for the Bitcoin OTC trust network.
"""
import heapq
import time
import numpy as np
from collections import deque

class AlgorithmTracer:
    """
    Utility class for tracing algorithm execution.
    """
    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.trace = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer and trace"""
        self.start_time = time.time()
        self.trace = [f"Starting {self.algorithm_name} algorithm..."]
    
    def add(self, message):
        """Add a message to the trace"""
        self.trace.append(message)
    
    def end(self):
        """End the timer and trace"""
        self.end_time = time.time()
        self.trace.append(f"Completed {self.algorithm_name} algorithm.")
        execution_time = self.end_time - self.start_time
        self.trace.append(f"Execution time: {execution_time:.6f} seconds")
        return execution_time
    
    def save_trace(self, filename):
        """Save the trace to a file"""
        with open(filename, 'w') as f:
            for line in self.trace:
                f.write(line + "\n")

# Single Source Shortest Path Algorithms
def dijkstra(graph, source, tracer=None):
    """
    Dijkstra's algorithm for single source shortest path.
    
    Args:
        graph: The graph
        source: The source node
        tracer: The algorithm tracer
    
    Returns:
        dict: Shortest distances
        dict: Predecessors
        float: Execution time
    """
    if tracer is None:
        tracer = AlgorithmTracer("Dijkstra")
    
    tracer.start()
    tracer.add(f"Source node: {source}")
    
    # Initialize distances and predecessors
    distances = {node: float('inf') for node in graph.get_all_nodes()}
    distances[source] = 0
    predecessors = {node: None for node in graph.get_all_nodes()}
    
    # Initialize priority queue
    priority_queue = [(0, source)]
    tracer.add(f"Initial priority queue: {priority_queue}")
    
    # Set to keep track of visited nodes
    visited = set()
    
    while priority_queue:
        # Get node with minimum distance
        current_distance, current_node = heapq.heappop(priority_queue)
        tracer.add(f"Dequeued: ({current_distance}, {current_node})")
        
        # If already visited, skip
        if current_node in visited:
            tracer.add(f"Node {current_node} already visited, skipping")
            continue
        
        # Mark as visited
        visited.add(current_node)
        tracer.add(f"Visiting node {current_node}")
        
        # If all nodes are visited, break
        if len(visited) == len(graph.get_all_nodes()):
            tracer.add("All nodes visited, breaking")
            break
        
        # Relaxation step
        for neighbor in graph.get_neighbors(current_node):
            weight = graph.get_weight(current_node, neighbor)
            
            if weight is None:
                tracer.add(f"No edge from {current_node} to {neighbor}")
                continue
            
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                tracer.add(f"Relaxing edge ({current_node}, {neighbor}): {distances[neighbor]} -> {distance}")
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
                tracer.add(f"Updated priority queue: {priority_queue}")
            else:
                tracer.add(f"No relaxation needed for edge ({current_node}, {neighbor})")
    
    execution_time = tracer.end()
    
    # Create a summary of the results
    tracer.add("\nShortest Path Results:")
    for node in sorted(graph.get_all_nodes()):
        if node == source:
            tracer.add(f"Node {node}: Distance = 0 (source)")
        elif distances[node] == float('inf'):
            tracer.add(f"Node {node}: Unreachable")
        else:
            path = []
            current = node
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            tracer.add(f"Node {node}: Distance = {distances[node]}, Path = {' -> '.join(map(str, path))}")
    
    return distances, predecessors, execution_time

def bellman_ford(graph, source, tracer=None):
    """
    Bellman-Ford algorithm for single source shortest path.
    
    Args:
        graph: The graph
        source: The source node
        tracer: The algorithm tracer
    
    Returns:
        dict: Shortest distances
        dict: Predecessors
        float: Execution time
        bool: Whether negative cycle exists
    """
    if tracer is None:
        tracer = AlgorithmTracer("Bellman-Ford")
    
    tracer.start()
    tracer.add(f"Source node: {source}")
    
    # Initialize distances and predecessors
    distances = {node: float('inf') for node in graph.get_all_nodes()}
    distances[source] = 0
    predecessors = {node: None for node in graph.get_all_nodes()}
    
    # Get all edges
    edges = graph.get_all_edges()
    tracer.add(f"Number of edges: {len(edges)}")
    
    # Relax all edges |V| - 1 times
    V = len(graph.get_all_nodes())
    tracer.add(f"Number of nodes: {V}")
    
    for i in range(V - 1):
        tracer.add(f"\nIteration {i+1}:")
        relaxed = False
        
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                tracer.add(f"Relaxing edge ({u}, {v}): {distances[v]} -> {distances[u] + weight}")
                distances[v] = distances[u] + weight
                predecessors[v] = u
                relaxed = True
            else:
                tracer.add(f"No relaxation needed for edge ({u}, {v})")
        
        if not relaxed:
            tracer.add(f"No edges relaxed in iteration {i+1}, breaking early")
            break
    
    # Check for negative cycles
    tracer.add("\nChecking for negative cycles:")
    negative_cycle = False
    
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            tracer.add(f"Negative cycle detected at edge ({u}, {v})")
            negative_cycle = True
            break
    
    if not negative_cycle:
        tracer.add("No negative cycles detected")
    
    execution_time = tracer.end()
    
    # Create a summary of the results
    tracer.add("\nShortest Path Results:")
    for node in sorted(graph.get_all_nodes()):
        if node == source:
            tracer.add(f"Node {node}: Distance = 0 (source)")
        elif distances[node] == float('inf'):
            tracer.add(f"Node {node}: Unreachable")
        else:
            path = []
            current = node
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            tracer.add(f"Node {node}: Distance = {distances[node]}, Path = {' -> '.join(map(str, path))}")
    
    return distances, predecessors, execution_time, negative_cycle

# Minimum Spanning Tree Algorithms
def prims(graph, start_node=None, tracer=None):
    """
    Prim's algorithm for minimum spanning tree.
    
    Args:
        graph: The graph
        start_node: The starting node
        tracer: The algorithm tracer
    
    Returns:
        list: Edges in the MST
        float: Total weight of MST
        float: Execution time
    """
    if tracer is None:
        tracer = AlgorithmTracer("Prim's Algorithm")
    
    tracer.start()
    
    # If the graph is empty
    if graph.node_count() == 0:
        tracer.add("Graph is empty")
        execution_time = tracer.end()
        return [], 0, execution_time
    
    # Get all nodes
    nodes = list(graph.get_all_nodes())
    
    # Choose a starting node if not provided
    if start_node is None:
        start_node = nodes[0]
    
    tracer.add(f"Starting node: {start_node}")
    
    # Initialize data structures
    mst_edges = []
    included = {node: False for node in nodes}
    key = {node: float('inf') for node in nodes}
    parent = {node: None for node in nodes}
    
    # Start with the starting node
    key[start_node] = 0
    pq = [(0, start_node)]
    tracer.add(f"Initial priority queue: {pq}")
    
    while pq:
        # Get node with minimum key
        min_key, u = heapq.heappop(pq)
        tracer.add(f"Dequeued: ({min_key}, {u})")
        
        # If already included, skip
        if included[u]:
            tracer.add(f"Node {u} already included, skipping")
            continue
        
        # Include the node
        included[u] = True
        tracer.add(f"Including node {u}")
        
        # If not the starting node, add the edge to MST
        if parent[u] is not None:
            mst_edges.append((parent[u], u, graph.get_weight(parent[u], u)))
            tracer.add(f"Added edge ({parent[u]}, {u}) to MST")
        
        # Update keys of adjacent nodes
        for v in graph.get_neighbors(u):
            weight = graph.get_weight(u, v)
            
            if not included[v] and weight < key[v]:
                parent[v] = u
                key[v] = weight
                heapq.heappush(pq, (weight, v))
                tracer.add(f"Updated key of node {v} to {weight}, parent to {u}")
                tracer.add(f"Updated priority queue: {pq}")
    
    # Calculate total weight
    total_weight = sum(edge[2] for edge in mst_edges)
    tracer.add(f"Total MST weight: {total_weight}")
    
    execution_time = tracer.end()
    
    return mst_edges, total_weight, execution_time

def find(parent, i):
    """Find operation for Union-Find data structure."""
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    """Union operation for Union-Find data structure."""
    root_x = find(parent, x)
    root_y = find(parent, y)
    
    if root_x == root_y:
        return
    
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1

def kruskals(graph, tracer=None):
    """
    Kruskal's algorithm for minimum spanning tree.
    
    Args:
        graph: The graph
        tracer: The algorithm tracer
    
    Returns:
        list: Edges in the MST
        float: Total weight of MST
        float: Execution time
    """
    if tracer is None:
        tracer = AlgorithmTracer("Kruskal's Algorithm")
    
    tracer.start()
    
    # Get all edges and sort by weight
    edges = graph.get_all_edges()
    edges.sort(key=lambda x: x[2])
    tracer.add(f"Sorted edges: {edges}")
    
    # Initialize Union-Find data structure
    nodes = list(graph.get_all_nodes())
    parent = {node: node for node in nodes}
    rank = {node: 0 for node in nodes}
    
    # Initialize MST
    mst_edges = []
    
    # Process edges in order of increasing weight
    for u, v, weight in edges:
        tracer.add(f"Processing edge ({u}, {v}) with weight {weight}")
        
        root_u = find(parent, u)
        root_v = find(parent, v)
        
        # If including this edge doesn't create a cycle
        if root_u != root_v:
            tracer.add(f"Adding edge ({u}, {v}) to MST")
            mst_edges.append((u, v, weight))
            union(parent, rank, root_u, root_v)
        else:
            tracer.add(f"Edge ({u}, {v}) would create a cycle, skipping")
    
    # Calculate total weight
    total_weight = sum(edge[2] for edge in mst_edges)
    tracer.add(f"Total MST weight: {total_weight}")
    
    execution_time = tracer.end()
    
    return mst_edges, total_weight, execution_time

# Graph Traversal Algorithms
def bfs(graph, start_node, tracer=None):
    """
    Breadth-first search traversal.
    
    Args:
        graph: The graph
        start_node: The starting node
        tracer: The algorithm tracer
    
    Returns:
        list: BFS traversal order
        dict: Distances from start_node
        float: Execution time
    """
    if tracer is None:
        tracer = AlgorithmTracer("BFS")
    
    tracer.start()
    tracer.add(f"Starting node: {start_node}")
    
    # Initialize data structures
    visited = set()
    distances = {node: float('inf') for node in graph.get_all_nodes()}
    distances[start_node] = 0
    queue = deque([start_node])
    traversal_order = []
    
    tracer.add(f"Initial queue: {list(queue)}")
    
    while queue:
        # Dequeue a node
        node = queue.popleft()
        tracer.add(f"Dequeued: {node}")
        
        # If already visited, skip
        if node in visited:
            tracer.add(f"Node {node} already visited, skipping")
            continue
        
        # Visit the node
        visited.add(node)
        traversal_order.append(node)
        tracer.add(f"Visiting node {node}")
        
        # Visit neighbors
        for neighbor in graph.get_neighbors(node):
            if neighbor not in visited:
                tracer.add(f"Enqueuing neighbor {neighbor}")
                queue.append(neighbor)
                
                # Update distance if not already set
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[node] + 1
                    tracer.add(f"Setting distance of {neighbor} to {distances[neighbor]}")
        
        tracer.add(f"Current queue: {list(queue)}")
    
    execution_time = tracer.end()
    
    # Create a summary
    tracer.add("\nBFS Traversal Results:")
    tracer.add(f"Traversal order: {traversal_order}")
    tracer.add("\nDistances from start node:")
    for node in sorted(graph.get_all_nodes()):
        if node == start_node:
            tracer.add(f"Node {node}: Distance = 0 (start)")
        elif distances[node] == float('inf'):
            tracer.add(f"Node {node}: Unreachable")
        else:
            tracer.add(f"Node {node}: Distance = {distances[node]}")
    
    return traversal_order, distances, execution_time

def dfs(graph, start_node, tracer=None):
    """
    Depth-first search traversal.
    
    Args:
        graph: The graph
        start_node: The starting node
        tracer: The algorithm tracer
    
    Returns:
        list: DFS traversal order
        dict: Discovery and finish times
        float: Execution time
    """
    if tracer is None:
        tracer = AlgorithmTracer("DFS")
    
    tracer.start()
    tracer.add(f"Starting node: {start_node}")
    
    # Initialize data structures
    visited = set()
    traversal_order = []
    discovery_time = {node: -1 for node in graph.get_all_nodes()}
    finish_time = {node: -1 for node in graph.get_all_nodes()}
    time = [0]  # Use a list to allow modification in nested function
    
    # Iterative implementation to avoid recursion limit with large graphs
    stack = [(start_node, True)]  # (node, is_discovery)
    parent = {}  # Keep track of parent nodes for backtracking
    
    # Start DFS from start_node
    tracer.add(f"Starting DFS from node {start_node}")
    
    while stack:
        node, is_discovery = stack.pop()
        
        if is_discovery:  # First time visiting this node
            if node not in visited:
                # Mark node as visited
                visited.add(node)
                traversal_order.append(node)
                time[0] += 1
                discovery_time[node] = time[0]
                tracer.add(f"Visiting node {node}, discovery time: {discovery_time[node]}")
                
                # Add finish event to the stack (will be processed after all descendants)
                stack.append((node, False))
                
                # Visit neighbors in reverse order (to match recursive DFS)
                neighbors = list(graph.get_neighbors(node))
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        tracer.add(f"Visiting neighbor {neighbor} of node {node}")
                        stack.append((neighbor, True))
                        parent[neighbor] = node
                    else:
                        tracer.add(f"Neighbor {neighbor} of node {node} already visited")
        else:  # Finishing the node
            # Mark node as finished
            time[0] += 1
            finish_time[node] = time[0]
            tracer.add(f"Finished node {node}, finish time: {finish_time[node]}")
    
    # Check for unvisited nodes
    for node in graph.get_all_nodes():
        if node not in visited:
            tracer.add(f"Node {node} not reachable from start node")
    
    execution_time = tracer.end()
    
    # Create a summary
    tracer.add("\nDFS Traversal Results:")
    tracer.add(f"Traversal order: {traversal_order}")
    tracer.add("\nDiscovery and finish times:")
    for node in sorted(graph.get_all_nodes()):
        if discovery_time[node] == -1:
            tracer.add(f"Node {node}: Unreachable")
        else:
            tracer.add(f"Node {node}: Discovery time = {discovery_time[node]}, Finish time = {finish_time[node]}")
    
    return traversal_order, (discovery_time, finish_time), execution_time

# Diameter Algorithm
def find_diameter(graph, tracer=None):
    """
    Find the diameter of the graph.
    
    Args:
        graph: The graph
        tracer: The algorithm tracer
    
    Returns:
        float: Diameter of the graph
        tuple: Nodes that achieve the diameter
        float: Execution time
    """
    if tracer is None:
        tracer = AlgorithmTracer("Diameter")
    
    tracer.start()
    
    # Get the adjacency matrix
    adj_matrix, node_to_index, index_to_node = graph.to_adjacency_matrix()
    n = len(adj_matrix)
    
    tracer.add(f"Number of nodes: {n}")
    
    # Run Floyd-Warshall algorithm
    tracer.add("Running Floyd-Warshall algorithm to find all pairs shortest paths")
    
    for k in range(n):
        if k % 100 == 0:  # Log progress for large graphs
            tracer.add(f"Processing intermediate node {k}")
        
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, k] != float('inf') and adj_matrix[k, j] != float('inf'):
                    adj_matrix[i, j] = min(adj_matrix[i, j], adj_matrix[i, k] + adj_matrix[k, j])
    
    # Find the diameter (maximum finite distance)
    diameter = -1
    u, v = None, None
    
    for i in range(n):
        for j in range(n):
            if i != j and adj_matrix[i, j] != float('inf') and adj_matrix[i, j] > diameter:
                diameter = adj_matrix[i, j]
                u, v = index_to_node[i], index_to_node[j]
    
    tracer.add(f"Diameter: {diameter}")
    tracer.add(f"Achieved between nodes {u} and {v}")
    
    execution_time = tracer.end()
    
    return diameter, (u, v), execution_time

# Cycle Detection Algorithm
def detect_cycle(graph, tracer=None):
    """
    Detect cycles in the graph.
    
    Args:
        graph: The graph
        tracer: The algorithm tracer
    
    Returns:
        bool: Whether a cycle exists
        list: A cycle, if one exists
        float: Execution time
    """
    if tracer is None:
        tracer = AlgorithmTracer("Cycle Detection")
    
    tracer.start()
    
    # Initialize data structures
    all_nodes = list(graph.get_all_nodes())
    visited = set()
    rec_stack = set()
    cycle_path = []
    
    def dfs_cycle(node, parent=None):
        # Mark node as visited
        visited.add(node)
        rec_stack.add(node)
        tracer.add(f"Visiting node {node}, recursion stack: {rec_stack}")
        
        # Visit neighbors
        for neighbor in graph.get_neighbors(node):
            tracer.add(f"Checking neighbor {neighbor} of node {node}")
            
            # If not visited, continue DFS
            if neighbor not in visited:
                tracer.add(f"Neighbor {neighbor} not visited, continuing DFS")
                if dfs_cycle(neighbor, node):
                    cycle_path.insert(0, neighbor)
                    return True
            
            # If visited and in recursion stack, cycle found
            elif neighbor in rec_stack:
                tracer.add(f"Cycle found: node {node} has an edge to node {neighbor} which is in recursion stack")
                cycle_path.append(node)
                cycle_path.append(neighbor)
                return True
        
        # Remove from recursion stack
        rec_stack.remove(node)
        tracer.add(f"Removing node {node} from recursion stack")
        return False
    
    # Run DFS from each unvisited node
    cycle_exists = False
    
    for node in all_nodes:
        if node not in visited:
            tracer.add(f"Starting DFS from node {node}")
            if dfs_cycle(node):
                cycle_exists = True
                break
    
    if cycle_exists:
        tracer.add(f"Cycle detected: {cycle_path}")
    else:
        tracer.add("No cycle detected")
    
    execution_time = tracer.end()
    
    return cycle_exists, cycle_path, execution_time

