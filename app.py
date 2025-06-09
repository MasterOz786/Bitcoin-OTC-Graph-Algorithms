"""
Streamlit web application for graph algorithm implementation and analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import base64
from io import BytesIO
import tempfile

# Import custom modules
from graph import Graph, load_bitcoin_otc_graph
from algorithms import (
    dijkstra, bellman_ford, prims, kruskals, bfs, dfs, 
    find_diameter, detect_cycle, AlgorithmTracer
)
from utils import (
    ensure_directory, save_shortest_path_results, save_mst_results,
    save_traversal_results, save_diameter_results, save_cycle_results,
    save_average_degree_results, save_execution_times
)
from visualization import (
    create_graph_visualization, visualize_shortest_path, visualize_mst,
    visualize_traversal, visualize_cycle, create_performance_plot
)

# Set page configuration
st.set_page_config(
    page_title="Graph Algorithm Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create output directories
ensure_directory("results")
ensure_directory("traces")

# Main app title
st.title("Graph Algorithm Implementation and Analysis")
st.subheader("Bitcoin OTC Trust Network Dataset")

# Sidebar for dataset and algorithm selection
st.sidebar.header("Dataset Configuration")

# Dataset upload option
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV dataset file", 
    type=["csv"],
    help="Upload a graph dataset in CSV format with columns: source, target, weight/rating, timestamp"
)

# Alternatively, allow using a local file path
st.sidebar.subheader("Or Use Local Path")
dataset_file = st.sidebar.text_input(
    "Dataset File Path", 
    value="soc-sign-bitcoinotc.csv", 
    help="Path to the Bitcoin OTC dataset CSV file"
)

use_abs_weights = st.sidebar.checkbox(
    "Use Absolute Trust Values", 
    value=True,
    help="If checked, negative trust values will be treated as positive"
)

is_directed = st.sidebar.checkbox(
    "Directed Graph", 
    value=True, 
    help="If checked, the graph will be treated as directed"
)

# Load the graph when the user clicks the button
if st.sidebar.button("Load Graph"):
    with st.spinner("Loading graph from dataset..."):
        # Check if file was uploaded
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            # Load graph from the temporary file
            graph = load_bitcoin_otc_graph(temp_path, is_directed, use_abs_weights)
            
            # Remove temporary file
            os.unlink(temp_path)
        else:
            # Load graph from file path
            graph = load_bitcoin_otc_graph(dataset_file, is_directed, use_abs_weights)
        
        if graph:
            st.session_state["graph"] = graph
            st.success(f"Graph loaded successfully with {graph.node_count()} nodes and {graph.edge_count} edges")
        else:
            st.error("Failed to load graph. Please upload a valid dataset file or check the file path and try again.")

# Import the report generation module at the top level
from report_generator import generate_comprehensive_report

# Algorithm selection
st.sidebar.header("Algorithm Selection")

# Add a performance comparison section to the sidebar
st.sidebar.subheader("Performance Analysis")
show_performance = st.sidebar.checkbox(
    "Enable Performance Analysis",
    value=False,
    help="Run algorithms on different graph sizes to compare performance"
)

# Add custom performance analysis section
st.sidebar.subheader("Custom Performance Analysis")
if "graph" in st.session_state:
    # Allow user to select algorithms for benchmarking
    st.sidebar.markdown("**Select Algorithms to Benchmark:**")
    benchmark_dijkstra = st.sidebar.checkbox("Dijkstra", value=True)
    benchmark_bellman = st.sidebar.checkbox("Bellman-Ford", value=True)
    benchmark_bfs = st.sidebar.checkbox("BFS", value=True) 
    benchmark_dfs = st.sidebar.checkbox("DFS", value=True)
    
    # Create a container for custom node sizes
    st.sidebar.markdown("**Enter Custom Node Sizes:**")
    st.sidebar.markdown("Enter comma-separated values, e.g., '500, 1000, 1500'")
    custom_sizes_input = st.sidebar.text_input("Node sizes", value="500, 1000, 1500")
    
    # Number of times to run each algorithm for each size
    st.sidebar.markdown("**Execution Settings:**")
    repeat_runs = st.sidebar.number_input("Repeat each test (times)", min_value=1, value=3)
    
    # Button to run the custom analysis
    if st.sidebar.button("Run Custom Performance Analysis"):
        # Parse the input node sizes
        try:
            custom_node_sizes = [int(size.strip()) for size in custom_sizes_input.split(",")]
            
            # Create list of selected algorithms
            algorithms_to_benchmark = []
            if benchmark_dijkstra:
                algorithms_to_benchmark.append((lambda g, s: dijkstra(g, s)[:-1], "Dijkstra"))
            if benchmark_bellman:
                algorithms_to_benchmark.append((lambda g, s: bellman_ford(g, s)[:-2], "Bellman-Ford"))
            if benchmark_bfs:
                algorithms_to_benchmark.append((lambda g, s: bfs(g, s)[:-1], "BFS"))
            if benchmark_dfs:
                algorithms_to_benchmark.append((lambda g, s: dfs(g, s)[:-1], "DFS"))
            
            if not algorithms_to_benchmark:
                st.sidebar.error("Please select at least one algorithm.")
            else:
                st.session_state['custom_benchmark_results'] = {}
                st.session_state['custom_benchmark_node_sizes'] = custom_node_sizes
                st.session_state['custom_benchmark_algorithms'] = [name for _, name in algorithms_to_benchmark]
                
                # Run the benchmarks
                for size in custom_node_sizes:
                    st.session_state['custom_benchmark_results'][size] = {}
                    
                    # Skip if size is larger than available nodes
                    if size > st.session_state["graph"].node_count():
                        st.sidebar.warning(f"Skipping size {size} (exceeds total nodes)")
                        continue
                    
                    # Run benchmarks for each algorithm with multiple trials
                    for func, name in algorithms_to_benchmark:
                        # Run multiple times and take average
                        total_time = 0
                        times = []
                        
                        for i in range(repeat_runs):
                            # Sample nodes
                            nodes = list(st.session_state["graph"].get_all_nodes())
                            sample_nodes = np.random.choice(nodes, size=size, replace=False)
                            
                            # Create a subgraph
                            from report_generator import create_subgraph
                            subgraph = create_subgraph(st.session_state["graph"], sample_nodes)
                            
                            # Use a node from the subgraph as source
                            source_node = list(subgraph.get_all_nodes())[0]
                            
                            # Time the algorithm
                            start_time = time.time()
                            func(subgraph, source_node)
                            end_time = time.time()
                            
                            exec_time = end_time - start_time
                            times.append(exec_time)
                            total_time += exec_time
                        
                        # Save average and individual times
                        avg_time = total_time / repeat_runs
                        st.session_state['custom_benchmark_results'][size][name] = {
                            'avg': avg_time,
                            'runs': times
                        }
                
                st.sidebar.success("Benchmark completed! See results below.")
                st.session_state["show_custom_benchmark"] = True
                
        except ValueError:
            st.sidebar.error("Invalid node sizes. Please enter comma-separated integers.")

# Add comprehensive report generation section
st.sidebar.subheader("Report Generation")
if st.sidebar.button("Generate Comprehensive Report"):
    if "graph" in st.session_state:
        with st.spinner("Generating comprehensive report..."):
            # Define algorithms to benchmark
            algorithms_to_benchmark = [
                (lambda g, s: dijkstra(g, s)[:-1], "Dijkstra"),
                (lambda g, s: bellman_ford(g, s)[:-2], "Bellman-Ford"),
                (lambda g, s: bfs(g, s)[:-1], "BFS"),
                (lambda g, s: dfs(g, s)[:-1], "DFS"),
            ]
            
            # Define more reasonable node samples based on the actual graph size
            max_nodes = st.session_state["graph"].node_count()
            node_samples = []
            if max_nodes > 100:
                step = max(int(max_nodes / 10), 50)
                for i in range(1, 11):
                    size = i * step
                    if size > max_nodes:
                        break
                    node_samples.append(size)
            
            if not node_samples:
                node_samples = [max_nodes // 2, max_nodes]
            
            # Generate the report
            report_path = generate_comprehensive_report(
                st.session_state["graph"],
                algorithms_to_benchmark,
                node_samples,
                output_file="results/algorithm_report.pdf"
            )
            
            st.sidebar.success("Report generated successfully!")
            
            # Read the PDF file and create a download button
            with open(report_path, "rb") as f:
                pdf_bytes = f.read()
            
            st.sidebar.download_button(
                label="Download Report",
                data=pdf_bytes,
                file_name="algorithm_report.pdf",
                mime="application/pdf"
            )
    else:
        st.sidebar.error("Please load the graph first.")

if show_performance and "graph" in st.session_state:
    st.sidebar.subheader("Analysis Settings")
    performance_option = st.sidebar.radio(
        "Analysis type",
        ["Algorithm Comparison", "Scaling with Graph Size", "Impact of Graph Properties"]
    )

algorithm_option = st.sidebar.selectbox(
    "Select Algorithm",
    [
        "Single Source Shortest Path (Dijkstra)",
        "Single Source Shortest Path (Bellman-Ford)",
        "Minimum Spanning Tree (Prim's)",
        "Minimum Spanning Tree (Kruskal's)",
        "Breadth-First Search (BFS)",
        "Depth-First Search (DFS)",
        "Diameter of Graph",
        "Cycle Detection",
        "Average Degree",
        "Algorithm Performance Comparison"
    ]
)

# Import the performance analysis module
from performance_analysis import (
    measure_algorithm_scaling, plot_scaling_results,
    compare_algorithms, plot_comparison_results,
    analyze_graph_properties, plot_property_results,
    generate_theoretical_complexity_plot
)

# This line can be deleted since we're already importing at the top level

# Main content
if "graph" not in st.session_state:
    st.info("Please load the graph from the sidebar to get started.")
    
    # Display information about the Bitcoin OTC dataset
    st.subheader("About the Bitcoin OTC Trust Network Dataset")
    st.write("""
    The Bitcoin OTC trust network dataset represents ratings that users of the Bitcoin OTC platform gave to each other. 
    Each rating is from one user to another, with a weight between -10 and 10, where:
    - Negative values represent distrust (the lower the value, the higher the distrust)
    - Positive values represent trust (the higher the value, the higher the trust)
    
    In this application, we'll use these trust ratings as edge weights for various graph algorithms.
    For algorithms requiring positive weights, we'll transform the ratings appropriately.
    """)
    
    st.subheader("Implementation Requirements")
    st.write("""
    This application implements various graph algorithms as per the requirements:
    
    1. **Single Source Shortest Path**: Using both Dijkstra and Bellman-Ford algorithms
    2. **Minimum Spanning Tree**: Using both Prim's and Kruskal's algorithms
    3. **Graph Traversal**: Breadth-First Search (BFS) and Depth-First Search (DFS)
    4. **Graph Diameter**: Finding the maximum shortest path distance between any two nodes
    5. **Cycle Detection**: Detecting cycles in the graph
    6. **Average Degree**: Calculating the average degree of nodes in the graph
    
    For each algorithm:
    - Complete execution traces are saved in the 'traces' directory
    - Results are stored in the 'results' directory
    - Execution times are measured and compared
    - Visualizations are provided for better understanding
    """)
else:
    graph = st.session_state["graph"]
    
    # Display general graph information
    st.subheader("Graph Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Nodes", graph.node_count())
    with col2:
        st.metric("Number of Edges", graph.edge_count)
    with col3:
        st.metric("Average Degree", f"{graph.get_average_degree():.2f}")
    
    # Show custom benchmark results if available
    if "show_custom_benchmark" in st.session_state and st.session_state["show_custom_benchmark"]:
        st.subheader("Custom Performance Analysis Results")
        
        # Plot individual algorithm results
        st.write("### Individual Algorithm Scaling")
        
        # Create tabs for each algorithm
        tabs = st.tabs(st.session_state['custom_benchmark_algorithms'])
        
        for i, alg_name in enumerate(st.session_state['custom_benchmark_algorithms']):
            with tabs[i]:
                # Extract data for this algorithm
                sizes = []
                avg_times = []
                
                for size in sorted(st.session_state['custom_benchmark_node_sizes']):
                    if size in st.session_state['custom_benchmark_results'] and alg_name in st.session_state['custom_benchmark_results'][size]:
                        sizes.append(size)
                        avg_times.append(st.session_state['custom_benchmark_results'][size][alg_name]['avg'])
                
                if sizes:
                    # Create figure for the algorithm
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(sizes, avg_times, 'o-', linewidth=2, label=f"Average Time")
                    
                    # Add error bars for individual runs
                    all_runs = []
                    for size in sizes:
                        runs = st.session_state['custom_benchmark_results'][size][alg_name]['runs']
                        all_runs.append(runs)
                    
                    if all(len(runs) > 1 for runs in all_runs):
                        # Calculate error bars if we have multiple runs
                        yerr = [np.std(runs) for runs in all_runs]
                        ax.errorbar(sizes, avg_times, yerr=yerr, fmt='o', alpha=0.5, capsize=5)
                    
                    # Add a trend line if we have enough data points
                    if len(sizes) > 2:
                        try:
                            z = np.polyfit(sizes, avg_times, 2)
                            p = np.poly1d(z)
                            x_trend = np.linspace(min(sizes), max(sizes), 100)
                            ax.plot(x_trend, p(x_trend), '--', color='red', 
                                    label=f'Trend: {z[0]:.2e}x² + {z[1]:.2e}x + {z[2]:.2e}')
                        except:
                            pass
                    
                    ax.set_xlabel('Number of Nodes')
                    ax.set_ylabel('Execution Time (seconds)')
                    ax.set_title(f'{alg_name} Scaling Performance')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Display the actual data in a table
                    st.write("### Data Table")
                    
                    # Create a DataFrame for the results
                    data = []
                    for i, size in enumerate(sizes):
                        runs = st.session_state['custom_benchmark_results'][size][alg_name]['runs']
                        avg = avg_times[i]
                        
                        run_data = {f"Run {j+1}": time for j, time in enumerate(runs)}
                        row_data = {"Nodes": size, "Average Time (s)": avg, **run_data}
                        data.append(row_data)
                    
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    # Add download button for CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {alg_name} Data as CSV",
                        data=csv,
                        file_name=f"{alg_name.lower().replace(' ', '_')}_scaling_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.write(f"No data available for {alg_name}.")
        
        # Create a comparison chart with all algorithms
        st.write("### Algorithm Comparison")
        
        # Get all sizes that have at least one algorithm result
        all_sizes = sorted(st.session_state['custom_benchmark_node_sizes'])
        valid_sizes = [size for size in all_sizes if size in st.session_state['custom_benchmark_results']]
        
        if valid_sizes:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for alg_name in st.session_state['custom_benchmark_algorithms']:
                sizes = []
                times = []
                
                for size in valid_sizes:
                    if alg_name in st.session_state['custom_benchmark_results'][size]:
                        sizes.append(size)
                        times.append(st.session_state['custom_benchmark_results'][size][alg_name]['avg'])
                
                if sizes:
                    ax.plot(sizes, times, 'o-', linewidth=2, label=alg_name)
            
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Algorithm Performance Comparison')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)
            
            # Create a comparison table
            st.write("### Comparison Table")
            
            comparison_data = []
            for size in valid_sizes:
                row = {"Nodes": size}
                for alg_name in st.session_state['custom_benchmark_algorithms']:
                    if alg_name in st.session_state['custom_benchmark_results'][size]:
                        row[alg_name] = st.session_state['custom_benchmark_results'][size][alg_name]['avg']
                    else:
                        row[alg_name] = None
                comparison_data.append(row)
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df)
                
                # Add download button for CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Comparison Data as CSV",
                    data=csv,
                    file_name="algorithm_comparison_data.csv",
                    mime="text/csv"
                )
        else:
            st.write("No valid data available for comparison.")
    
    # Graph visualization - always show by default
    st.subheader("Graph Visualization")
    with st.spinner("Generating graph visualization..."):
        viz_img = create_graph_visualization(graph, "Bitcoin OTC Trust Network (Sample)")
        st.image(f"data:image/png;base64,{viz_img}", use_container_width=True)
        st.caption("Note: Only a subset of nodes is shown for visualization clarity")
    
    # Execute selected algorithm
    st.subheader(f"Algorithm: {algorithm_option}")
    
    if "Single Source Shortest Path" in algorithm_option:
        # Source node selection
        sorted_nodes = sorted(graph.get_all_nodes())
        source_node = st.selectbox("Select source node", sorted_nodes)
        
        if st.button("Run Algorithm"):
            with st.spinner("Running algorithm..."):
                if "Dijkstra" in algorithm_option:
                    # Run Dijkstra's algorithm
                    tracer = AlgorithmTracer("Dijkstra")
                    distances, predecessors, execution_time = dijkstra(graph, source_node, tracer)
                    
                    # Save results and trace
                    save_shortest_path_results(distances, predecessors, source_node, "results/dijkstra_results.txt")
                    tracer.save_trace("traces/dijkstra_trace.txt")
                    
                    # Display results
                    st.success(f"Dijkstra's algorithm completed in {execution_time:.6f} seconds")
                    
                    # Visualization
                    viz_img = visualize_shortest_path(
                        graph, source_node, distances, predecessors, 
                        f"Dijkstra's Shortest Paths from Node {source_node}"
                    )
                    st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                    
                    # Display results and trace
                    if st.checkbox("Show Detailed Results", value=False):
                        with open("results/dijkstra_results.txt", "r") as f:
                            st.text(f.read())
                    
                    if st.checkbox("Show Algorithm Trace", value=False):
                        with open("traces/dijkstra_trace.txt", "r") as f:
                            st.text(f.read())
                
                else:  # Bellman-Ford
                    # Run Bellman-Ford algorithm
                    tracer = AlgorithmTracer("Bellman-Ford")
                    distances, predecessors, execution_time, negative_cycle = bellman_ford(graph, source_node, tracer)
                    
                    # Save results and trace
                    save_shortest_path_results(distances, predecessors, source_node, "results/bellman_ford_results.txt")
                    tracer.save_trace("traces/bellman_ford_trace.txt")
                    
                    # Display results
                    if negative_cycle:
                        st.warning("Negative cycle detected in the graph")
                    
                    st.success(f"Bellman-Ford algorithm completed in {execution_time:.6f} seconds")
                    
                    # Visualization
                    viz_img = visualize_shortest_path(
                        graph, source_node, distances, predecessors, 
                        f"Bellman-Ford Shortest Paths from Node {source_node}"
                    )
                    st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                    
                    # Display results and trace
                    if st.checkbox("Show Detailed Results", value=False):
                        with open("results/bellman_ford_results.txt", "r") as f:
                            st.text(f.read())
                    
                    if st.checkbox("Show Algorithm Trace", value=False):
                        with open("traces/bellman_ford_trace.txt", "r") as f:
                            st.text(f.read())
    
    elif "Minimum Spanning Tree" in algorithm_option:
        if st.button("Run Algorithm"):
            with st.spinner("Running algorithm..."):
                if "Prim's" in algorithm_option:
                    # Run Prim's algorithm
                    tracer = AlgorithmTracer("Prim's")
                    
                    # If graph is directed, show warning as MST is typically for undirected graphs
                    if graph.is_directed:
                        st.warning("Prim's algorithm is typically used for undirected graphs. Treating as undirected.")
                    
                    # For MST, we can choose any start node
                    start_node = list(graph.get_all_nodes())[0]
                    mst_edges, total_weight, execution_time = prims(graph, start_node, tracer)
                    
                    # Save results and trace
                    save_mst_results(mst_edges, total_weight, "results/prims_results.txt")
                    tracer.save_trace("traces/prims_trace.txt")
                    
                    # Display results
                    st.success(f"Prim's algorithm completed in {execution_time:.6f} seconds")
                    st.metric("MST Total Weight", f"{total_weight:.2f}")
                    
                    # Visualization
                    viz_img = visualize_mst(graph, mst_edges, "Minimum Spanning Tree (Prim's Algorithm)")
                    st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                    
                    # Display results and trace
                    if st.checkbox("Show Detailed Results", value=False):
                        with open("results/prims_results.txt", "r") as f:
                            st.text(f.read())
                    
                    if st.checkbox("Show Algorithm Trace", value=False):
                        with open("traces/prims_trace.txt", "r") as f:
                            st.text(f.read())
                
                else:  # Kruskal's
                    # Run Kruskal's algorithm
                    tracer = AlgorithmTracer("Kruskal's")
                    
                    # If graph is directed, show warning as MST is typically for undirected graphs
                    if graph.is_directed:
                        st.warning("Kruskal's algorithm is typically used for undirected graphs. Treating as undirected.")
                    
                    mst_edges, total_weight, execution_time = kruskals(graph, tracer)
                    
                    # Save results and trace
                    save_mst_results(mst_edges, total_weight, "results/kruskals_results.txt")
                    tracer.save_trace("traces/kruskals_trace.txt")
                    
                    # Display results
                    st.success(f"Kruskal's algorithm completed in {execution_time:.6f} seconds")
                    st.metric("MST Total Weight", f"{total_weight:.2f}")
                    
                    # Visualization
                    viz_img = visualize_mst(graph, mst_edges, "Minimum Spanning Tree (Kruskal's Algorithm)")
                    st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                    
                    # Display results and trace
                    if st.checkbox("Show Detailed Results", value=False):
                        with open("results/kruskals_results.txt", "r") as f:
                            st.text(f.read())
                    
                    if st.checkbox("Show Algorithm Trace", value=False):
                        with open("traces/kruskals_trace.txt", "r") as f:
                            st.text(f.read())
    
    elif "First Search" in algorithm_option:  # BFS or DFS
        # Source node selection
        sorted_nodes = sorted(graph.get_all_nodes())
        start_node = st.selectbox("Select start node", sorted_nodes)
        
        if st.button("Run Algorithm"):
            with st.spinner("Running algorithm..."):
                if "Breadth-First" in algorithm_option:
                    # Run BFS
                    tracer = AlgorithmTracer("BFS")
                    traversal_order, distances, execution_time = bfs(graph, start_node, tracer)
                    
                    # Save results and trace
                    save_traversal_results(traversal_order, distances, start_node, "results/bfs_results.txt")
                    tracer.save_trace("traces/bfs_trace.txt")
                    
                    # Display results
                    st.success(f"BFS completed in {execution_time:.6f} seconds")
                    st.write(f"Number of nodes visited: {len(traversal_order)}")
                    
                    # Visualization
                    viz_img = visualize_traversal(graph, traversal_order, f"BFS Traversal from Node {start_node}")
                    st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                    
                    # Display results and trace
                    if st.checkbox("Show Detailed Results", value=False):
                        with open("results/bfs_results.txt", "r") as f:
                            st.text(f.read())
                    
                    if st.checkbox("Show Algorithm Trace", value=False):
                        with open("traces/bfs_trace.txt", "r") as f:
                            st.text(f.read())
                
                else:  # DFS
                    # Run DFS
                    tracer = AlgorithmTracer("DFS")
                    traversal_order, times, execution_time = dfs(graph, start_node, tracer)
                    
                    # Save results and trace
                    save_traversal_results(traversal_order, times, start_node, "results/dfs_results.txt")
                    tracer.save_trace("traces/dfs_trace.txt")
                    
                    # Display results
                    st.success(f"DFS completed in {execution_time:.6f} seconds")
                    st.write(f"Number of nodes visited: {len(traversal_order)}")
                    
                    # Visualization
                    viz_img = visualize_traversal(graph, traversal_order, f"DFS Traversal from Node {start_node}")
                    st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                    
                    # Display results and trace
                    if st.checkbox("Show Detailed Results", value=False):
                        with open("results/dfs_results.txt", "r") as f:
                            st.text(f.read())
                    
                    if st.checkbox("Show Algorithm Trace", value=False):
                        with open("traces/dfs_trace.txt", "r") as f:
                            st.text(f.read())
    
    elif "Diameter" in algorithm_option:
        if st.button("Run Algorithm"):
            with st.spinner("Running algorithm..."):
                # Run diameter calculation
                tracer = AlgorithmTracer("Diameter")
                diameter, nodes, execution_time = find_diameter(graph, tracer)
                
                # Save results and trace
                save_diameter_results(diameter, nodes, "results/diameter_results.txt")
                tracer.save_trace("traces/diameter_trace.txt")
                
                # Display results
                st.success(f"Diameter calculation completed in {execution_time:.6f} seconds")
                st.metric("Diameter", f"{diameter:.2f}")
                st.write(f"Diameter is achieved between nodes {nodes[0]} and {nodes[1]}")
                
                # Display results and trace
                if st.checkbox("Show Detailed Results", value=False):
                    with open("results/diameter_results.txt", "r") as f:
                        st.text(f.read())
                
                if st.checkbox("Show Algorithm Trace", value=False):
                    with open("traces/diameter_trace.txt", "r") as f:
                        st.text(f.read())
    
    elif "Cycle Detection" in algorithm_option:
        if st.button("Run Algorithm"):
            with st.spinner("Running algorithm..."):
                # Run cycle detection
                tracer = AlgorithmTracer("Cycle Detection")
                cycle_exists, cycle_path, execution_time = detect_cycle(graph, tracer)
                
                # Save results and trace
                save_cycle_results(cycle_exists, cycle_path, "results/cycle_results.txt")
                tracer.save_trace("traces/cycle_trace.txt")
                
                # Display results
                st.success(f"Cycle detection completed in {execution_time:.6f} seconds")
                
                if cycle_exists:
                    st.write("Cycle detected!")
                    st.write(f"Cycle: {' -> '.join(map(str, cycle_path))}")
                    
                    # Visualization
                    viz_img = visualize_cycle(graph, cycle_path, "Cycle in Graph")
                    st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                else:
                    st.write("No cycle detected in the graph.")
                
                # Display results and trace
                if st.checkbox("Show Detailed Results", value=False):
                    with open("results/cycle_results.txt", "r") as f:
                        st.text(f.read())
                
                if st.checkbox("Show Algorithm Trace", value=False):
                    with open("traces/cycle_trace.txt", "r") as f:
                        st.text(f.read())
    
    elif "Average Degree" in algorithm_option:
        if st.button("Calculate Average Degree"):
            with st.spinner("Calculating average degree..."):
                # Calculate average degree
                avg_degree = graph.get_average_degree()
                
                # Save results
                save_average_degree_results(avg_degree, "results/avg_degree_results.txt")
                
                # Display results
                st.success("Average degree calculation completed")
                st.metric("Average Degree", f"{avg_degree:.2f}")
                
                # Display detailed results
                if st.checkbox("Show Detailed Results", value=False):
                    with open("results/avg_degree_results.txt", "r") as f:
                        st.text(f.read())
    
    elif "Performance Comparison" in algorithm_option:
        if st.button("Run Performance Comparison"):
            with st.spinner("Running performance comparison..."):
                # Dictionary to store execution times
                execution_times = {}
                
                # Select a sample of nodes for testing
                all_nodes = list(graph.get_all_nodes())
                sample_node = all_nodes[0]
                
                # Run Dijkstra
                tracer = AlgorithmTracer("Dijkstra")
                _, _, execution_time = dijkstra(graph, sample_node, tracer)
                execution_times["Dijkstra"] = execution_time
                
                # Run Bellman-Ford
                tracer = AlgorithmTracer("Bellman-Ford")
                _, _, execution_time, _ = bellman_ford(graph, sample_node, tracer)
                execution_times["Bellman-Ford"] = execution_time
                
                # Run Prim's
                tracer = AlgorithmTracer("Prim's")
                _, _, execution_time = prims(graph, sample_node, tracer)
                execution_times["Prim's"] = execution_time
                
                # Run Kruskal's
                tracer = AlgorithmTracer("Kruskal's")
                _, _, execution_time = kruskals(graph, tracer)
                execution_times["Kruskal's"] = execution_time
                
                # Run BFS
                tracer = AlgorithmTracer("BFS")
                _, _, execution_time = bfs(graph, sample_node, tracer)
                execution_times["BFS"] = execution_time
                
                # Run DFS
                tracer = AlgorithmTracer("DFS")
                _, _, execution_time = dfs(graph, sample_node, tracer)
                execution_times["DFS"] = execution_time
                
                # Run Diameter
                tracer = AlgorithmTracer("Diameter")
                _, _, execution_time = find_diameter(graph, tracer)
                execution_times["Diameter"] = execution_time
                
                # Run Cycle Detection
                tracer = AlgorithmTracer("Cycle Detection")
                _, _, execution_time = detect_cycle(graph, tracer)
                execution_times["Cycle Detection"] = execution_time
                
                # Save execution times
                save_execution_times(execution_times, "results/execution_times.txt")
                
                # Display results
                st.success("Performance comparison completed")
                
                # Create chart
                algorithm_names = list(execution_times.keys())
                times = list(execution_times.values())
                
                # Sort by execution time
                sorted_indices = np.argsort(times)
                sorted_algorithms = [algorithm_names[i] for i in sorted_indices]
                sorted_times = [times[i] for i in sorted_indices]
                
                # Display chart
                viz_img = create_performance_plot(sorted_algorithms, sorted_times)
                st.image(f"data:image/png;base64,{viz_img}", use_column_width=True)
                
                # Display detailed results
                st.subheader("Execution Times")
                for algorithm, time_value in sorted(execution_times.items(), key=lambda x: x[1]):
                    st.write(f"{algorithm}: {time_value:.6f} seconds")
                
                if st.checkbox("Show Detailed Results", value=False):
                    with open("results/execution_times.txt", "r") as f:
                        st.text(f.read())

# Footer information
st.markdown("---")
st.caption("Graph Algorithm Implementation and Analysis Tool for Bitcoin OTC Trust Network")

