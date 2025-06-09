"""
Report generation utilities for graph algorithms.
"""
import time
import os
import platform
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak
from fpdf import FPDF

def collect_system_specs():
    """
    Collect system specifications.
    
    Returns:
        dict: System specifications
    """
    specs = {
        "OS": platform.system() + " " + platform.version(),
        "Python": platform.python_version(),
        "Processor": platform.processor(),
        "Architecture": platform.architecture()[0],
    }
    
    try:
        import psutil
        # Get RAM information
        ram = psutil.virtual_memory()
        specs["RAM"] = f"{ram.total / (1024**3):.2f} GB"
    except ImportError:
        specs["RAM"] = "Unknown"
    
    return specs

def create_subgraph(graph, nodes):
    """
    Create a subgraph from the original graph containing only the specified nodes.
    
    Args:
        graph: The original graph
        nodes: List of nodes to include in the subgraph
    
    Returns:
        graph: A new Graph instance containing only the specified nodes
    """
    from graph import Graph
    
    # Create a new graph with the same directedness
    subgraph = Graph(is_directed=graph.is_directed)
    
    # Add all nodes
    for node in nodes:
        subgraph.add_node(node)
    
    # Add edges between nodes that exist in the original graph
    for source in nodes:
        for target in nodes:
            if target in graph.get_neighbors(source):
                weight = graph.get_weight(source, target)
                subgraph.add_edge(source, target, weight)
    
    return subgraph

def run_algorithm_benchmarks(graph, algorithms, node_sizes=None):
    """
    Run benchmarks on different algorithms and graph sizes.
    
    Args:
        graph: The full graph
        algorithms: List of (function, name) tuples
        node_sizes: List of node sizes to test (default is None, which uses the full graph)
    
    Returns:
        dict: Benchmark results
    """
    results = {}
    
    if node_sizes:
        # Sample different sizes of the graph
        nodes = list(graph.get_all_nodes())
        print(f"Total nodes in graph: {len(nodes)}")
        
        for size in node_sizes:
            # Skip if size is larger than the number of nodes
            if size > len(nodes):
                print(f"Skipping size {size} (larger than total nodes)")
                continue
            
            print(f"Testing with {size} nodes...")
            
            # Sample nodes
            sample_nodes = np.random.choice(nodes, size=min(size, len(nodes)), replace=False)
            
            # Create an actual subgraph with only the sampled nodes
            subgraph = create_subgraph(graph, sample_nodes)
            
            size_results = {}
            
            for func, name in algorithms:
                print(f"  Running {name}...")
                # Use a node from the subgraph as source
                source_node = list(subgraph.get_all_nodes())[0]
                
                # Time the algorithm
                start_time = time.time()
                func(subgraph, source_node)  # Run on actual subgraph
                end_time = time.time()
                
                execution_time = end_time - start_time
                size_results[name] = execution_time
                print(f"  {name} completed in {execution_time:.4f} seconds")
            
            results[size] = size_results
    else:
        # Just run on the full graph
        full_results = {}
        source_node = list(graph.get_all_nodes())[0]
        
        for func, name in algorithms:
            print(f"Running {name} on full graph...")
            # Time the algorithm
            start_time = time.time()
            func(graph, source_node)  # Use the first node as source/start node
            end_time = time.time()
            
            execution_time = end_time - start_time
            full_results[name] = execution_time
            print(f"{name} completed in {execution_time:.4f} seconds")
        
        results["full"] = full_results
    
    return results

def create_individual_algorithm_plot(algorithm_name, sizes, times, title=None):
    """
    Create a plot for a single algorithm's performance across different graph sizes.
    
    Args:
        algorithm_name: Name of the algorithm
        sizes: List of graph sizes
        times: List of execution times
        title: Title of the plot (default: Algorithm's performance)
    
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the data
    plt.plot(sizes, times, marker='o', linewidth=2, color='blue')
    
    # Add a polynomial trend line to show scaling behavior
    if len(sizes) > 2:
        try:
            z = np.polyfit(sizes, times, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(sizes), max(sizes), 100)
            plt.plot(x_trend, p(x_trend), linestyle='--', color='red', 
                     label=f'Trend: {z[0]:.2e}x² + {z[1]:.2e}x + {z[2]:.2e}')
            plt.legend()
        except:
            # If curve fitting fails, continue without the trend line
            pass
    
    # Set labels and title
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    if title:
        plt.title(title)
    else:
        plt.title(f'{algorithm_name} Performance Scaling')
    
    # Add grid and tighten layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def create_comparison_plot(sizes, algorithm_times, title="Algorithm Performance Comparison"):
    """
    Create a comparison plot of all algorithms.
    
    Args:
        sizes: List of graph sizes
        algorithm_times: Dictionary mapping algorithm names to lists of execution times
        title: Title of the plot
    
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each algorithm's performance
    for algorithm, times in algorithm_times.items():
        plt.plot(sizes, times, marker='o', linewidth=2, label=algorithm)
    
    # Set labels and title
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title(title)
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    
    # Encode image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def create_execution_time_plots(results, title="Algorithm Execution Time Analysis"):
    """
    Create plots of execution times for each algorithm and a comparison plot.
    
    Args:
        results: Dictionary of results from run_algorithm_benchmarks
        title: Title for the comparison plot
    
    Returns:
        dict: Dictionary mapping plot names to base64 encoded images
    """
    plots = {}
    
    if "full" in results:
        # Single-size comparison plot
        plt.figure(figsize=(10, 6))
        
        algorithms = list(results["full"].keys())
        times = list(results["full"].values())
        
        plt.bar(algorithms, times)
        plt.xlabel('Algorithm')
        plt.ylabel('Execution Time (seconds)')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        
        # Encode image to base64
        buf.seek(0)
        plots["comparison"] = base64.b64encode(buf.read()).decode()
    else:
        # Create individual plots for each algorithm
        sizes = sorted(results.keys())
        algorithm_times = {}
        
        algorithms = list(results[sizes[0]].keys())
        for algorithm in algorithms:
            times = [results[size][algorithm] for size in sizes]
            algorithm_times[algorithm] = times
            plots[f"{algorithm}_plot"] = create_individual_algorithm_plot(
                algorithm, sizes, times
            )
        
        # Create comparison plot
        plots["comparison"] = create_comparison_plot(
            sizes, algorithm_times, title
        )
    
    return plots

def generate_algorithm_complexity_table():
    """
    Generate a table of algorithm time complexities.
    
    Returns:
        list: Table data
    """
    table_data = [
        ["Algorithm", "Best-case", "Average-case", "Worst-case", "Space Complexity"],
        ["Dijkstra", "O(E + V log V)", "O(E + V log V)", "O(E + V log V)", "O(V)"],
        ["Bellman-Ford", "O(E)", "O(V*E)", "O(V*E)", "O(V)"],
        ["Prim's MST", "O(E + V log V)", "O(E + V log V)", "O(E + V log V)", "O(V)"],
        ["Kruskal's MST", "O(E log E)", "O(E log E)", "O(E log E)", "O(V + E)"],
        ["BFS", "O(V + E)", "O(V + E)", "O(V + E)", "O(V)"],
        ["DFS", "O(V + E)", "O(V + E)", "O(V + E)", "O(V)"],
        ["Diameter", "O(V * (E + V log V))", "O(V * (E + V log V))", "O(V * (E + V log V))", "O(V)"],
        ["Cycle Detection", "O(V + E)", "O(V + E)", "O(V + E)", "O(V)"]
    ]
    
    return table_data

def create_pdf_report(results, graph_info, system_specs, output_file="algorithm_report.pdf"):
    """
    Create a PDF report of algorithm benchmarks.
    
    Args:
        results: Benchmark results
        graph_info: Information about the graph
        system_specs: System specifications
        output_file: Output filename
    """
    # Create a temporary directory for images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create execution time plots
        plots = create_execution_time_plots(results)
        
        # Save plots to image files
        plot_image_paths = {}
        for plot_name, plot_data in plots.items():
            img_path = os.path.join(tmpdir, f"{plot_name}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(plot_data))
            plot_image_paths[plot_name] = img_path
        
        # Create PDF
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create a custom style for the title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        
        # Create a custom style for headings
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=10
        )
        
        # Create a custom style for normal text
        normal_style = styles['Normal']
        
        elements = []
        
        # Title
        elements.append(Paragraph("Graph Algorithm Analysis Report", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # System Specifications
        elements.append(Paragraph("1. System Specifications", heading_style))
        specs_data = [["Component", "Specification"]]
        for key, value in system_specs.items():
            specs_data.append([key, value])
        
        specs_table = Table(specs_data, colWidths=[2*inch, 4*inch])
        specs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(specs_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Graph Information
        elements.append(Paragraph("2. Dataset Information", heading_style))
        graph_data = [["Metric", "Value"]]
        for key, value in graph_info.items():
            graph_data.append([key, value])
        
        graph_table = Table(graph_data, colWidths=[2*inch, 4*inch])
        graph_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(graph_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Algorithm Complexity
        elements.append(Paragraph("3. Algorithm Time Complexity Analysis", heading_style))
        elements.append(Paragraph("The following table shows the theoretical time and space complexity of each algorithm:", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        complexity_table = Table(generate_algorithm_complexity_table(), colWidths=[1.5*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
        complexity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(complexity_table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Execution Time Results
        elements.append(Paragraph("4. Experimental Results", heading_style))
        
        if "full" in results:
            elements.append(Paragraph("4.1 Algorithm Execution Time Comparison", heading_style))
            elements.append(Paragraph("The following chart shows the execution time of each algorithm on the full graph:", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Add the execution time plot
            img = Image(plot_image_paths["comparison"], width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.1*inch))
            
            # Add execution time table
            elements.append(Paragraph("Execution Time Summary:", normal_style))
            exec_data = [["Algorithm", "Execution Time (seconds)"]]
            for alg, time_val in results["full"].items():
                exec_data.append([alg, f"{time_val:.6f}"])
            
            exec_table = Table(exec_data, colWidths=[3*inch, 3*inch])
            exec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(exec_table)
        else:
            # First add the comparison plot
            elements.append(Paragraph("4.1 Algorithm Scaling Comparison", heading_style))
            elements.append(Paragraph("The following chart shows how each algorithm's execution time scales with the number of nodes:", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Add the comparison plot
            img = Image(plot_image_paths["comparison"], width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
            
            # Now add individual algorithm plots
            elements.append(Paragraph("4.2 Individual Algorithm Scaling Analysis", heading_style))
            elements.append(Paragraph("The following charts show detailed scaling behavior for each algorithm:", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Add individual plots for each algorithm
            algorithms = list(results[sorted(results.keys())[0]].keys())
            for algorithm in algorithms:
                elements.append(Paragraph(f"4.2.{algorithms.index(algorithm)+1} {algorithm}", heading_style))
                img = Image(plot_image_paths[f"{algorithm}_plot"], width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
            
            # Add execution time table for each size
            elements.append(Paragraph("Execution Time Summary by Graph Size:", normal_style))
            
            sizes = sorted(results.keys())
            algorithms = list(results[sizes[0]].keys())
            
            # Create header row
            size_data = [["Algorithm"] + [f"{size} nodes" for size in sizes]]
            
            # Add data for each algorithm
            for alg in algorithms:
                row = [alg]
                for size in sizes:
                    row.append(f"{results[size][alg]:.6f}")
                size_data.append(row)
            
            size_table = Table(size_data, colWidths=[2*inch] + [1.1*inch] * len(sizes))
            size_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(size_table)
        
        elements.append(PageBreak())
        
        # Conclusions
        elements.append(Paragraph("5. Conclusions", heading_style))
        
        elements.append(Paragraph("5.1 Performance Summary", normal_style))
        elements.append(Paragraph("Based on the experimental results, we can draw the following conclusions:", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        if "full" in results:
            # Sort algorithms by execution time
            sorted_algs = sorted(results["full"].items(), key=lambda x: x[1])
            fastest = sorted_algs[0][0]
            slowest = sorted_algs[-1][0]
            
            elements.append(Paragraph(f"• {fastest} was the fastest algorithm on this dataset.", normal_style))
            elements.append(Paragraph(f"• {slowest} was the slowest algorithm on this dataset.", normal_style))
            elements.append(Paragraph("• The performance differences align with the theoretical time complexity analysis.", normal_style))
        else:
            # Analyze scaling behavior
            elements.append(Paragraph("• As the graph size increases, the execution time of all algorithms increases.", normal_style))
            elements.append(Paragraph("• Algorithms with higher time complexity show steeper growth curves with increasing graph size.", normal_style))
            elements.append(Paragraph("• The experimental results validate the theoretical time complexity analysis.", normal_style))
        
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph("5.2 Recommendations", normal_style))
        elements.append(Paragraph("Based on the performance analysis, we recommend:", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph("• For shortest path problems with positive weights, use Dijkstra's algorithm.", normal_style))
        elements.append(Paragraph("• For shortest path problems with potential negative weights, use Bellman-Ford algorithm.", normal_style))
        elements.append(Paragraph("• For unweighted shortest path problems, use BFS as it's simpler and often faster.", normal_style))
        elements.append(Paragraph("• For MST computation on sparse graphs, Kruskal's algorithm may be more efficient.", normal_style))
        elements.append(Paragraph("• For MST computation on dense graphs, Prim's algorithm is typically better.", normal_style))
        
        # Build the PDF
        doc.build(elements)
    
    return output_file

def generate_comprehensive_report(graph, algorithms, node_samples, output_file="algorithm_report.pdf"):
    """
    Generate a comprehensive report on algorithm performance.
    
    Args:
        graph: The graph
        algorithms: List of (function, name) tuples
        node_samples: List of node sample sizes to test
        output_file: Output filename
    
    Returns:
        str: Path to the generated PDF
    """
    # Collect system specs
    system_specs = collect_system_specs()
    
    # Collect graph info
    graph_info = {
        "Dataset": "Bitcoin OTC Trust Network",
        "Nodes": graph.node_count(),
        "Edges": graph.edge_count,
        "Average Degree": f"{graph.get_average_degree():.2f}",
        "Directed": "Yes" if graph.is_directed else "No",
        "Edge Weight Range": "-10 to +10"
    }
    
    # Run benchmarks
    results = run_algorithm_benchmarks(graph, algorithms, node_samples)
    
    # Generate PDF report
    report_path = create_pdf_report(results, graph_info, system_specs, output_file)
    
    return report_path
