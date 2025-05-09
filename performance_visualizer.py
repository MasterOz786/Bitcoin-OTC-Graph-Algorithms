
import matplotlib.pyplot as plt
import re
import glob
import os
import numpy as np

def parse_execution_time_file(file_path):
    """Parse an execution time file to extract input sizes and execution times"""
    input_sizes = []
    execution_times = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            size_match = re.search(r'Input size: (\d+)', lines[i])
            time_match = re.search(r'Execution time: (\d+)', lines[i+1])
            
            if size_match and time_match:
                input_size = int(size_match.group(1))
                execution_time = int(time_match.group(1))
                
                input_sizes.append(input_size)
                execution_times.append(execution_time)
    
    return input_sizes, execution_times

def generate_performance_graph(algorithm_name, output_filename=None):
    """Generate a performance graph for a specific algorithm"""
    # Find all execution time files for this algorithm
    pattern = f"{algorithm_name}_ExecutionTime_*.txt"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No execution time files found for {algorithm_name}")
        return
    
    all_input_sizes = []
    all_execution_times = []
    
    # Parse all files and collect data
    for file_path in files:
        input_sizes, execution_times = parse_execution_time_file(file_path)
        all_input_sizes.extend(input_sizes)
        all_execution_times.extend(execution_times)
    
    # Sort data by input size
    sorted_data = sorted(zip(all_input_sizes, all_execution_times))
    if not sorted_data:
        print(f"No data found in {algorithm_name} execution time files")
        return
        
    sorted_input_sizes, sorted_execution_times = zip(*sorted_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_input_sizes, sorted_execution_times, marker='o', linestyle='-')
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Input Size (number of nodes)')
    plt.ylabel('Execution Time (microseconds)')
    plt.title(f'Graph of Execution Time of {algorithm_name} Algorithm')
    
    # Add annotations for a few points
    # Choose points to annotate - every quarter of the data
    if len(sorted_input_sizes) > 4:
        indices = [len(sorted_input_sizes) // 4, 
                   len(sorted_input_sizes) // 2, 
                   3 * len(sorted_input_sizes) // 4]
        
        for i in indices:
            plt.annotate(f'{sorted_execution_times[i]}',
                         xy=(sorted_input_sizes[i], sorted_execution_times[i]),
                         xytext=(10, 10), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                         bbox=dict(boxstyle='round,pad=0.3', fc='#e0f0ff', ec='gray', alpha=0.8))
    
    # Save the plot if output filename is provided
    if output_filename:
        plt.savefig(output_filename)
        print(f"Graph saved to {output_filename}")
    
    plt.tight_layout()
    plt.show()

def compare_algorithms(algorithms, output_filename=None):
    """Compare the performance of multiple algorithms"""
    plt.figure(figsize=(14, 10))
    
    for algorithm in algorithms:
        # Find execution time files for this algorithm
        pattern = f"{algorithm}_ExecutionTime_*.txt"
        files = glob.glob(pattern)
        
        if not files:
            print(f"No execution time files found for {algorithm}")
            continue
        
        all_input_sizes = []
        all_execution_times = []
        
        # Parse all files and collect data
        for file_path in files:
            input_sizes, execution_times = parse_execution_time_file(file_path)
            all_input_sizes.extend(input_sizes)
            all_execution_times.extend(execution_times)
        
        # Sort data by input size
        sorted_data = sorted(zip(all_input_sizes, all_execution_times))
        if not sorted_data:
            print(f"No data found in {algorithm} execution time files")
            continue
            
        sorted_input_sizes, sorted_execution_times = zip(*sorted_data)
        
        # Plot this algorithm's data
        plt.plot(sorted_input_sizes, sorted_execution_times, marker='o', linestyle='-', label=algorithm)
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels, title and legend
    plt.xlabel('Input Size (number of nodes)')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Comparison of Algorithm Execution Times')
    plt.legend()
    
    # Save the plot if output filename is provided
    if output_filename:
        plt.savefig(output_filename)
        print(f"Comparison graph saved to {output_filename}")
    
    plt.tight_layout()
    plt.show()

def generate_synthetic_data(algorithm_name, min_size=1000, max_size=100000, step=5000, 
                            time_function=None, output_file=None):
    """Generate synthetic data for testing with different input sizes"""
    if time_function is None:
        # Default time function: O(n log n) with some randomness
        time_function = lambda n: n * np.log(n) * (1 + 0.2 * np.random.random())
    
    input_sizes = list(range(min_size, max_size + step, step))
    execution_times = [int(time_function(n)) for n in input_sizes]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(input_sizes, execution_times, marker='o', linestyle='-')
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Input Size (k = 1000)')
    plt.ylabel('Running Time (milliseconds)')
    plt.title(f'Graph of Execution time of {algorithm_name}')
    
    # Add x-axis labels with k notation
    plt.xticks([i for i in input_sizes if i % 10000 == 0], 
               [f"{i//1000}k" for i in input_sizes if i % 10000 == 0])
    
    # Add annotations for a few points
    indices = [0, len(input_sizes) // 2, -1]  # First, middle, last
    for i in indices:
        plt.annotate(f'{execution_times[i]}',
                     xy=(input_sizes[i], execution_times[i]),
                     xytext=(10, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                     bbox=dict(boxstyle='round,pad=0.3', fc='#e0f0ff', ec='gray', alpha=0.8))
    
    # Save the plot if output filename is provided
    if output_file:
        plt.savefig(output_file)
        print(f"Synthetic data graph saved to {output_file}")
    
    plt.tight_layout()
    plt.show()
    
    # If requested, save synthetic data to a file for later use
    if output_file and output_file.endswith('.txt'):
        with open(output_file, 'w') as f:
            for size, time in zip(input_sizes, execution_times):
                f.write(f"Input size: {size} nodes\n")
                f.write(f"Execution time: {time} microseconds\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python performance_visualizer.py <command> [options]")
        print("Commands:")
        print("  visualize <algorithm_name> [output_file] - Generate performance graph for an algorithm")
        print("  compare <algorithm1,algorithm2,...> [output_file] - Compare multiple algorithms")
        print("  synthetic <algorithm_name> [output_file] - Generate synthetic data graph")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "visualize" and len(sys.argv) >= 3:
        algorithm = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else f"{algorithm}_performance.png"
        generate_performance_graph(algorithm, output_file)
    
    elif command == "compare" and len(sys.argv) >= 3:
        algorithms = sys.argv[2].split(',')
        output_file = sys.argv[3] if len(sys.argv) > 3 else "algorithm_comparison.png"
        compare_algorithms(algorithms, output_file)
    
    elif command == "synthetic" and len(sys.argv) >= 3:
        algorithm = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else f"{algorithm}_synthetic.png"
        
        # Different complexity functions
        complexities = {
            "linear": lambda n: n * 0.5,
            "nlogn": lambda n: n * np.log2(n) * 0.1,
            "quadratic": lambda n: n * n * 0.005,
            "cubic": lambda n: n * n * n * 0.0001,
            "exponential": lambda n: 2 ** (n * 0.01)
        }
        
        # Default to nlogn complexity
        time_function = complexities.get("nlogn")
        
        # Add some randomness
        def random_time_function(n):
            base = time_function(n)
            return base * (1 + 0.2 * np.random.random())
        
        generate_synthetic_data(algorithm, time_function=random_time_function, output_file=output_file)
    
    else:
        print("Invalid command or missing arguments")
        print("Usage: python performance_visualizer.py <command> [options]")

