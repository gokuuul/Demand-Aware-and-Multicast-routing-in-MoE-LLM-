"""
Plotting utilities for simulation results.

This module provides functions to visualize simulation results and compare with benchmarks.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from toy_chiplet_sim import SystemConfig


def load_benchmark_data(csv_path: str = "verify_ref/moe_bench_results.csv") -> Dict[str, pd.DataFrame]:
    """
    Load benchmark data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing benchmark results
        
    Returns:
        Dictionary mapping model_name to DataFrame with columns: batch_size, decode_time
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Benchmark file not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    benchmark_data = {}
    
    # Map CSV model names to simulator model names (one CSV name can map to multiple sim names)
    model_mapping = {
        "Qwen3-235B (4096x1536)": ["qwen"],
        "DeepSeek (7168x2048)": ["deepseek", "kimi"],
        "Llama4 (5120x8192)": ["llama4"],
    }
    
    for csv_model_name, sim_model_names in model_mapping.items():
        model_df = df[df['model'] == csv_model_name].copy()
        if not model_df.empty:
            renamed = model_df[['batch', 'avg_ms']].rename(
                columns={'batch': 'batch_size', 'avg_ms': 'decode_time'}
            )
            for sim_model_name in sim_model_names:
                benchmark_data[sim_model_name] = renamed
    
    return benchmark_data


def plot_decode_time_vs_batch(
    plot_data: Dict[Tuple[str, Tuple[int, int], SystemConfig.AllocationStrategy, SystemConfig.ExeStrategy], Tuple[List[int], List[float]]],
    benchmark_csv_path: str = "verify_ref/moe_bench_results.csv",
    output_dir: str = "figures"
) -> None:
    """
    Plot decode time vs batch size for all configurations, including benchmark data.
    
    Args:
        plot_data: Dictionary mapping (model_name, chiplet_arch, allocation_strategy, exe_strategy) 
                   to ([batch_sizes], [decode_times])
        benchmark_csv_path: Path to CSV file with benchmark results
        output_dir: Directory to save the plots (default: "figures")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(benchmark_csv_path)
    
    for (model_name, chiplet_arch, allocation_strategy, exe_strategy), (batch_sizes, decode_times) in plot_data.items():
        # Sort by batch_size for clean plot
        sorted_pairs = sorted(zip(batch_sizes, decode_times))
        batch_sizes_sorted, decode_times_sorted = zip(*sorted_pairs)
        
        plt.figure(figsize=(10, 6))
        
        # Plot simulator results
        plt.plot(batch_sizes_sorted, decode_times_sorted, marker='o', linewidth=2, 
                markersize=6, label='Simulator', color='blue')
        
        # Plot benchmark data if available for this model
        if model_name in benchmark_data:
            bench_df = benchmark_data[model_name].copy()
            bench_df_sorted = bench_df.sort_values('batch_size')

            # Original benchmark curve
            plt.plot(bench_df_sorted['batch_size'], bench_df_sorted['decode_time'],
                    marker='s', linewidth=1, markersize=4, label='Benchmark (original)', color='orange', linestyle=':', alpha=0.6)

            # Compute offset: average difference (sim - bench) at batch 1, 2, 4, 8
            sim_dict = dict(zip(batch_sizes_sorted, decode_times_sorted))
            calib_batches = [b for b in [1, 2, 4, 8] if b in sim_dict]
            if calib_batches:
                diffs = []
                for b in calib_batches:
                    bench_row = bench_df_sorted[bench_df_sorted['batch_size'] == b]
                    if not bench_row.empty:
                        diffs.append(sim_dict[b] - float(bench_row['decode_time'].iloc[0]))
                if diffs:
                    offset = sum(diffs) / len(diffs)
                    bench_df_sorted = bench_df_sorted.copy()
                    bench_df_sorted['decode_time'] = bench_df_sorted['decode_time'] + offset
                    print(f"Benchmark offset (avg diff at small batches): {offset:.6f} ms")

            # Shifted benchmark curve
            plt.plot(bench_df_sorted['batch_size'], bench_df_sorted['decode_time'], 
                    marker='s', linewidth=2, markersize=6, label='Benchmark (shifted)', color='red', linestyle='--')
        
        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Decode Time (ms)', fontsize=12)
        
        # Format title
        allo_prefix = str(allocation_strategy).split('.')[-1]
        exe_prefix = str(exe_strategy).split('.')[-1]
        plt.title(f'Decode Time vs Batch Size\nModel: {model_name}, Chiplets: {chiplet_arch[0]}×{chiplet_arch[1]}, '
                 f'Allocation: {allo_prefix}, Execution: {exe_prefix}', fontsize=11)
        
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        # X-axis: show actual batch size numbers
        ax = plt.gca()
        all_batches = set(batch_sizes_sorted)
        if model_name in benchmark_data:
            all_batches |= set(benchmark_data[model_name]['batch_size'].tolist())
        all_batches = sorted(all_batches)
        if all_batches:
            ax.set_xticks(all_batches)
            ax.set_xticklabels([int(b) for b in all_batches], rotation=45, ha='right')
        plt.legend(loc='best', fontsize=10)
        
        # Save figure
        filename = f"{output_dir}/decode_time_vs_batch_{model_name}_{chiplet_arch[0]}x{chiplet_arch[1]}_{allo_prefix}_{exe_prefix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")
