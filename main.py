"""
Main entry point for MoE simulation benchmarks.

This script runs simulations across multiple configurations (models, chiplet architectures,
strategies, and batch sizes) and saves results to CSV files.
"""

import os
import sys
from typing import Dict, Tuple, List

from toy_chiplet_sim import SystemConfig, Simulator
from toy_chiplet_sim.simulation.simulator import ENABLE_PROFILING
from experiment_config import (
    MODELS,
    CHIPLET_ARCHITECTURES,
    STRATEGIES,
    STRATEGY_MAP,
    BATCH_SIZES,
    GEMM_VERIFY_MODE,
    P2P_VERIFY_MODE,
    get_strategy_config,
)
from data_loader import gen_full_trace_data_list, load_trace_files, rotate_trace_list
from simulator_runner import (
    run_simulation,
    collect_simulation_results,
    format_result_line,
)
from plotting import plot_decode_time_vs_batch


# Per-config expert usage/allocation CSVs under results/expert_analysis/.
# Off during result-gathering runs; set True to write them again.
EXPORT_EXPERT_ANALYSIS = False

# Per-config D2D communication CSVs under results/comm_analysis/ (Phase 2).
# Needed for the multicast study; cheap to write.
EXPORT_COMM_ANALYSIS = True

CSV_HEADER = "model,strategy,y_chips,x_chips,batch,usage,die_max_over_min,die_max_over_avg,allo_exp_skew,allo_req_skew,dram_access_skew,dram,local_dram_read,remote_dram_read,local_dram_write,hop,time,prefill,decode,throughput\n"


def should_skip_configuration(
    allocation_strategy: SystemConfig.AllocationStrategy,
    exe_strategy: SystemConfig.ExeStrategy
) -> bool:
    """
    Check if a configuration should be skipped.
    
    Args:
        allocation_strategy: Allocation strategy
        exe_strategy: Execution strategy
        
    Returns:
        True if configuration should be skipped, False otherwise
    """
    # Skip NEAREST + PRED_NEXT_TOKEN combination
    if allocation_strategy == SystemConfig.AllocationStrategy.NEAREST and \
       exe_strategy == SystemConfig.ExeStrategy.PRED_NEXT_TOKEN:
        return True
    return False


def run_benchmark_suite(model_filter=None):
    """
    Run the complete benchmark suite across all configurations.
    
    Args:
        model_filter: If provided, only run this single model (e.g. "qwen").
                      If None, run all models in MODELS list.
    """
    # Data collection for plotting
    plot_data: Dict[Tuple[str, Tuple[int, int], SystemConfig.AllocationStrategy, SystemConfig.ExeStrategy], Tuple[List[int], List[float]]] = {}
    
    models_to_run = [model_filter] if model_filter else MODELS
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Load trace files for this model
        full_trace_path_list = gen_full_trace_data_list(model_name)
        # print("org first 8192 traces: ", full_trace_path_list[0])
        full_trace_path_list = rotate_trace_list(full_trace_path_list, start_index=8192)
        # print("rotated first 8192 traces: ", full_trace_path_list[0])
        print(f"Total number of trace files: {len(full_trace_path_list)}")
        
        # Prepare result file
        result_file_name = f"results/{model_name}_results.csv"
        os.makedirs(os.path.dirname(result_file_name), exist_ok=True)
        
        with open(result_file_name, "w") as file:
            # Write CSV header
            file.write(CSV_HEADER)
            
            BATCH_SIZES_TO_TEST = BATCH_SIZES
            if GEMM_VERIFY_MODE:
                BATCH_SIZES_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
            BATCH_SIZES_SORTED = sorted(BATCH_SIZES_TO_TEST)

            # Iterate through strategies first, then batch sizes
            for strategy_name in STRATEGIES:
                print(f"\nStrategy: {strategy_name}")
                print("-" * 60)

                exe_strategy, allocation_strategy = get_strategy_config(strategy_name)

                # Incremental trace loading: reuse previously loaded data across batch sizes
                trace_cache = []

                for batch_size in BATCH_SIZES_SORTED:
                    # Load trace data incrementally -------------------------
                    if GEMM_VERIFY_MODE or P2P_VERIFY_MODE:
                        trace_data = load_trace_files(full_trace_path_list[:1]) * batch_size
                    else:
                        needed = batch_size
                        if needed > len(trace_cache):
                            new_paths = full_trace_path_list[len(trace_cache):needed]
                            new_data = load_trace_files(new_paths)
                            trace_cache.extend(new_data)
                            # print(f"[Data] Loaded {len(new_data)} new traces (cache: {len(trace_cache)} total)")
                        trace_data = trace_cache[:batch_size]

                    # Iterate through chiplet architectures
                    for chiplet_arch in CHIPLET_ARCHITECTURES:
                        # Skip invalid configurations
                        if should_skip_configuration(allocation_strategy, exe_strategy):
                            continue
                        
                        # Create configuration
                        config = SystemConfig(
                            y_chiplets=chiplet_arch[0],
                            x_chiplets=chiplet_arch[1],
                            exe_strategy=exe_strategy,
                            allocation_strategy=allocation_strategy,
                            model_name=model_name
                        )
                        
                        # Run simulation (trace_data already loaded, no file I/O)
                        sim, elapsed = run_simulation(config, trace_data, batch_size)
                        
                        if ENABLE_PROFILING:
                            print(
                                f"[Timer] model={model_name} strategy={strategy_name} "
                                f"batch={batch_size} chiplets={chiplet_arch} -> {elapsed:.2f}s"
                            )
                        
                        # Collect results
                        results = collect_simulation_results(sim, batch_size)
                        
                        # Format and write result line
                        line = format_result_line(
                            model_name,
                            strategy_name,
                            chiplet_arch,
                            batch_size,
                            results,
                        )
                        
                        print(line)
                        file.write(line + "\n")
                        file.flush()

                        if EXPORT_EXPERT_ANALYSIS:
                            usage_path, allocation_path = sim.export_expert_analysis(
                                "results/expert_analysis", strategy_name, batch_size
                            )
                            print(f"[Expert analysis] {usage_path}")
                            print(f"[Expert analysis] {allocation_path}")

                        if EXPORT_COMM_ANALYSIS:
                            for comm_path in sim.export_comm_analysis(
                                "results/comm_analysis", strategy_name, batch_size
                            ):
                                print(f"[Comm analysis] {comm_path}")
                        
                        # Collect data for plotting
                        key = (model_name, chiplet_arch, allocation_strategy, exe_strategy)
                        if key not in plot_data:
                            plot_data[key] = ([], [])
                        plot_data[key][0].append(batch_size)
                        plot_data[key][1].append(results['decode_time'])
        
        # Plot decode time vs batch size (only in VERIFY mode)
        if GEMM_VERIFY_MODE or P2P_VERIFY_MODE:
            print(f"\nGenerating plots for {model_name}...")
            plot_decode_time_vs_batch(plot_data)


def main():
    """
    Simple test function for single simulation run.
    """
    dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/abstract_algebra"
    trace_path_list = [
        os.path.join(dataset_dir, f) 
        for f in os.listdir(dataset_dir) 
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]
    trace_path_list = trace_path_list[:6]
    print("File paths:", trace_path_list)
    
    config = SystemConfig(
        y_chiplets=3,
        x_chiplets=3,
        allocation_strategy=SystemConfig.AllocationStrategy.NEAREST,
        exe_strategy=SystemConfig.ExeStrategy.PRED_NEXT_TOKEN,
    )
    
    sim = Simulator(config, gemm_verify_mode=GEMM_VERIFY_MODE, p2p_verify_mode=P2P_VERIFY_MODE)
    sim.process_trace(trace_path_list)
    
    print("=== Simulation Results ===")
    print(f"Total Time: {sim.current_time / 1e6:.2f} ms")
    print(f"Local DRAM Hit Rate: {sim.stats['local_dram_hit']/sim.stats['total_access']:.2%}")
    hop_time = sim.stats['hop'] * (sim.timing_config.link_latency + sim.config.expert_slice_size / sim.config.d2d_link_bw) / 1e6
    print(f"Total Hops: {sim.stats['hop']}")
    
    sim.process_records()
    print(f"chiplet usage: {sim.avg_chiplet_usage:.2f}")
    
    print(f"req num of each chiplet: ")
    for chiplet in sim.interconnect.chiplets:
        req_num = chiplet.req_num
        print(f"chiplet: {chiplet.id}, req_num: {req_num:.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_benchmark_suite(model_filter=sys.argv[1])
    else:
        run_benchmark_suite()
