"""
Simulation runner utilities.

This module provides functions to run simulations and collect results.
"""

import time
from typing import Dict, Tuple, List
from toy_chiplet_sim import SystemConfig, Simulator
from experiment_config import (GEMM_VERIFY_MODE, P2P_VERIFY_MODE,
                               REBALANCE_ENABLED, REBALANCE_MAX_ROUNDS, REBALANCE_DISTANCE)


def run_simulation(
    config: SystemConfig,
    trace_path_list: List[str],
    batch_size: int
) -> Tuple[Simulator, float]:
    """
    Run a single simulation.
    
    Args:
        config: System configuration
        trace_path_list: List of trace file paths
        batch_size: Batch size for this simulation
        
    Returns:
        Tuple of (simulator instance, elapsed_time_in_seconds)
    """
    sim = Simulator(config, gemm_verify_mode=GEMM_VERIFY_MODE, p2p_verify_mode=P2P_VERIFY_MODE,
                    rebalance_enabled=REBALANCE_ENABLED,
                    rebalance_max_rounds=REBALANCE_MAX_ROUNDS,
                    rebalance_distance=REBALANCE_DISTANCE)
    # Pre-loaded: list of trace data (each trace can be list of iters or dict; path would be str)
    if trace_path_list and not isinstance(trace_path_list[0], str):
        # print("trace_path_list is a pre-loaded list of trace data")
        trace_subset = trace_path_list  # pre-loaded list of traces
    else:
        # print("trace_path_list is a list of file paths")
        trace_subset = trace_path_list[:batch_size]  # list of file paths
    t0 = time.perf_counter()
    sim.process_trace(trace_subset)
    sim.process_records()
    elapsed = time.perf_counter() - t0
    
    return sim, elapsed


def collect_simulation_results(sim: Simulator, batch_size: int) -> Dict[str, float]:
    """
    Collect results from a simulation run.
    
    Args:
        sim: Simulator instance after running simulation
        batch_size: Batch size used in simulation
        
    Returns:
        Dictionary containing all simulation metrics
    """
    tot_time = sim.current_time / 1e6  # ms
    prefill_time = sim.prefill_time / 1e6
    decode_time = tot_time - prefill_time
    tot_hop = sim.stats['hop']
    chiplet_usage = sim.avg_chiplet_usage
    duration_max_over_min = sim.duration_max_over_min
    duration_max_over_avg = sim.duration_max_over_avg
    allo_exp_avg_over_max = sim.allo_exp_avg_over_max
    allo_req_avg_over_max = sim.allo_req_avg_over_max
    dram_access_avg_over_max = sim.dram_access_avg_over_max
    local_dram_read_size = sim.local_dram_read_size / (1024**3)
    local_dram_write_size = sim.local_dram_write_size / (1024**3)
    remote_dram_read_size = sim.remote_dram_read_size / (1024**3)
    throughput = 1 * batch_size * 128 / decode_time * 1000
    
    dram_size_GB = sum(chiplet.dram.total_data_size_GB for chiplet in sim.interconnect.chiplets)
    
    return {
        'tot_time': tot_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'tot_hop': tot_hop,
        'chiplet_usage': chiplet_usage,
        'duration_max_over_min': duration_max_over_min,
        'duration_max_over_avg': duration_max_over_avg,
        'allo_exp_avg_over_max': allo_exp_avg_over_max,
        'allo_req_avg_over_max': allo_req_avg_over_max,
        'dram_access_avg_over_max': dram_access_avg_over_max,
        'local_dram_read_size': local_dram_read_size,
        'local_dram_write_size': local_dram_write_size,
        'remote_dram_read_size': remote_dram_read_size,
        'throughput': throughput,
        'dram_size_GB': dram_size_GB,
    }


def format_result_line(
    model_name: str,
    strategy_name: str,
    chiplet_arch: Tuple[int, int],
    batch_size: int,
    results: Dict[str, float],
) -> str:
    """
    Format simulation results as a CSV line.
    
    Args:
        model_name: Name of the model
        strategy_name: Strategy name (from STRATEGIES)
        chiplet_arch: Tuple of (y_chiplets, x_chiplets)
        batch_size: Batch size
        results: Dictionary of simulation results
        
    Returns:
        Formatted CSV line string
    """
    line = (
        f"{model_name},{strategy_name},{chiplet_arch[0]},{chiplet_arch[1]},{batch_size},"
        f"{results['chiplet_usage']:.2f},{results['duration_max_over_min']:.2f},"
        f"{results['duration_max_over_avg']:.2f},{results['allo_exp_avg_over_max']:.2f},"
        f"{results['allo_req_avg_over_max']:.2f},{results['dram_access_avg_over_max']:.2f},"
        f"{results['dram_size_GB']:.0f},{results['local_dram_read_size']:.0f},"
        f"{results['remote_dram_read_size']:.0f},{results['local_dram_write_size']:.0f},"
        f"{results['tot_hop']},{results['tot_time']:.2f},{results['prefill_time']:.2f},"
        f"{results['decode_time']:.2f},{results['throughput']:.0f}"
    )
    
    return line
