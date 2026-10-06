"""
Experiment configuration for MoE simulation benchmarks.

This module defines the experiment configurations including models, chiplet architectures,
strategies, and batch sizes.
"""

from toy_chiplet_sim import SystemConfig
from typing import List, Tuple, Dict


# Model configurations
MODELS = [
    "qwen",
    "deepseek",
    "kimi",
    "llama4",
    ]

# Chiplet architecture configurations (y_chiplets, x_chiplets)
# CHIPLET_ARCHITECTURES = [(5, 5)]
# Alternative chiplet architecture configurations (commented out):
# CHIPLET_ARCHITECTURES = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 16), (2, 24), (2, 32), (2, 40), (2, 48)]
# CHIPLET_ARCHITECTURES = [(2, 16), (2, 32), (8, 8), (16, 16)]
# CHIPLET_ARCHITECTURES = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
# CHIPLET_ARCHITECTURES = [(5, 5), (8, 3)]
CHIPLET_ARCHITECTURES = [(8, 3)]

# Strategy mapping: maps strategy name to (exe_strategy, allocation_strategy)
STRATEGY_MAP = {
    "base": [SystemConfig.ExeStrategy.BASELINE_CACHE_LOCAL, SystemConfig.AllocationStrategy.EXP_EVEN],
    "pred_only": [SystemConfig.ExeStrategy.PRED_NEXT_TOKEN, SystemConfig.AllocationStrategy.EXP_EVEN],
    "allo_only": [SystemConfig.ExeStrategy.BASELINE_CACHE_LOCAL, SystemConfig.AllocationStrategy.OURS],
    "allo_and_pred": [SystemConfig.ExeStrategy.PRED_NEXT_TOKEN, SystemConfig.AllocationStrategy.OURS],
    "norm_ep": [SystemConfig.ExeStrategy.BASELINE_CACHE_LOCAL, SystemConfig.AllocationStrategy.NEAREST],
    "token_par": [SystemConfig.ExeStrategy.BASELINE_CACHE_LOCAL, SystemConfig.AllocationStrategy.TOKEN_PARALLEL],
    "allo_only_org": [SystemConfig.ExeStrategy.BASELINE_CACHE_LOCAL, SystemConfig.AllocationStrategy.OURS_ORG],
}

# List of strategies to test
STRATEGIES = [
    "base",
    "norm_ep", 
    "allo_only",
    "pred_only",
    "allo_and_pred",
    # "allo_only_org",
]


# Batch sizes to test
# BATCH_SIZES = [1, 2, 4]
# Alternative batch size configurations (commented out):
# BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
# BATCH_SIZES = [8, 16, 32, 6144]
# BATCH_SIZES = [128, 2048, 4096, 6144, 8192, 16384]
# BATCH_SIZES = [4096, 6144, 8192, 10240, 12288, 16384]
BATCH_SIZES = [4096, 8192, 16384]
# BATCH_SIZES = [32, 64, 128, 256]
# BATCH_SIZES = [32]
# BATCH_SIZES = [4096]
# BATCH_SIZES = [6144]
# BATCH_SIZES = [2048]
# BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
# BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024]
# Using numpy for batch sizes:
# import numpy as np
# BATCH_SIZES = list(2 ** np.arange(12))  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# BATCH_SIZES_1 = list(2 ** np.arange(9))  # [1, 2, 4, 8, 16, 32, 64, 128, 256]
# BATCH_SIZES_2 = list(np.arange(512, 16385, 512))  # [512, 1024, 1536, ..., 16384]
# BATCH_SIZES = BATCH_SIZES_1 + BATCH_SIZES_2

# Verify mode flags
GEMM_VERIFY_MODE = False
P2P_VERIFY_MODE = False

# Rebalance pass after greedy allocation (OURS strategy only)
REBALANCE_ENABLED = False
REBALANCE_MAX_ROUNDS = 3
REBALANCE_DISTANCE = 2


def get_strategy_config(strategy_name: str) -> Tuple[SystemConfig.ExeStrategy, SystemConfig.AllocationStrategy]:
    """
    Get execution and allocation strategy from strategy name.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Tuple of (exe_strategy, allocation_strategy)
        
    Raises:
        ValueError: If strategy_name is not in STRATEGY_MAP
    """
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_MAP.keys())}")
    exe_strategy, allocation_strategy = STRATEGY_MAP[strategy_name]
    return exe_strategy, allocation_strategy
