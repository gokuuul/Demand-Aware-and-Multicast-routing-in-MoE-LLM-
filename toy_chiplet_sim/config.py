"""
System and timing configuration for the multi-chiplet MoE simulator.
"""

from enum import Enum
from typing import List


class SystemConfig:
    """System-level configuration: topology, model, allocation and execution strategy."""

    class InterconnectType(Enum):
        MESH = 1
        RING = 2
        CROSSBAR = 3

    class CacheStrategy(Enum):
        CACHE_LOCAL = 1
        CACHE_ALL = 2
        DISABLED = 3

    class ExeStrategy(Enum):
        BASELINE_CACHE_LOCAL = 1
        BASELINE_CACHE_ALL = 2
        BASELINE_DISABLE_CACHE = 3
        PRED_NEXT_LAYER = 4
        PRED_NEXT_TOKEN = 5

    class AllocationStrategy(Enum):
        EVEN = 1      # evenly distribute requests to all dies
        EXP_EVEN = 2  # evenly distribute expert requests to all dies
        RANDOM = 3    # chiplet_id = expert_id % chiplet_num
        NEAREST = 4   # allocate to the die where the expert is placed
        OURS = 5      # load-balanced allocation
        OURS_ORG = 6  # load-balanced allocation with original placement
        TOKEN_PARALLEL = 7  # tokens pinned per die; each die runs the union of
                            # experts its own tokens selected (data-parallel
                            # placement). Produces multicast fanout: one expert
                            # is requested by many dies at once.
    
    def __init__(
        self,
        y_chiplets=2,
        x_chiplets=2,
        allocation_strategy=None,
        exe_strategy=None,
        model_name="deepseek",
    ):
        if allocation_strategy is None:
            allocation_strategy = self.AllocationStrategy.NEAREST
        if exe_strategy is None:
            exe_strategy = self.ExeStrategy.BASELINE_CACHE_LOCAL

        self.y_chiplets = y_chiplets
        self.x_chiplets = x_chiplets
        self.chiplet_num = self.x_chiplets * self.y_chiplets
        self.interconnect = self.InterconnectType.MESH
        self.routing_table = {}
        self.model_name = model_name

        if model_name == "deepseek":
            self.layer_num = 61
            self.hidden_dim = 7168
            self.expert_dim = 2048
            self.expert_num = 256
            self.routed_expert_num = 8
            self.shared_expert_num = 1
            self.data_type = 2
        elif model_name == "qwen":
            self.layer_num = 94
            self.hidden_dim = 4096
            self.expert_dim = 1536
            self.expert_num = 128
            self.routed_expert_num = 8
            self.shared_expert_num = 0
            self.data_type = 2
        elif model_name == "kimi":
            self.layer_num = 61
            self.hidden_dim = 7168
            self.expert_dim = 2048
            self.expert_num = 384
            self.routed_expert_num = 8
            self.shared_expert_num = 1
            self.data_type = 2
        elif model_name == "llama4":
            self.layer_num = 48
            self.hidden_dim = 5120
            self.expert_dim = 8192
            self.expert_num = 128
            self.routed_expert_num = 1
            self.shared_expert_num = 0
            self.data_type = 2
        else:
            raise ValueError("Undefined model name")
        
        self.whole_expert_as_one_slice = False
        self.expert_matrix_num = 3
        self.expert_matrix_slice_num = 1
        self.expert_size = self.expert_matrix_num * self.hidden_dim * self.expert_dim * self.data_type
        if self.whole_expert_as_one_slice:
            self.expert_tot_slice_num = 1
            self.expert_slice_size = self.expert_size
        else:
            self.expert_tot_slice_num = self.expert_matrix_num * self.expert_matrix_slice_num
            self.expert_slice_size = self.expert_size / self.expert_matrix_num / self.expert_matrix_slice_num
        # self.expert_slice_operations = 2 * 2 * self.hidden_dim * self.expert_dim
        
        self.d2d_link_bw = 1741
        self.cache_size = 64 * 1024 * 1024
        self.cache_assoc = 2
        self.line_size = self.expert_slice_size
        self.dram_size_per_die = 1000
        self.dram_bw_per_die = 3350
        self.hardware_manage_dram_size_per_die = 50
        self.compte_power_per_die = 1000

        self.allocation_strategy = allocation_strategy
        self.exe_strategy = exe_strategy
        self.cache_strategy = self.CacheStrategy.CACHE_LOCAL
        if self.exe_strategy == self.ExeStrategy.BASELINE_CACHE_ALL:
            self.cache_strategy = self.CacheStrategy.CACHE_ALL
        if self.exe_strategy == self.ExeStrategy.BASELINE_DISABLE_CACHE:
            self.cache_strategy = self.CacheStrategy.DISABLED
        self.pred_next_token = self.exe_strategy == self.ExeStrategy.PRED_NEXT_TOKEN
        self.pred_next_layer = self.exe_strategy == self.ExeStrategy.PRED_NEXT_LAYER


class TimingConfig:
    """Latency and timing parameters (nanoseconds)."""

    def __init__(self):
        self.cache_hit_latency = 100
        self.cache_miss_penalty = 110
        self.cache_store_latency = 30
        self.link_latency = 200
        self.mem_access_latency = 300
        self.compute_latency = 200
