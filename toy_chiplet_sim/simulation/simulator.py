"""
Discrete-event simulator: trace-driven MoE execution on multi-chiplet mesh.
"""

import statistics
import sys
import copy
import csv
import heapq
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import SystemConfig, TimingConfig
from ..models import Addr_expert, Addr_chip, Expert, Expert_slice
from ..hardware import Chiplet, Interconnect
from .events import EventType, ResourceType, Resource, SimulationEvent
from .resource_manager import ResourceManager
from .comm_recorder import CommRecorder, KIND_REQ, KIND_DATA, KIND_DATA_LLC

DEBUG = False
PRINT_DEBUG = False
SECURITY_CHECK = True

DEBUG_ALLO = True
SUM_TABLE_EN = True

DEBUG_ALLO = False
SUM_TABLE_EN = False 
ENABLE_PROFILING = False

# Phase 2: per-link / per-die / per-pair D2D recording -> results/comm_analysis/.
# ~15% of event-loop time; False disables it entirely.
COMM_TRACE_ENABLED = True

# Print a progress line every N iterations (0 = silent). A long run otherwise
# prints nothing at all until the config completes, which makes a slow run
# indistinguishable from a hung one.  MOE_PROGRESS=0 to disable.
PROGRESS_EVERY_ITERS = int(os.environ.get("MOE_PROGRESS", "16"))

# Phase 3: record the multicast fanout distribution (cheap, measurement only).
MEASURE_FANOUT = True

# Phase 3: deliver a slice to all its requesters as one tree instead of one
# unicast per requester. Groups of size <= 1 fall back to the unicast path.
MULTICAST_ENABLED = bool(int(os.environ.get("MOE_MULTICAST", "0")))

# >0 truncates every run to this many iterations (fast smoke test).
# Override without editing this file:  MOE_MAX_ITERS=8 python main_ae.py ...
LIMIT_MAX_ITER_NUM = int(os.environ.get("MOE_MAX_ITERS", "0"))
# LIMIT_MAX_LAYER_NUM = 5

class Simulator:
    def __init__(self, config_file, gemm_verify_mode=False, p2p_verify_mode=False,
                 rebalance_enabled=False, rebalance_max_rounds=3, rebalance_distance=2):
        self.config: SystemConfig = config_file
        self.gemm_verify_mode = gemm_verify_mode
        self.p2p_verify_mode = p2p_verify_mode
        self.rebalance_enabled = rebalance_enabled
        self.rebalance_max_rounds = rebalance_max_rounds
        self.rebalance_distance = rebalance_distance
        self.timing_config = TimingConfig()
        self.interconnect: Interconnect = Interconnect(self.config)
        self.expert_list: Dict[int, List[Expert]] = {} # {layer_id: [expert1, expert2 ...], ...}
        self.event_queue: List[SimulationEvent] = []
        self.res_mgr = ResourceManager()

        self.expert_chiplet_map = {}
        self.org_exp_chiplet_distribution = {}
        self._init_experts()
        self._init_data_placement()
        self._init_resources()
        # print(self.expert_chiplet_map)
        # print("DRAM")
        # for chiplet_id in range(self.config.chiplet_num):
        #     print("chiplet :%d " %(chiplet_id), end="")
        #     self.interconnect.chiplets[chiplet_id].dram.print_layer(0)
        # print("\nend")
        # print(self.interconnect.chiplets[0].dram.data)
         
        # stat
        self.stats = {
            'total_cycles': 0,
            'local_cache_hit': 0,
            'local_dram_hit': 0,
            'hop': 0,
            'total_access': 0
        }
        self.local_dram_read_size = 0
        self.remote_dram_read_size = 0
        self.local_dram_write_size = 0

        # D2D communication recorder (see simulation/comm_recorder.py)
        self.comm = CommRecorder(self.config, self.interconnect,
                                 enabled=COMM_TRACE_ENABLED)
        
        
        self.current_time = 0

        self.trace_generator = None
        self.current_request = None
        self.pending_requests = 0

        self.req_num = 0
        self.completion_num = 0

        self.layer_pred_table = {} # frequent use pair (layer n, layer n+1) {layer: {(expert pair):num, ():num, ...}, ...}
        self.token_pred_table = {} # frequent use pair (token n, token n+1)
        # print(self.config.routed_expert_num)
        self.token_pred_heatmap = np.zeros((self.config.layer_num, self.config.expert_num, self.config.expert_num)) # (layer, token n, token n+1)
        
        self.die_expert_counts = None  # TOKEN_PARALLEL only; see _trace_reader_core

        # Phase 3: distribution of multicast fanout -- how many dies request the
        # same expert slice in one layer. fanout_hist[k] = number of slice
        # broadcasts with exactly k remote requesters. Measurement only; it does
        # not change delivery.
        self.fanout_hist = np.zeros(self.config.chiplet_num + 1, dtype=np.int64)

        self.current_experts = {}
        self.current_layer = None   #
        self.current_stage = None   # prefill / decode
        self.last_iter_stage = None

        self.iter_id = 0
        self.layer_id = 0

        self.pre_layer_begin_time = 0
        self.cur_layer_begin_time = 0
        self.layer_time_records: Dict[int, Dict[int, Dict[int, int]]] = {} 
        
        # records
        self.access_records: Dict[int, Dict[int, Dict[int, List]]] = {} # {iter_id: {layer_id: {chiplet_id: [min_start, max_end]}}}
        self.duration_records: Dict[int, Dict[int, Dict[int, int]]] = {} # {iter_id: {layer_id: {chiplet_id: duration}}}
        self.duration_sum: np.array = np.empty(0)
        self.allocation_records: Dict[int, Dict[int, Dict[int, List]]] = {} # {iter_id: {layer_id: {chiplet_id: [expert_num, req_num]}}}
        self.allocation_summary = {} # summary of which chiplet is allocated
        # Aggregated expert-level records, used for post-run analysis exports.
        # Keys are (stage, layer_id, expert_id) and
        # (stage, layer_id, expert_id, executing_chiplet), respectively.
        self.expert_selection_records = Counter()
        self.expert_execution_records = Counter()
        
        self.dram_access_records: Dict[int, Dict[int, np.array]] = {} # {iter_id: {layer_id: [dram_of_die0, dram_of_die1, ...]}}
        self.dram_access_avg_over_max = 0
        
        self.avg_chiplet_usage = 0
        self.duration_max_over_min = 0.0
        self.duration_max_over_avg = 0.0
        self.allo_exp_min_over_max = 0
        self.allo_exp_avg_over_max = 0
        self.allo_req_min_over_max = 0
        self.allo_req_avg_over_max = 0
        
        # for security check
        self.max_iter_num = 0
        self.max_layer_num = 0
            
        # to support attention
        self.operation = "moe" # "moe", "attention"
        self.iter_info = [] # temporally store the info from generater [expert_dict, layer, stage]
        self.tot_batch_size = None 
        self.tot_attn_time = 0
        
        self.prefill_time = 0

        # temporal storage
        self.exp_allo_plan = None # [[expert_id, req_num, chiplet_id], ...]

        # profiling timers (wall-clock seconds)
        self._prof = {
            'event_loop': 0.0,
            'allocate_experts': 0.0,
            'stage_map': 0.0,
            'aggregate_decode': 0.0,
            'stat_prefill_pairs': 0.0,
            'trace_gen_yield': 0.0,
            'record_allocation': 0.0,
            'process_records': 0.0,
            'event_loop_iterations': 0,
        }
        
        # for idx, chiplet in enumerate(self.interconnect.chiplets):
        #     print(idx, chiplet.id)
        
    def _init_experts(self):
        for layer_id in range(self.config.layer_num):
            self.expert_list[layer_id] = []
            for expert_id in range(self.config.expert_num):
                expert = Expert(self.config, layer_id=layer_id, expert_id=expert_id)
                self.expert_list[layer_id].append(expert)

    def _init_data_placement(self):

        # slice each experts into slice
        for layer_id in self.expert_list:
            for expert in self.expert_list[layer_id]:
                expert.slice_expert()
        
        # expert placement
        # expert_per_chiplet = self.config.expert_num // self.config.chiplet_num
        # if expert_per_chiplet % self.config.chiplet_num != 0:
        #     raise ValueError("Wrong config of chiplet num and expert num")
        # # print(expert_per_chiplet)
        # for chiplet_id, chiplet in enumerate(self.interconnect.chiplets):
        #     for i in range(expert_per_chiplet):
        #         expert_id = chiplet_id * expert_per_chiplet + i # 0-64 in die 0
        #         for layer_id in range(self.config.layer_num):
        #             expert = self.expert_list[layer_id][expert_id]
        #             for expert_slice in expert.slice_list:
        #                 chiplet.dram.store(expert_slice)


        for layer_id in self.expert_list:
            for expert in self.expert_list[layer_id]:
                for expert_slice in expert.slice_list:
                    address = Addr_expert(
                        expert_id=expert_slice.expert_id,
                        layer_id=expert_slice.layer_id,
                        matrix_id=expert_slice.matrix_id,
                        slice_id=expert_slice.slice_id
                        )
                    chiplet_id = self._expert_placement_map(address)
                    chiplet = self.interconnect.chiplets[chiplet_id]
                    chiplet.dram.store(address, self.config.expert_slice_size, is_hm_data=False)
                    
                    self.expert_chiplet_map[expert_slice.expert_id] = chiplet_id
                    self.org_exp_chiplet_distribution[address.expert_id] = [chiplet_id]
        # print(self.interconnect.chiplets[3].dram)
        # dram3 = self.interconnect.chiplets[3].dram
        # for slice in dram3.data:
        #     print(slice.expert_id, slice.layer_id, )
        # print(dram3)

        # self._print_dram()
        # print("org exp placement: ", self.org_exp_chiplet_distribution)
        return
    
    def _init_resources(self):
        for i in range(self.config.chiplet_num):
            self.res_mgr.register_resource(ResourceType.DRAM_PACKAGE, i)
            self.res_mgr.register_resource(ResourceType.CACHE_PORT, i)
            self.res_mgr.register_resource(ResourceType.COMPUTE_UNIT, i)
        for link in self.interconnect.links:
            self.res_mgr.register_resource(ResourceType.D2D_LINK, link)

    def _print_dram(self):
        for chiplet in self.interconnect.chiplets:
            dram_size = len(chiplet.dram.data) * self.config.expert_slice_size
            dram_size_GB = dram_size / (1024 * 1024 * 1024)
            print(f"chiplet: {chiplet.id}, dram size: {dram_size_GB:.0f} GB")

    def _add_event(self, event: SimulationEvent):
        if self.res_mgr.check_resource(event):
            self.res_mgr.request_resource(event)
            if PRINT_DEBUG:
                print("\tsuccessfully add event queue: ", event.event_type)
            heapq.heappush(self.event_queue, event)
        else:
            # Only attempt swap for high-priority events (priority > 0)
            # if event.priority > 0:
            #     self._try_swap_into_event_queue(event, self.current_time)
            if self.res_mgr.check_resource(event):
                self.res_mgr.request_resource(event)
                if PRINT_DEBUG:
                    print("\tswap-in event queue: ", event.event_type)
                heapq.heappush(self.event_queue, event)
            else:
                self.res_mgr.add_to_wait_queue(event)
                if PRINT_DEBUG:
                    print("\tfail to add event, need wait: ", event.event_type)
            
        return
        # if self.res_mgr.request_resource(event):
        #     if PRINT_DEBUG:
        #         print("\tsuccessfully add event queue: ", event.event_type)
        #     heapq.heappush(self.event_queue, event)
        # else:
        #     if PRINT_DEBUG:
        #         print("\tfail to add event, need wait: ", event.event_type)
        #     # print(f"adding event into wait queue @ {self.current_time}ns")
        #     pass
        # return

    def _parse_path(self, path: List[Chiplet]):
        path_list = []
        for chiplet in path:
            pos_x = chiplet.xidx
            pos_y = chiplet.yidx
            path_list.append((pos_x, pos_y))
        return path_list
    
    def _parse_link_resources(self, path: List[Chiplet]):
        # print("---------------jajajajajajaj----------------")
        path_link_resources = []
        for idx, chiplet1 in enumerate(path):
            if idx >= len(path) - 1:
                continue
            chiplet2 = path[idx + 1]
            link_id = tuple(sorted([chiplet1.id, chiplet2.id]))
            link_resource = self.res_mgr.resources[(ResourceType.D2D_LINK, link_id)]
            path_link_resources.append(link_resource)
        return path_link_resources

    def _cal_network_latency(self, path, data_size):
        use_rate = 0.9
        base_latency = self.timing_config.link_latency * (len(path) - 1)
        data_latency = math.ceil(data_size / (self.config.d2d_link_bw * use_rate))
        latency = base_latency + data_latency
        # print("cal_network_latency==============================")
        return latency
    
    def _parse_tree_link_resources(self, links):
        """D2D resources for a multicast tree. Every tree link is reserved once,
        which works unchanged because links are keyed undirected."""
        d2d = self.res_mgr.resources
        return [d2d[(ResourceType.D2D_LINK, link)] for link in links]

    def _cal_tree_latency(self, depth, data_size):
        """Latency for one tree delivery.

        The unicast form is 200*hops + size/bw. A tree pays the per-hop term
        along its longest root-to-leaf branch (the last leaf sets completion)
        and the serialisation term once, because the payload streams through
        the tree rather than being re-sent per destination.
        """
        use_rate = 0.9
        return (self.timing_config.link_latency * depth
                + math.ceil(data_size / (self.config.d2d_link_bw * use_rate)))

    def _cal_mem_latency(self, data_size, batch_size, is_remote=False):
        cold_start_latency = 0
        if self.config.model_name == "deepseek" or self.config.model_name == "kimi":
            # cold_start_latency = 6000
            cold_start_latency = 0
            if is_remote:
                cold_start_latency = 0
            usage_rate = 0.9
        elif self.config.model_name == "qwen":
            # print("batch", batch_size)
            # cold_start_latency = 7600
            cold_start_latency = 0
            if  is_remote:
                cold_start_latency = 0
            usage_rate = 0.9
            # if batch_size <= 64:
            #     usage_rate = 0.9
            # elif batch_size <= 256:
            #     usage_rate = 0.9
            # elif batch_size <= 1024:
            #     usage_rate = 0.9
        elif self.config.model_name == "llama4":
            usage_rate = 0.85
            cold_start_latency = 0
        else:
            raise ValueError("Unverified model")

        base_latency = self.timing_config.mem_access_latency
        data_latency = math.ceil(data_size / (self.config.dram_bw_per_die*usage_rate))
        latency = base_latency + data_latency + cold_start_latency
        # print("dram latency", latency)
        return latency
    
    def _cal_compute_latency(self, operations, batch_size):
        if (self.config.model_name == "deepseek" or self.config.model_name == "kimi"):
            usage_rate = 0.69
            if batch_size <= 128:
                usage_rate = 0.5
            elif batch_size <= 256:
                usage_rate = 0.53
            elif batch_size <= 512:
                usage_rate = 0.68
            elif batch_size <= 1024:
                usage_rate = 0.7
            elif batch_size <= 4096:
                usage_rate = 0.69
        elif self.config.model_name == "qwen":
            usage_rate = 0.70
            if batch_size <= 128:
                usage_rate = 0.7
            elif batch_size <= 256:
                usage_rate = 0.6
            elif batch_size <= 512:
                usage_rate = 0.7
            elif batch_size <= 1024:
                usage_rate = 0.7
            elif batch_size <= 2048:
                usage_rate = 0.69
            elif batch_size <= 4096:
                usage_rate = 0.71
        elif self.config.model_name == "llama4":
            usage_rate = 0.65
            if batch_size <= 128:
                usage_rate = 0.65
            elif batch_size <= 256:
                usage_rate = 0.65
            elif batch_size <= 512:
                usage_rate = 0.72
            elif batch_size <= 1024:
                usage_rate = 0.7
        else:
            raise ValueError(f"Undefined model: {self.config.model_name}")
        compute_latency = math.ceil(operations / (self.config.compte_power_per_die * 1000 * usage_rate)) # ns

        return compute_latency
    
    def _addr_trans(self, addr_expert: Addr_expert):

        slice_size = 1 * 1024 # unit KB
        matrix_size = 14 * 1024
        layer_size = 28 * 1024
        expert_size = 61 * layer_size

        expert_id = addr_expert.expert_id // self.config.chiplet_num
        layer_id = addr_expert.layer_id
        matrix_id = addr_expert.matrix_id
        slice_id = addr_expert.slice_id
        addr_chip = Addr_chip(
            chip_id=-1,
            addr_id=-1
        )

        addr_chip.chip_id = addr_expert.expert_id % self.config.chiplet_num
        addr_chip.addr_id = slice_id * slice_size + \
                            matrix_id * matrix_size + \
                            layer_id * layer_size + \
                            expert_id * expert_size
        return addr_chip

    def _expert_placement_map(self, address: Addr_expert):
        '''
            map each expert to a chiplet.
        '''
        # 0-63 in die0, 64-127 in die1, ...
        # org strategy
        expert_per_die = math.ceil(self.config.expert_num / self.config.chiplet_num)
        chiplet_id = math.floor(address.expert_id / expert_per_die)
        
        # new strategy
        expert_per_die = math.floor(self.config.expert_num / self.config.chiplet_num)
        chiplet_id = math.floor(address.expert_id / expert_per_die)
        max_suitable_expert_id = expert_per_die * self.config.chiplet_num - 1
        if address.expert_id > max_suitable_expert_id:
            chiplet_id = address.expert_id % self.config.chiplet_num
        # print(address.), chiplet_id) 
        return chiplet_id


    def _stat_prefill_pairs(self, all_data):
        heatmap = self.token_pred_heatmap
        for data in all_data:
            prefill_iter = data[0]
            for layer, expert_list_2d in prefill_iter.items():
                layer_id = int(layer)
                if expert_list_2d is None:
                    continue
                for token_id in range(len(expert_list_2d) - 1):
                    prev_arr = np.asarray(expert_list_2d[token_id], dtype=np.intp)
                    curr_arr = np.asarray(expert_list_2d[token_id + 1], dtype=np.intp)
                    row_idx = np.repeat(prev_arr, len(curr_arr))
                    col_idx = np.tile(curr_arr, len(prev_arr))
                    np.add.at(heatmap[layer_id], (row_idx, col_idx), 1)
        return

    def _stat_decode_cross_token_pairs(self, all_data, iter_id, layer_id):
        # Per-(iter_id, layer_id) version; kept for compatibility. Prefer _stat_decode_cross_token_pairs_bulk.
        if iter_id < 2:
            return
        layer = str(layer_id)
        for data in all_data:
            if iter_id >= len(data):
                continue
            current_iter = data[iter_id]
            previous_iter = data[iter_id - 1]
            if (layer not in current_iter) or (layer not in previous_iter):
                continue
            current_iter_experts_2d = current_iter[layer]
            previous_iter_experts_2d = previous_iter[layer]
            if (current_iter_experts_2d is None) or (previous_iter_experts_2d is None):
                continue
            for previous_expert in previous_iter_experts_2d[0]:
                for current_expert in current_iter_experts_2d[0]:
                    self.token_pred_heatmap[layer_id][previous_expert][current_expert] += 1
        return

    def _aggregate_decode_cross_token_pairs(self, all_data, max_iter_num, max_layer_num):
        """One-pass aggregation using numpy arrays.
        Returns a 4D array of shape (max_iter_num, max_layer_num, E, E) with cumulative counts."""
        E = self.config.expert_num

        # Step 1: incremental[iter_id, layer_id] = 2D histogram of (prev_e, curr_e)
        incremental = np.zeros((max_iter_num, max_layer_num, E, E), dtype=np.int32)

        # iter_id=1: seed with prefill heatmap
        for layer_id in range(max_layer_num):
            incremental[1, layer_id] = self.token_pred_heatmap[layer_id].astype(np.int32)

        # iter_id>=2: collect cross-iter pairs
        for data in all_data:
            for iter_id in range(2, min(len(data), max_iter_num)):
                current_iter = data[iter_id]
                previous_iter = data[iter_id - 1]
                for layer in current_iter:
                    if layer not in previous_iter:
                        continue
                    layer_id = int(layer)
                    if layer_id >= max_layer_num:
                        continue
                    current_experts = current_iter[layer]
                    previous_experts = previous_iter[layer]
                    if current_experts is None or previous_experts is None:
                        continue
                    prev_arr = np.asarray(previous_experts[0], dtype=np.intp)
                    curr_arr = np.asarray(current_experts[0], dtype=np.intp)
                    row_idx = np.repeat(prev_arr, len(curr_arr))
                    col_idx = np.tile(curr_arr, len(prev_arr))
                    np.add.at(incremental[iter_id, layer_id], (row_idx, col_idx), 1)

        # Step 2: cumulative sum along iter axis
        decode_pairs_np = np.cumsum(incremental, axis=0)

        return decode_pairs_np
        


    def _stat_prefill(self, iter: Dict[str, List]):
        for layer in range(self.config.layer_num):
            self.token_pred_table[str(layer)] = defaultdict(int)
            self.layer_pred_table[str(layer)] = defaultdict(int)

        for layer, expert_list_2d in iter.items():
            if expert_list_2d is None:
                continue
            for token_id, experts_1d in enumerate(expert_list_2d):
                if token_id + 1 >= len(expert_list_2d):
                    continue
                next_token_experts = expert_list_2d[token_id + 1]
                for expert in experts_1d:
                    for next_token_expert in next_token_experts:
                        expert_pair = tuple(sorted([expert, next_token_expert]))
                        self.token_pred_table[layer][expert_pair] += 1
        self.token_pred_table[layer] = dict(sorted(self.token_pred_table[layer].items(), key=lambda item: item[1], reverse=True))

        return

    def _is_prefill(self, iter: Dict[str, List]):
        for layer, expert_list_2d in iter.items():
            if expert_list_2d is not None:
                if len(expert_list_2d) > 1:
                    return True
        return False

    def _try_swap_into_event_queue(self, new_event: SimulationEvent, current_time: float) -> None:
        """If event_queue contains an event that uses the same resources and is strictly worse
        (lower priority, or same priority and later time), remove it, release its resources,
        and put it back on the resource wait queues so the new (better) request can run first.
        """
        key_new = frozenset((r.type, r.id) for r in new_event.resources)
        for i, ev in enumerate(self.event_queue):
            if frozenset((r.type, r.id) for r in ev.resources) != key_new:
                continue
            # new_event is better: higher priority, or same priority and strictly earlier time
            if new_event.priority > ev.priority or (
                new_event.priority == ev.priority and new_event.time < ev.time
            ):
                del self.event_queue[i]
                heapq.heapify(self.event_queue)
                for res in ev.resources:
                    self.res_mgr.release_resource(res, self.event_queue, current_time)
                self.res_mgr.add_to_wait_queue(ev)
                break

    def process_trace(self, trace_file):
        if isinstance(trace_file, list):
            if trace_file and isinstance(trace_file[0], dict):
                self.trace_generator = self._trace_reader_from_data(trace_file)
            else:
                self.trace_generator = self._trace_reader_list(trace_file)
        else:
            raise ValueError("trace_file must be a list of trace file paths or pre-loaded list of trace dicts")
        # for item in self.trace_generator:
        #     print(item)
        self._run_t0 = time.perf_counter()
        self._schedule_next_request()
        
        _loop_t0 = time.perf_counter()
        while self.event_queue:
            self._prof['event_loop_iterations'] += 1
                        
            event = heapq.heappop(self.event_queue)
            
            # information for debugging 
            # expert_id = event.address.expert_id
            # chiplet_id = event.chiplet_id
            # address = Addr_expert(expert_id=expert_id, layer_id=0, matrix_id=0, slice_id=0)
            # org_chiplet_id = self._expert_placement_map(address) # 
            # if event.event_type == EventType.NET_TRANS_REQ:
            #     if self.iter_id > 0:
            #         if org_chiplet_id == 10:
            #             print("network transfer event, exp:", event.address.expert_id, "chiplet: ", event.chiplet_id, "org_chiplet: ", org_chiplet_id, "time: ", event.time)
            # if event.event_type == EventType.MEMORY_ACCESS:
            #     if self.iter_id > 0:
            #         if org_chiplet_id == 10:
            #             print("exp: ", event.address.expert_id, "chiplet: ", chiplet_id, "org_chiplet: ", org_chiplet_id, "time: ", event.time)
             
            # if LOG:
            #     print("poped event, expert_id=%s, matirx_id=%s, type=%s, time=%d" \
            #         %(event.address.expert_id, event.address.matrix_id, event.event_type, event.time))
            
            # if self.current_stage == "decode":
            #     if self.iter_id >= 2:
            #         print("poped event: ", event.event_type, "evnet time=%d" % (event.time), "sys time=%d" % (self.current_time))
            
            # print("queue len after: ", len(self.event_queue))
            # for event0 in self.event_queue:
            #     print("\texpert:%d, event type:%s, time:%d" % (event0.address.expert_id, event0.event_type, event0.time))
            #     for resource in event0.resources:
            #         print("\t\t resource: ", resource.type, resource.id)
            # print("---------------------------------------------", event.event_type, event.time, self.current_time)
            
            # 更新时间需要放在回调前
            prev_time = self.current_time
            self.current_time = event.time
            # 继承 priority，供回调中创建的新 event 使用
            self._current_event_priority = getattr(event, "priority", 0.0)

            if event.callback:
                event.callback(event)

            sys_time_after_event = self.current_time
            self.current_time = max(prev_time, self.current_time)


            # release related resources
            # print(event.resources)
            for res in event.resources:
                self.res_mgr.release_resource(res, self.event_queue, sys_time_after_event)
            
            # allocate resource to waiting event and add then to event queue (res.queue head = highest priority, then earliest time)
            waiting_queue = []
            for res in event.resources:
            # for res in self.res_mgr.resources.values():
                waiting_event = self.res_mgr.peek_waiter(res)
                if waiting_event is not None:
                    heapq.heappush(waiting_queue, waiting_event)
            # from the earliest event
            while waiting_queue:
                # print("wait queue: %d, event queue: %d" % (len(waiting_queue), len(self.event_queue)))
                pending_event = heapq.heappop(waiting_queue)
                self.res_mgr.add_to_event_queue(pending_event, self.event_queue, sys_time_after_event)

        self._prof['event_loop'] += time.perf_counter() - _loop_t0

        # security check
        if SECURITY_CHECK:
            assert self.iter_id >= self.max_iter_num - 2, \
                "trace is not finished. expect iter=%d, actual iter=%d" % (self.max_iter_num, self.iter_id + 1)
        

    def _print_progress(self):
        """One line per PROGRESS_EVERY_ITERS iterations, with an ETA."""
        done = self.iter_id
        total = getattr(self, "max_iter_num", 0) or 1
        el = time.perf_counter() - getattr(self, "_run_t0", time.perf_counter())
        eta = (el / done * (total - done)) if done else float("nan")
        print(f"      [sim] iter {done:>4}/{total}  "
              f"events {self._prof['event_loop_iterations']:>10}  "
              f"sim {self.current_time / 1e6:9.1f} ms  "
              f"elapsed {el / 60:6.1f} min  eta {eta / 60:6.1f} min",
              flush=True)

    def _print_profiling(self):
        p = self._prof
        total = (p['stage_map'] + p['stat_prefill_pairs'] + p['aggregate_decode']
                 + p['trace_gen_yield'] + p['allocate_experts'] + p['event_loop']
                 + p['record_allocation'] + p['process_records'])
        print("\n" + "=" * 60)
        print("PROFILING BREAKDOWN (wall-clock seconds)")
        print("=" * 60)
        print(f"  stage_map generation     : {p['stage_map']:10.3f}s")
        print(f"  stat_prefill_pairs       : {p['stat_prefill_pairs']:10.3f}s")
        print(f"  aggregate_decode_pairs   : {p['aggregate_decode']:10.3f}s")
        print(f"  trace_gen_yield (heatmap): {p['trace_gen_yield']:10.3f}s")
        print(f"  allocate_experts (total) : {p['allocate_experts']:10.3f}s")
        print(f"  record_allocation (total): {p['record_allocation']:10.3f}s")
        print(f"  event_loop (total)       : {p['event_loop']:10.3f}s")
        print(f"    event_loop iterations  : {p['event_loop_iterations']}")
        event_loop_other = p['event_loop'] - p['allocate_experts'] - p['record_allocation']
        print(f"    event_loop - alloc/rec : {event_loop_other:10.3f}s")
        print(f"  process_records          : {p['process_records']:10.3f}s")
        print(f"  ---")
        print(f"  sum (approx total)       : {total:10.3f}s")
        print("=" * 60 + "\n")

    def process_records(self):
        _pr_t0 = time.perf_counter()
        # chiplet usage
        for iter_id in self.access_records:
            self.duration_records[iter_id] = {}
            for layer_id in self.access_records[iter_id]:
                self.duration_records[iter_id][layer_id] = {}
                for chiplet_id in range(self.config.chiplet_num):
                    self.duration_records[iter_id][layer_id][chiplet_id] = 0
                # print("duration init: ", self.duration_records[iter_id][layer_id])
        used_time = 0
        tot_time = 0
        tot_duration = 0
        sum_max_over_min = 0.0
        sum_max_over_avg = 0.0
        layer_count = 0
        for iter_id in self.access_records:
            for layer_id in self.access_records[iter_id]:
                chiplet_duration = {}
                chiplet_start_time = {}
                chiplet_end_time = {}
                for chiplet_id, span in self.access_records[iter_id][layer_id].items():
                    start_time, end_time = span
                    duration = end_time - start_time
                    self.duration_records[iter_id][layer_id][chiplet_id] = duration
                    chiplet_duration[chiplet_id] = duration
                    chiplet_start_time[chiplet_id] = start_time
                    chiplet_end_time[chiplet_id] = end_time
                
                durations = np.array(list(chiplet_duration.values()))
                max_duration = np.max(durations)
                min_duration = np.min(durations)
                avg_duration = np.average(durations)
                
                if min_duration > 0:
                    sum_max_over_min += max_duration / min_duration
                if avg_duration > 0:
                    sum_max_over_avg += max_duration / avg_duration
                layer_count += 1
                
                used_time += np.sum(durations)
                tot_time += max_duration * self.config.chiplet_num
        self.avg_chiplet_usage = used_time / tot_time
        if layer_count > 0:
            self.duration_max_over_min = sum_max_over_min / layer_count
            self.duration_max_over_avg = sum_max_over_avg / layer_count
        else:
            self.duration_max_over_min = 0
            self.duration_max_over_avg = 0
        
        # allocation evenness
        for chiplet_id in range(self.config.chiplet_num):
            self.allocation_summary[chiplet_id] = [0, 0]
        allo_exp_min = 0
        allo_exp_max = 0
        allo_exp_avg = 0
        allo_req_min = 0
        allo_req_max = 0
        allo_req_avg = 0
        for iter_id in self.allocation_records:
            for layer_id in self.allocation_records[iter_id]:
                allo_exp_list = np.zeros(self.config.chiplet_num)
                allo_req_list = np.zeros(self.config.chiplet_num)
                for chiplet_id in self.allocation_records[iter_id][layer_id]:
                    self.allocation_summary[chiplet_id][0] += self.allocation_records[iter_id][layer_id][chiplet_id][0]
                    self.allocation_summary[chiplet_id][1] += self.allocation_records[iter_id][layer_id][chiplet_id][1]
                    allo_exp_list[chiplet_id] += self.allocation_records[iter_id][layer_id][chiplet_id][0]
                    allo_req_list[chiplet_id] += self.allocation_records[iter_id][layer_id][chiplet_id][1]
                
                allo_exp_max += np.max(allo_exp_list)
                allo_exp_min += np.min(allo_exp_list)
                allo_exp_avg += np.average(allo_exp_list)
                
                allo_req_max += np.max(allo_req_list)
                allo_req_min += np.min(allo_req_list)
                allo_req_avg += np.average(allo_req_list)
                
        self.allo_exp_avg_over_max = allo_exp_avg / allo_exp_max
        self.allo_exp_min_over_max = allo_exp_min / allo_exp_max
        self.allo_req_avg_over_max = allo_req_avg / allo_req_max
        self.allo_req_min_over_max = allo_req_min / allo_req_max
        
        
        # dram access evenness
        evenness_list= []
        for iter_id in self.dram_access_records:
            for layer_id in self.dram_access_records[iter_id]:
                dram_access_for_all_chiplet = self.dram_access_records[iter_id][layer_id]
                max_access = np.max(dram_access_for_all_chiplet)
                avg_access = np.average(dram_access_for_all_chiplet)
                dram_access_avg_over_max = avg_access / max_access
                evenness_list.append(dram_access_avg_over_max)
                # print("dram access records: ", dram_access_for_all_chiplet)
        self.dram_access_avg_over_max = np.average(evenness_list)   
        # print(dram_access_avg_over_max)     
        self._prof['process_records'] += time.perf_counter() - _pr_t0
        if ENABLE_PROFILING:
            self._print_profiling()
        return

    def export_expert_analysis(self, output_dir, strategy_name, batch_size):
        """Write logical expert popularity and chiplet execution assignments.

        Files are per simulation configuration so experiments with different
        strategies, chiplet shapes, or batches never overwrite one another.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shape = f"{self.config.y_chiplets}x{self.config.x_chiplets}"
        stem = f"{self.config.model_name}_{strategy_name}_{shape}_batch{batch_size}"

        totals = Counter()
        for (stage, layer_id, _expert_id), count in self.expert_selection_records.items():
            totals[(stage, layer_id)] += count

        usage_path = output_dir / f"{stem}_expert_usage.csv"
        with usage_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "stage", "layer", "expert_id", "selection_count",
                "selection_share", "home_chiplet",
            ])
            writer.writeheader()
            rows = sorted(self.expert_selection_records.items(),
                          key=lambda item: (item[0][0], item[0][1], -item[1], item[0][2]))
            for (stage, layer_id, expert_id), count in rows:
                address = Addr_expert(expert_id=expert_id, layer_id=layer_id,
                                      matrix_id=0, slice_id=0)
                writer.writerow({
                    "stage": stage,
                    "layer": layer_id,
                    "expert_id": expert_id,
                    "selection_count": count,
                    "selection_share": f"{count / totals[(stage, layer_id)]:.8f}",
                    "home_chiplet": self._expert_placement_map(address),
                })

        allocation_path = output_dir / f"{stem}_expert_allocation.csv"
        with allocation_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "stage", "layer", "expert_id", "executing_chiplet",
                "request_count", "request_share_for_expert", "home_chiplet", "is_remote",
            ])
            writer.writeheader()
            rows = sorted(self.expert_execution_records.items(),
                          key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
            for (stage, layer_id, expert_id, chiplet_id), count in rows:
                address = Addr_expert(expert_id=expert_id, layer_id=layer_id,
                                      matrix_id=0, slice_id=0)
                home_chiplet = self._expert_placement_map(address)
                total = self.expert_selection_records[(stage, layer_id, expert_id)]
                writer.writerow({
                    "stage": stage,
                    "layer": layer_id,
                    "expert_id": expert_id,
                    "executing_chiplet": chiplet_id,
                    "request_count": count,
                    "request_share_for_expert": f"{count / total:.8f}" if total else "0",
                    "home_chiplet": home_chiplet,
                    "is_remote": chiplet_id != home_chiplet,
                })

        return usage_path, allocation_path

    def _trace_reader_list(self, trace_path_list):
        
        if self.gemm_verify_mode:
            # print("GEMM verify mode")
            self.tot_batch_size = len(trace_path_list)
            expert_cnt = {0:self.tot_batch_size}
            yield dict(expert_cnt), 1, "decode", [0]
            return
        elif self.p2p_verify_mode:
            self.tot_batch_size = len(trace_path_list)
            expert_cnt = {0:self.tot_batch_size}
            yield dict(expert_cnt), 1, "decode", [0]
            return
        
        all_data = []
        data_len_list = []
        prefill_len_list = []
        self.tot_batch_size = len(trace_path_list)
        path_list = trace_path_list[0:self.tot_batch_size]
        # If caller passed pre-loaded list of traces (each trace is list/dict of iters), use from_data path
        if path_list and isinstance(path_list[0], (list, dict)) and not isinstance(path_list[0], str):
            yield from self._trace_reader_from_data(path_list)
            return
        if path_list and not isinstance(path_list[0], str):
            raise TypeError(
                "_trace_reader_list expects list of path strings; got list of %s. "
                "For pre-loaded data use process_trace(list_of_dicts) instead."
                % type(path_list[0]).__name__
            )

        def _load_one_trace(idx_path):
            idx, path = idx_path
            with open(path, "r", encoding="utf-8") as f:
                return (idx, json.load(f))

        max_workers = min(32, len(path_list), (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_load_one_trace, enumerate(path_list)))
        results.sort(key=lambda x: x[0])
        all_data = [r[1] for r in results]
        data_len_list = [len(d) for d in all_data]

        yield from self._trace_reader_core(all_data)

    def _trace_reader_from_data(self, trace_data_list):
        """Run trace generator from pre-loaded list of trace dicts (no file I/O)."""
        self.tot_batch_size = len(trace_data_list)
        yield from self._trace_reader_core(trace_data_list)

    def _trace_reader_core(self, all_data):
        """Aggregate trace data and yield (expert_dict, layer, stage, prefill_len_list) per (iter, layer)."""

        # ---- Step 1: Determine iteration / layer bounds ---- #
        prefill_len_list = []
        data_len_list = [len(d) for d in all_data]
        # max_iter_num = max(data_len_list)
        # max_iter_num = max(max_iter_num, 128)
        max_iter_num = 128
        if LIMIT_MAX_ITER_NUM > 0:
            max_iter_num = LIMIT_MAX_ITER_NUM

        # print("length of first trace: ", len(all_data[0]))
        # print("length of all data: ", len(all_data))
        # print("type of all data: ", type(all_data))
        max_layer_num = len(all_data[0][1])


        # if all_data:
        #     first_trace = all_data[0]
        #     if isinstance(first_trace, list) and len(first_trace) > 0:
        #         one_iter = first_trace[0]
        #     else:
        #         one_iter = next(iter(first_trace.values()))
        #     max_layer_num = len(one_iter)
        # else:
        #     max_layer_num = 0
        self.max_iter_num = max_iter_num
        self.max_layer_num = max_layer_num

        # ---- Step 2: Aggregate traces from all requests ---- #
        # Merge per-request traces into a single (iter, layer) -> expert counter
        # map, and record whether each (iter, layer) is prefill or decode.
        _t0 = time.perf_counter()
        aggregated = defaultdict(lambda: defaultdict(Counter))
        stage_map = {}

        # TOKEN_PARALLEL: tokens are pinned to dies, so we also need the expert
        # demand *per die*, not just globally. Requests are split into
        # contiguous blocks, one block per die. Only built for this strategy --
        # the array is ~150 MB at qwen/24 dies and nothing else reads it.
        token_parallel = (self.config.allocation_strategy
                          == self.config.AllocationStrategy.TOKEN_PARALLEL)
        if token_parallel:
            self.die_expert_counts = np.zeros(
                (max_iter_num, max_layer_num, self.config.chiplet_num,
                 self.config.expert_num), dtype=np.int32)

        for trace_idx, data in enumerate(all_data):             # loop over requests
            if token_parallel:
                die_of_trace = trace_idx * self.config.chiplet_num // len(all_data)
                ti, tl, te = [], [], []
            for iter_id, iter_data in enumerate(data):          # loop over iterations (tokens)
                if iter_id >= max_iter_num:
                    break
                for layer, expert_list_2d in iter_data.items(): # loop over layers
                    if expert_list_2d is None:
                        continue
                    layer_id = int(layer)
                    if layer_id >= max_layer_num:
                        continue
                    # Classify stage: iter 0 with multiple tokens -> prefill
                    if len(expert_list_2d) > 1 and iter_id == 0:
                        stage = "prefill"
                    elif len(expert_list_2d) == 1:
                        stage = "decode"
                    else:
                        raise ValueError("Invalid trace format: multiple prefill stages")
                    # Accumulate expert selection counts across all tokens
                    for expert_list_1d in expert_list_2d:
                        aggregated[iter_id][layer_id].update(expert_list_1d)
                        if token_parallel:
                            n_sel = len(expert_list_1d)
                            ti.extend([iter_id] * n_sel)
                            tl.extend([layer_id] * n_sel)
                            te.extend(expert_list_1d)
                    stage_map[(iter_id, layer_id)] = stage
            # One batched scatter-add per request, not one per token row:
            # ~4k numpy calls instead of ~50M. Measured ~90s at batch 4096.
            if token_parallel and ti:
                np.add.at(self.die_expert_counts[:, :, die_of_trace, :],
                          (ti, tl, te), 1)
        self._prof['stage_map'] += time.perf_counter() - _t0

        # ---- Step 3: Build prediction heatmaps (if prediction enabled) ---- #
        # print("pred_next_token: ", self.config.pred_next_token)
        # print("pred_next_layer: ", self.config.pred_next_layer)

        # Collect cross-token expert co-occurrence stats from prefill phase
        if self.config.pred_next_token or self.config.pred_next_layer:
            _t0 = time.perf_counter()
            self._stat_prefill_pairs(all_data)
            self._prof['stat_prefill_pairs'] += time.perf_counter() - _t0

        # Build per-(iter, layer) prediction matrices for decode phase
        decode_pairs = None
        if self.config.pred_next_token or self.config.pred_next_layer:
            _t0 = time.perf_counter()
            decode_pairs = self._aggregate_decode_cross_token_pairs(all_data, max_iter_num, max_layer_num)
            self._prof['aggregate_decode'] += time.perf_counter() - _t0

        # ---- Step 4: Yield one (iter, layer) at a time to the simulator ---- #
        _t0 = time.perf_counter()
        for iter_id in range(max_iter_num):
            for layer_id in range(max_layer_num):
                # Update prediction heatmap for decode tokens (iter >= 1)
                if decode_pairs is not None and iter_id >= 1:
                    self.token_pred_heatmap[layer_id] = decode_pairs[iter_id, layer_id].astype(self.token_pred_heatmap.dtype)
                expert_cnt = aggregated[iter_id][layer_id]
                if not expert_cnt:
                    continue
                stage = stage_map.get((iter_id, layer_id), "decode")
                self.iter_id = iter_id
                self.layer_id = layer_id
                self._prof['trace_gen_yield'] += time.perf_counter() - _t0
                yield dict(expert_cnt), str(layer_id), stage, prefill_len_list
                _t0 = time.perf_counter()

    def _record_allocation(self, chiplet_id, batch_size):
        if self.iter_id not in self.allocation_records:
            self.allocation_records[self.iter_id] = {}
        if self.layer_id not in self.allocation_records[self.iter_id]:
            self.allocation_records[self.iter_id][self.layer_id] = {}
            for chiplet_id_temp in range(self.config.chiplet_num):
                self.allocation_records[self.iter_id][self.layer_id][chiplet_id_temp] = [0, 0]
        # print("chiplet id: ", chiplet_id)
        # print("before: ", self.allocation_records[self.iter_id][self.layer_id])
        self.allocation_records[self.iter_id][self.layer_id][chiplet_id][0] += 1
        self.allocation_records[self.iter_id][self.layer_id][chiplet_id][1] += batch_size
        # print("after: ", self.allocation_records[self.iter_id][self.layer_id])
        return
        
    def _allocate_request(self, expert_id, allocated_req_num, allocated_exp_num):
        '''
            allocate a request using expert x to a specific chiplet
        '''
        if self.config.allocation_strategy == self.config.AllocationStrategy.RANDOM:
            chiplet_id = expert_id % self.config.chiplet_num        
            return chiplet_id
        elif self.config.allocation_strategy == self.config.AllocationStrategy.NEAREST:
            address = Addr_expert(expert_id=expert_id, layer_id=0, matrix_id=0, slice_id=0)
            chiplet_id = self._expert_placement_map(address)
            return chiplet_id
        elif self.config.allocation_strategy == self.config.AllocationStrategy.EVEN:
            chiplet_id = allocated_req_num % self.config.chiplet_num
            return chiplet_id
        elif self.config.allocation_strategy == self.config.AllocationStrategy.EXP_EVEN:
            chiplet_id = (allocated_exp_num+7) % self.config.chiplet_num
            return chiplet_id

    def _cal_allo_cost_org(self, my_load_of_chiplets, allo_chiplet_idx, data_chiplet_idx, allocated_req_num, allocated_before):
        # print("idxs: ", allo_chiplet_idx, data_chiplet_idx)
        def cal_compute_latency(batch_size):
            operations = 2 * 2 * self.config.hidden_dim * self.config.expert_dim * batch_size / \
                    self.config.expert_tot_slice_num
            compute_latency = math.ceil(operations / (self.config.compte_power_per_die * 1000)) # ns
            return compute_latency        
        is_allo_local = allo_chiplet_idx == data_chiplet_idx
        new_load_of_chiplets = copy.deepcopy(my_load_of_chiplets)
        compute_latency = cal_compute_latency(allocated_req_num)
        dram_latency = self.timing_config.mem_access_latency + self.config.expert_slice_size / self.config.dram_bw_per_die
        d2d_latency = 2 * self.timing_config.link_latency + self.config.expert_slice_size / self.config.d2d_link_bw

        if allocated_before:
            dram_latency = 0
        if is_allo_local:
            d2d_latency = 0            
            new_load_of_chiplets[allo_chiplet_idx] += compute_latency + dram_latency + d2d_latency
        else:
            new_load_of_chiplets[allo_chiplet_idx] += compute_latency + d2d_latency
            new_load_of_chiplets[data_chiplet_idx] += dram_latency
            
        return new_load_of_chiplets

    def _allocate_experts_org(self, expert_dict, layer, stage):
        debug_allo = False
        sum_table_en = False
        
        # ctrl
        # debug_allo = True
        # sum_table_en = True
        if debug_allo or sum_table_en:
            print("iter = %d, layer = %d, stage = %s ==================="%(self.iter_id, self.layer_id, stage))
        import statistics
        counts = list(expert_dict.values())

        max_count = max(counts)
        min_count = min(counts)
        mean_count = float(sum(counts)) / float(self.config.expert_num)
        median_count = statistics.median(counts)
        
        if debug_allo:
            print("count for expert dict. max=%d, min=%d, median=%.2f, mean=%.4f, tot_count=%d" % (max_count, min_count, median_count, mean_count, sum(counts)))
         
        exp_allo_plan = []
        if self.gemm_verify_mode:
            for expert_id in expert_dict:
                req_num = expert_dict[expert_id]
                chiplet_id = 0
                exp_allo_plan.append([expert_id, req_num, chiplet_id])
            return exp_allo_plan
        elif self.p2p_verify_mode:
            for expert_id in expert_dict:
                req_num = expert_dict[expert_id]
                chiplet_id = 1
                exp_allo_plan.append([expert_id, req_num, chiplet_id])
            return exp_allo_plan
        
        if self.config.allocation_strategy in [self.config.AllocationStrategy.RANDOM, 
                                               self.config.AllocationStrategy.NEAREST, 
                                               self.config.AllocationStrategy.EVEN, 
                                               self.config.AllocationStrategy.EXP_EVEN]:
            allocated_req_num = 0
            allocated_expert_num = 0
            for expert_id in expert_dict:
                req_num = expert_dict[expert_id]
                chiplet_id = self._allocate_request(int(expert_id), allocated_req_num, allocated_expert_num)
                exp_allo_plan.append([expert_id, req_num, chiplet_id])
                allocated_expert_num += 1
                allocated_req_num += req_num
        elif self.config.allocation_strategy == self.config.AllocationStrategy.OURS_ORG:
            
            load_of_chiplets = np.zeros(self.config.chiplet_num)
            
            nearest_allo_dram_cost  = np.zeros(self.config.chiplet_num)
            for expert_id in expert_dict:
                org_chiplet_id = self.org_exp_chiplet_distribution[expert_id][0]
                dram_latency = self.timing_config.mem_access_latency + self.config.expert_slice_size / self.config.dram_bw_per_die
                nearest_allo_dram_cost[org_chiplet_id] += dram_latency
            
            exp_chiplet_distribution = self._gen_exp_chiplet_distribution(int(layer))
            
            expert_dict = dict(sorted(expert_dict.items(), key=lambda x: x[1], reverse=False))
            for expert_id in expert_dict:
                req_num = expert_dict[expert_id]
                local_chiplet_list = exp_chiplet_distribution[expert_id]
                address = Addr_expert(expert_id=expert_id, layer_id=0, matrix_id=0, slice_id=0)
                org_chiplet_id = self._expert_placement_map(address)
                tmp_remote_chiplet_map = self._gen_remote_chiplet_list([org_chiplet_id], distance=1)
                remote_chiplet_map = {}
                for remote_chiplet_idx in tmp_remote_chiplet_map:
                    if remote_chiplet_idx not in local_chiplet_list:
                        remote_chiplet_map[remote_chiplet_idx] = tmp_remote_chiplet_map[remote_chiplet_idx]
                remote_chiplet_list = list(remote_chiplet_map.keys())
                if debug_allo:
                    print("expert_id=%d, req_num=%d --------------------------" % (expert_id, req_num))
                    print("\torg chiplet id: ", org_chiplet_id)
                    print("\tlocal chiplet list: ", local_chiplet_list)
                    print("\tremote chiplet map: ", remote_chiplet_map)
                merge_list = []
                for chiplet_id in local_chiplet_list:
                    merge_list.append((load_of_chiplets[chiplet_id], chiplet_id))
                
                for chiplet_id in remote_chiplet_map:
                    if chiplet_id not in local_chiplet_list:
                        remote_extra_cost = self.timing_config.link_latency + self.config.expert_size / self.config.d2d_link_bw
                        base_cost = max(load_of_chiplets[chiplet_id], nearest_allo_dram_cost[chiplet_id])
                        tot_cost = base_cost + remote_extra_cost
                        merge_list.append((tot_cost, chiplet_id))
                merge_list.sort()
                sorted_chiplet_list = [chiplet_id for _, chiplet_id in merge_list]
                split_num = max(mean_count, 300)
                candidate_num = int(max(1, req_num // split_num))
                candidate_num = min(len(merge_list), candidate_num)
                candidate_chiplet_list = sorted_chiplet_list[:candidate_num]
                
                relevant_list = candidate_chiplet_list
                        
                if debug_allo:
                    print("\tsplit num=%d, req_num=%d, threshod=%d "%(candidate_num, req_num, split_num))
                    print("\tcandidate chiplet list: ", candidate_chiplet_list)

                req_block_size = 256
                req_of_candidate_chiplets = [0 for _ in range(candidate_num)]
                is_chiplet_allocated_list = [False for _ in range(candidate_num)]
                tmp_load_of_chiplets = load_of_chiplets.copy()
                remaining_req = req_num
                while remaining_req > 0:
                    allocation_req_num = min(req_block_size, remaining_req)
                    cost_of_allo_to_each_candidate = np.zeros(candidate_num)
                    for idx_in_candidate_list, candidate_chiplet_idx in enumerate(candidate_chiplet_list):
                        tmp_tmp_load_of_chiplets = tmp_load_of_chiplets.copy()
                        for chiplet_id in range(len(tmp_tmp_load_of_chiplets)):
                            tmp_tmp_load_of_chiplets[chiplet_id] = max(tmp_tmp_load_of_chiplets[chiplet_id], nearest_allo_dram_cost[chiplet_id])
                        if candidate_chiplet_idx in local_chiplet_list:
                            data_chiplet_idx = candidate_chiplet_idx
                        else:
                            data_chiplet_idx = remote_chiplet_map[candidate_chiplet_idx][0]
                            
                        tmp_tmp_load_of_chiplets = self._cal_allo_cost_org(
                            tmp_tmp_load_of_chiplets,
                            allo_chiplet_idx = candidate_chiplet_idx,
                            data_chiplet_idx = data_chiplet_idx,
                            allocated_req_num = allocation_req_num,
                            allocated_before = is_chiplet_allocated_list[idx_in_candidate_list],
                        )    
                        cost = np.max(tmp_tmp_load_of_chiplets[relevant_list])
                        cost_of_allo_to_each_candidate[idx_in_candidate_list] = cost
                    min_idx_in_candidate_list = np.argmin(cost_of_allo_to_each_candidate)
                    allo_chiplet_idx = candidate_chiplet_list[min_idx_in_candidate_list]
                    
                    if allo_chiplet_idx in local_chiplet_list:
                        data_chiplet_idx = allo_chiplet_idx
                    else:
                        data_chiplet_idx = remote_chiplet_map[allo_chiplet_idx][0]
                        
                    tmp_load_of_chiplets = self._cal_allo_cost_org(
                        tmp_load_of_chiplets,
                        allo_chiplet_idx = allo_chiplet_idx,
                        data_chiplet_idx = data_chiplet_idx,
                        allocated_req_num = allocation_req_num,
                        allocated_before = is_chiplet_allocated_list[min_idx_in_candidate_list],
                    )     
                    is_chiplet_allocated_list[min_idx_in_candidate_list] = True
                    req_of_candidate_chiplets[min_idx_in_candidate_list] += allocation_req_num
                    remaining_req -= allocation_req_num

                for i, allo_chiplet_idx in enumerate(candidate_chiplet_list):
                    req_of_one_candidate = req_of_candidate_chiplets[i]
                    if req_of_one_candidate != 0:                        
                        if allo_chiplet_idx in local_chiplet_list:
                            data_chiplet_idx = allo_chiplet_idx
                        else:
                            data_chiplet_idx = remote_chiplet_map[allo_chiplet_idx][0]    
                        load_of_chiplets = self._cal_allo_cost_org(
                            load_of_chiplets,
                            allo_chiplet_idx = allo_chiplet_idx,
                            data_chiplet_idx = data_chiplet_idx,
                            allocated_req_num = req_of_one_candidate,
                            allocated_before = False,
                        )    

                        exp_allo_plan.append([expert_id, req_of_one_candidate, allo_chiplet_idx])
                        if debug_allo:
                            print("\t\tkernel: expert_id=%d, mini_req_num=%d, chiplet_id=%d" % (expert_id, req_of_one_candidate, allo_chiplet_idx))
                
                if debug_allo:
                    print("\n")
        
        unexpected_allo = []
        if sum_table_en:
            sum_table = {}
            for expert_id in range(self.config.expert_num):
                org_die = self.org_exp_chiplet_distribution[expert_id][0]
                sum_table[expert_id] = [org_die, [], []]
            for plan_for_one_exp in exp_allo_plan:
                expert_id, req_num, chiplet_id = plan_for_one_exp
                sum_table[expert_id][1].append(chiplet_id)
                sum_table[expert_id][2].append(req_num)
            print("summary table for iter %d, layer %d: " % (self.iter_id, self.layer_id))
            for expert_id, allo_summary in sum_table.items():
                org_die, allo_dies, allo_reqs = allo_summary
                print("exp=%s org_die=%d allo_dies: " %(expert_id, org_die), allo_dies, " ", allo_reqs)
                
                if self.current_stage == "decode":
                    if len(allo_dies) > 1 or (len(allo_dies) == 1 and allo_dies[0] != org_die):
                        unexpeted_info = "exp=%s org_die=%d allo_dies: " %(expert_id, org_die) + str(allo_dies) + " " + str(allo_reqs)
                        unexpected_allo.append(unexpeted_info)
                        
            print("\n")
            if self.current_stage == "decode":
                print("unexpected allocation for iter %d, layer %d: " % (self.iter_id, self.layer_id))
                for unexpeted_info in unexpected_allo:
                    print(unexpeted_info)
                print("\n")
            
        return exp_allo_plan

    def _allocate_experts(self, expert_dict, layer, stage):

        
        # --- profiling ---
        counts = list(expert_dict.values())
        mean_count = float(sum(counts)) / float(self.config.expert_num)

        if DEBUG_ALLO or SUM_TABLE_EN:
            print("iter = %d, layer = %d, stage = %s ===================" % (self.iter_id, self.layer_id, stage))
        if DEBUG_ALLO:
            print("count for expert dict. max=%d, min=%d, median=%.2f, mean=%.4f, tot_count=%d"
                  % (max(counts), min(counts), statistics.median(counts), mean_count, sum(counts)))

        # --- verify modes (for testing) ---
        exp_allo_plan = []
        if self.gemm_verify_mode:
            for expert_id, req_num in expert_dict.items():
                exp_allo_plan.append([expert_id, req_num, 0])
            return exp_allo_plan
        if self.p2p_verify_mode:
            for expert_id, req_num in expert_dict.items():
                exp_allo_plan.append([expert_id, req_num, 1])
            return exp_allo_plan

        # --- simple strategies ---
        if self.config.allocation_strategy in [self.config.AllocationStrategy.RANDOM,
                                               self.config.AllocationStrategy.NEAREST,
                                               self.config.AllocationStrategy.EVEN,
                                               self.config.AllocationStrategy.EXP_EVEN]:
            allocated_req_num = 0
            allocated_expert_num = 0
            for expert_id, req_num in expert_dict.items():
                chiplet_id = self._allocate_request(int(expert_id), allocated_req_num, allocated_expert_num)
                exp_allo_plan.append([expert_id, req_num, chiplet_id])
                allocated_expert_num += 1
                allocated_req_num += req_num

        # --- TOKEN_PARALLEL: tokens pinned, each die runs its own expert union ---
        elif self.config.allocation_strategy == self.config.AllocationStrategy.TOKEN_PARALLEL:
            exp_allo_plan = self._allocate_experts_token_parallel()

        # --- OURS strategy ---
        elif self.config.allocation_strategy == self.config.AllocationStrategy.OURS:
            exp_allo_plan = self._allocate_experts_ours(expert_dict, layer, mean_count, DEBUG_ALLO)

        # --- sort by expert_id ascending (consistent event ordering across strategies) ---
        exp_allo_plan.sort(key=lambda x: x[0])

        # --- summary table ---
        if SUM_TABLE_EN:
            self._print_allocation_summary(exp_allo_plan)

        return exp_allo_plan

    # ------------------------------------------------------------------ #
    #  Latency helpers (single source of truth for cost formulas)        #
    # ------------------------------------------------------------------ #

    def _allocate_experts_token_parallel(self):
        """Data-parallel placement: tokens stay on their home die, so a die must
        run every expert its own tokens selected.

        Unlike the other strategies this makes no placement *decision* -- the
        plan is read straight off the per-die routing demand. One expert is
        therefore requested by many dies in the same layer, which is the
        multicast fanout the other strategies never produce.
        """
        counts = self.die_expert_counts[self.iter_id, self.layer_id]  # (dies, experts)
        plan = []
        for die_id in range(self.config.chiplet_num):
            row = counts[die_id]
            for expert_id in np.nonzero(row)[0]:
                plan.append([int(expert_id), int(row[expert_id]), die_id])
        return plan

    def _compute_lat(self, batch_size):
        """Compute latency (ns) for batch_size requests on one whole expert."""
        ops = (2 * self.config.hidden_dim * self.config.expert_dim
               * batch_size * self.config.expert_matrix_num)
        return math.ceil(ops / (self.config.compte_power_per_die * 1000 * 0.7))

    def _dram_read_lat(self):
        """DRAM read latency (ns) for loading one whole expert from local DRAM."""
        return (self.timing_config.mem_access_latency
                + self.config.expert_size / (self.config.dram_bw_per_die * 0.9))

    def _d2d_transfer_lat(self, hops=1):
        """D2D transfer latency (ns) for one whole expert over a multi-hop path.
        All links on the path are occupied simultaneously; data flows through at
        link bandwidth, with per-hop startup latency."""
        return (2 * hops * self.timing_config.link_latency # read instruction latency + transfer latency
                + self.config.expert_size / (self.config.d2d_link_bw * 0.9))

    def _manhattan_distance(self, cid_a, cid_b):
        """Manhattan distance between two chiplet IDs on the mesh."""
        ca = self.interconnect.chiplets[cid_a]
        cb = self.interconnect.chiplets[cid_b]
        return abs(ca.xidx - cb.xidx) + abs(ca.yidx - cb.yidx)

    # ------------------------------------------------------------------ #
    #  OURS: greedy load-balanced allocation with overlap-aware cost model #
    # ------------------------------------------------------------------ #

    def _init_die_state(self):
        """Create per-die state arrays tracking compute / dram / d2d breakdown."""
        n = self.config.chiplet_num
        return {
            'compute_local': np.zeros(n),   # compute time for local experts (weights on this die)
            'compute_remote': np.zeros(n),  # compute time for remote experts (weights arrive via D2D)
            'dram_local': np.zeros(n),      # DRAM read time for experts computed locally on this die
            'dram_remote': np.zeros(n),     # DRAM read time for experts sent to remote dies for computation
            'd2d_in': np.zeros(n),          # total D2D receive time (weights arriving from other dies)
            'd2d_out': np.zeros(n),         # total D2D send time (weights sent to other dies)
        }

    def _die_estimated_time(self, state, die_ids=None):
        """Estimated wall time per die considering overlap.

        Timeline on one die (single compute unit, separate DRAM & link HW):
          - local compute overlaps with dram_local reads (different HW units)
          - remote compute must wait for d2d_in to finish (weights must arrive first)
          - compute unit is shared: local and remote compute are sequential
          - dram_remote + d2d_out are sequential (read from DRAM then send over link)
          - dram_local and dram_remote share the DRAM controller (sequential)

        Path 1 (compute):
            local_phase  = max(compute_local, dram_local)  -- local compute overlaps with local DRAM reads
            remote_start = max(local_phase, d2d_in)        -- remote compute waits for weights + compute unit
            compute_path = remote_start + compute_remote

        Path 2 (memory/link serving remote requesters):
            memory_path  = dram_local + dram_remote + d2d_out
            dram_local must also finish (DRAM is shared), then dram_remote reads, then D2D sends.

        die_time = max(compute_path, memory_path)
        """
        cl = state['compute_local']
        cr = state['compute_remote']
        dram_l = state['dram_local']
        dram_r = state['dram_remote']
        d2d_in = state['d2d_in']
        d2d_out = state['d2d_out']

        # Path 1: compute pipeline
        local_phase = np.maximum(cl, dram_l + dram_r)

        # Path 2: memory serving pipeline (all DRAM reads + D2D sends are sequential)
        # memory_path = dram_l + dram_r + d2d_out

        t = np.maximum(local_phase, d2d_in) + cr

        if die_ids is not None:
            return t[die_ids]
        return t

    def _copy_die_state(self, state):
        return {k: v.copy() for k, v in state.items()}

    def _allocate_experts_ours(self, expert_dict, layer, mean_count, debug):
        exp_allo_plan = []
        dram_lat_per_expert = self._dram_read_lat()

        state = self._init_die_state()
        # Baseline: every active expert's home die must read weights from DRAM (local read)
        for expert_id in expert_dict:
            home = self.org_exp_chiplet_distribution[expert_id][0]
            state['dram_local'][home] += dram_lat_per_expert #TEMP
            # state['dram_local'][home] += 0
        
        # print("state_dram: ", state['dram'])
        exp_chiplet_distribution = self._gen_exp_chiplet_distribution(int(layer))
        sorted_experts = sorted(expert_dict.items(), key=lambda x: (x[1], x[0]), reverse=False)

        for expert_id, req_num in sorted_experts:
            local_chiplets = exp_chiplet_distribution[expert_id]
            local_set = set(local_chiplets)
            address = Addr_expert(expert_id=expert_id, layer_id=0, matrix_id=0, slice_id=0)
            org_chiplet_id = self._expert_placement_map(address)

            raw_remote_map = self._gen_remote_chiplet_list([org_chiplet_id], distance=1)
            remote_map = {cid: srcs for cid, srcs in raw_remote_map.items()
                          if cid not in local_set}

            if debug:
                print("expert_id=%d, req_num=%d --------------------------" % (expert_id, req_num))
                print("\torg chiplet id: ", org_chiplet_id)
                print("\tlocal chiplet list: ", local_chiplets)
                print("\tremote chiplet map: ", remote_map)

            # Decide whether remote candidates are worth considering
            d2d_cost = self._d2d_transfer_lat()
            local_times = self._die_estimated_time(state, list(local_chiplets))
            min_local_time = float(np.min(local_times))
            remote_times = {rid: float(self._die_estimated_time(state, [rid])[0]) for rid in remote_map}
            remote_helps = any(min_local_time - rt > d2d_cost for rt in remote_times.values())
            # remote_helps = True
            
            compute_for_expert = self._compute_lat(req_num)
            allow_remote = compute_for_expert > d2d_cost and remote_helps
            allow_remote = True

            if allow_remote:
                candidates = self._build_candidate_list_v2(local_chiplets, remote_map, state)
            else:
                local_t = self._die_estimated_time(state, list(local_chiplets))
                candidates = sorted(zip(local_t, local_chiplets))

            split_threshold = max(mean_count, 512)
            num_candidates = int(max(5, req_num // split_threshold))
            num_candidates = min(len(candidates), num_candidates)
            
            # num_candidates = 5
            candidate_ids = [cid for _, cid in candidates[:num_candidates]]

            if debug:
                print("\tsplit num=%d, req_num=%d, threshod=%d" % (num_candidates, req_num, split_threshold))
                print("\tcandidate chiplet list: ", candidate_ids)

            alloc_per_candidate, state = self._greedy_block_assign_v2(
                req_num, candidate_ids, local_set, remote_map, num_candidates, state)

            for idx, cid in enumerate(candidate_ids):
                assigned = alloc_per_candidate[idx]
                if assigned == 0:
                    continue
                exp_allo_plan.append([expert_id, assigned, cid])
                if debug:
                    print("\t\tkernel: expert_id=%d, mini_req_num=%d, chiplet_id=%d"
                          % (expert_id, assigned, cid))

            if debug:
                print()

        if self.rebalance_enabled:
            exp_allo_plan = self._rebalance_pass(exp_allo_plan, expert_dict, debug)

        return exp_allo_plan

    # ------------------------------------------------------------------ #
    #  Rebalance pass: migrate entries from overloaded dies               #
    # ------------------------------------------------------------------ #

    def _rebuild_state_from_plan(self, exp_allo_plan, expert_dict):
        """Reconstruct per-die cost state from a complete allocation plan.

        Unlike the greedy baseline which conservatively pre-charges dram_local
        for every active expert, this rebuild only charges dram_local when the
        expert actually has local computation.  dram_remote is charged once per
        unique remote destination die (each remote die requires a separate DRAM
        read to send the expert weights over D2D).
        """
        state = self._init_die_state()
        dram_lat_per_expert = self._dram_read_lat()

        has_local = set()
        remote_pairs = set()
        for expert_id, _req_num, allo_chiplet in exp_allo_plan:
            data_chiplet = self.org_exp_chiplet_distribution[expert_id][0]
            if allo_chiplet == data_chiplet:
                has_local.add(expert_id)
            else:
                remote_pairs.add((expert_id, allo_chiplet))

        for expert_id in expert_dict:
            home = self.org_exp_chiplet_distribution[expert_id][0]
            if expert_id in has_local:
                state['dram_local'][home] += dram_lat_per_expert

        for expert_id, allo_chiplet in remote_pairs:
            home = self.org_exp_chiplet_distribution[expert_id][0]
            state['dram_remote'][home] += dram_lat_per_expert

        for expert_id, req_num, allo_chiplet in exp_allo_plan:
            data_chiplet = self.org_exp_chiplet_distribution[expert_id][0]
            compute_lat = self._compute_lat(req_num)

            if allo_chiplet == data_chiplet:
                state['compute_local'][allo_chiplet] += compute_lat
            else:
                hops = self._manhattan_distance(allo_chiplet, data_chiplet)
                d2d_lat = self._d2d_transfer_lat(hops)
                state['compute_remote'][allo_chiplet] += compute_lat
                state['d2d_in'][allo_chiplet] += d2d_lat
                state['d2d_out'][data_chiplet] += d2d_lat

        return state

    def _rebalance_pass(self, exp_allo_plan, expert_dict, debug):
        """After greedy allocation, try migrating entries off the bottleneck
        die to reduce the global max estimated time.  Uses a wider search
        radius (rebalance_distance) than the greedy pass."""
        for round_idx in range(self.rebalance_max_rounds):
            state = self._rebuild_state_from_plan(exp_allo_plan, expert_dict)
            die_times = self._die_estimated_time(state)
            max_die = int(np.argmax(die_times))
            max_time = float(die_times[max_die])
            active = die_times[die_times > 0]
            avg_time = float(np.mean(active)) if len(active) > 0 else 0.0

            if debug:
                print(f"[Rebalance round {round_idx}] max_die={max_die}, "
                      f"max_time={max_time:.1f}, avg_time={avg_time:.1f}")

            entries_on_max = [(i, exp_allo_plan[i])
                              for i in range(len(exp_allo_plan))
                              if exp_allo_plan[i][2] == max_die]

            best_entry_idx = None
            best_target = None
            best_new_max = max_time

            for entry_idx, (expert_id, req_num, _) in entries_on_max:
                data_chiplet = self.org_exp_chiplet_distribution[expert_id][0]
                remote_map = self._gen_remote_chiplet_list(
                    [data_chiplet], distance=self.rebalance_distance)
                candidates = list(remote_map.keys()) + [data_chiplet]
                candidates = [c for c in candidates if c != max_die]

                saved = exp_allo_plan[entry_idx][2]
                for target_die in candidates:
                    exp_allo_plan[entry_idx][2] = target_die
                    trial_state = self._rebuild_state_from_plan(exp_allo_plan, expert_dict)
                    new_max = float(np.max(self._die_estimated_time(trial_state)))
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_entry_idx = entry_idx
                        best_target = target_die
                exp_allo_plan[entry_idx][2] = saved

            if best_entry_idx is not None:
                old = exp_allo_plan[best_entry_idx]
                if debug:
                    print(f"  Migrate expert {old[0]} ({old[1]} reqs): "
                          f"die {max_die} -> die {best_target}, "
                          f"max_time {max_time:.1f} -> {best_new_max:.1f}")
                exp_allo_plan[best_entry_idx][2] = best_target
            else:
                if debug:
                    print(f"  No improvement found, stopping rebalance.")
                break

        return exp_allo_plan

    def _build_candidate_list_v2(self, local_chiplets, remote_map, state):
        """Return [(estimated_time, chiplet_id), ...] sorted ascending."""
        entries = []
        for cid in local_chiplets:
            t = float(self._die_estimated_time(state, [cid])[0])
            entries.append((t, cid))
        for cid, src_list in remote_map.items():
            data_cid = src_list[0]
            hops = self._manhattan_distance(cid, data_cid)
            d2d_penalty = self._d2d_transfer_lat(hops)
            t = float(self._die_estimated_time(state, [cid])[0])
            entries.append((t + d2d_penalty, cid))
        entries.sort()
        return entries

    def _greedy_block_assign_v2(self, req_num, candidate_ids, local_set, remote_map,
                                num_candidates, state):
        """Assign requests in blocks, greedy-pick lowest-cost candidate using overlap model."""
        REQ_BLOCK_SIZE = 512
        alloc = [0] * num_candidates
        first_alloc = [False] * num_candidates
        remaining = req_num

        while remaining > 0:
            block = min(REQ_BLOCK_SIZE, remaining)
            if remaining - block < REQ_BLOCK_SIZE:
                block = remaining
            best_idx, best_cost = 0, float('inf')

            for idx, cid in enumerate(candidate_ids):
                data_cid = cid if cid in local_set else remote_map[cid][0]
                trial_state = self._cal_allo_cost_v2(
                    state, cid, data_cid, block, allocated_before=first_alloc[idx])
                affected_dies = list({cid, data_cid})
                cost = float(np.max(self._die_estimated_time(trial_state, affected_dies)))
                if cost < best_cost or (cost == best_cost and cid in local_set
                        and candidate_ids[best_idx] not in local_set):
                    best_cost = cost
                    best_idx = idx

            chosen_cid = candidate_ids[best_idx]
            data_cid = chosen_cid if chosen_cid in local_set else remote_map[chosen_cid][0]
            state = self._cal_allo_cost_v2(
                state, chosen_cid, data_cid, block, allocated_before=first_alloc[best_idx])
            first_alloc[best_idx] = True
            alloc[best_idx] += block
            remaining -= block

        return alloc, state

    # ------------------------------------------------------------------ #
    #  Overlap-aware cost model                                           #
    # ------------------------------------------------------------------ #

    def _cal_allo_cost_v2(self, state, allo_chiplet, data_chiplet, batch_size, allocated_before):
        """Update die state after allocating batch_size requests.
        Returns a new state dict (does not mutate input).

        Local assignment (allo == data):
          - allo_die.compute_local += compute_lat   (overlaps with dram_local)

        Remote assignment (allo != data):
          - allo_die.compute_remote += compute_lat  (must wait for d2d_in first)
          - allo_die.d2d_in  += d2d_lat             (receive weights over link)
          - data_die.dram_remote += dram_lat         (read weights to send, first time only)
          - data_die.d2d_out += d2d_lat              (send weights over link)
        """
        compute_lat = self._compute_lat(batch_size)
        dram_lat = self._dram_read_lat()
        hops = self._manhattan_distance(allo_chiplet, data_chiplet) if allo_chiplet != data_chiplet else 0
        d2d_lat = self._d2d_transfer_lat(hops) if hops > 0 else 0

        ns = self._copy_die_state(state)

        if allo_chiplet == data_chiplet:
            ns['compute_local'][allo_chiplet] += compute_lat
        else:
            ns['compute_remote'][allo_chiplet] += compute_lat
            ns['d2d_in'][allo_chiplet] += d2d_lat
            if not allocated_before:
                ns['dram_remote'][data_chiplet] += dram_lat
            ns['d2d_out'][data_chiplet] += d2d_lat

        return ns

    # ------------------------------------------------------------------ #
    #  Summary printing                                                   #
    # ------------------------------------------------------------------ #

    def _print_allocation_summary(self, exp_allo_plan):
        sum_table = {}
        for expert_id in range(self.config.expert_num):
            org_die = self.org_exp_chiplet_distribution[expert_id][0]
            sum_table[expert_id] = [org_die, [], []]
        for expert_id, req_num, chiplet_id in exp_allo_plan:
            sum_table[expert_id][1].append(chiplet_id)
            sum_table[expert_id][2].append(req_num)

        print("summary table for iter %d, layer %d: " % (self.iter_id, self.layer_id))
        unexpected_allo = []
        for expert_id, (org_die, allo_dies, allo_reqs) in sum_table.items():
            print("exp=%s org_die=%d allo_dies: " % (expert_id, org_die), allo_dies, " ", allo_reqs)
            if self.current_stage == "decode":
                if len(allo_dies) > 1 or (len(allo_dies) == 1 and allo_dies[0] != org_die):
                    unexpected_allo.append(
                        "exp=%s org_die=%d allo_dies: %s %s" % (expert_id, org_die, allo_dies, allo_reqs))
        print()
        if self.current_stage == "decode":
            print("unexpected allocation for iter %d, layer %d: " % (self.iter_id, self.layer_id))
            for info in unexpected_allo:
                print(info)
            print()

    def _gen_remote_chiplet_list(self, local_chiplet_list, distance=1):
        remote_chiplet_set = set()
        y_chiplets = self.config.y_chiplets
        x_chiplets = self.config.x_chiplets

        local_chiplet_set = set(local_chiplet_list)  # 提高判断效率
        remote_chiplet_map = {}
        for local_chiplet_idx in local_chiplet_list:
            local_chiplet = self.interconnect.chiplets[local_chiplet_idx]
            x0 = local_chiplet.xidx
            y0 = local_chiplet.yidx

            # 枚举曼哈顿距离小于等于distance的所有点
            for dx in range(-distance, distance + 1):
                for dy in range(-distance, distance + 1):
                    if (dx == 0 and dy == 0) or abs(dx) + abs(dy) > distance:
                        continue  # 距离大于distance或本身
                    x = x0 + dx
                    y = y0 + dy
                    if 0 <= x < x_chiplets and 0 <= y < y_chiplets:
                        chiplet_id = y + x * y_chiplets
                        if chiplet_id not in local_chiplet_set:
                            if chiplet_id not in remote_chiplet_map:
                                remote_chiplet_map[chiplet_id] = []
                                
                            remote_chiplet_map[chiplet_id].append(local_chiplet_idx)
                            remote_chiplet_set.add(chiplet_id)

        # return list(remote_chiplet_set)
        return remote_chiplet_map


    def _gen_chiplet_exp_distribution(self):
        chiplet_exp_distribution = {} # {chiplet_idx: [exp0, exp1, exp2, ...]}
        for chiplet_idx in range(self.config.chiplet_num):
            chiplet = self.interconnect.chiplets[chiplet_idx]
            experts_in_chiplet = set()
            for exp_slice in chiplet.dram.data:
                experts_in_chiplet.add(exp_slice.expert_id)
            chiplet_exp_distribution[chiplet_idx] = list(experts_in_chiplet)
        return chiplet_exp_distribution
    
    def _gen_exp_chiplet_distribution(self, layer_idx):
        exp_chiplet_distribution = {} # {exp_idx: [chiplet0, chiplet1, ...]}    
        for exp_idx in range(self.config.expert_num): # init
            exp_chiplet_distribution[exp_idx] = []
        for chiplet_idx in range(self.config.chiplet_num):
            chiplet = self.interconnect.chiplets[chiplet_idx]
            for exp_slice in chiplet.dram.data:
                if exp_slice.layer_id == layer_idx:
                    exp_idx = exp_slice.expert_id
                    if chiplet_idx not in exp_chiplet_distribution[exp_idx]:
                        exp_chiplet_distribution[exp_idx].append(chiplet_idx)
                
        return exp_chiplet_distribution

    def _schedule_next_request(self):
        # print("req_num: %d" % (self.access_records))
        # print("event queue before:", len(self.event_queue))
        # print("pending requests: ", self.pending_requests)
        # for event in self.event_queue:
        #     print(event.event_type)
        # print("event queue before new: %d" % (len(self.event_queue)))
        # print("in schedule next request")
        # print("iter id is: %d" %(self.iter_id))
        try:
            # record execution time for each req
            self.pre_layer_begin_time = self.cur_layer_begin_time
            self.cur_layer_begin_time = self.current_time
            if self.iter_id not in self.layer_time_records:
                self.layer_time_records[self.iter_id] = {}
            # if self.layer_id not in self.layer_time_records[self.iter_id]:
            #     self.layer_time_records[self.iter_id][self.layer_id] = 0
            self.layer_time_records[self.iter_id][self.layer_id] = self.cur_layer_begin_time - self.pre_layer_begin_time

            if self.operation == "attention":
                # print("attention operation")
                expert_dict, layer, stage, prefill_len_list = next(self.trace_generator)
                self.iter_info = [expert_dict, layer, stage]
                self.current_experts = expert_dict
                self.current_layer = layer  
                              
                # update current stage and last iter stage
                self.last_iter_stage = self.current_stage
                self.current_stage = stage
                # print(self.last_iter_stage, self.current_stage)
                if self.current_stage == "decode" and self.last_iter_stage == "prefill":
                    self.prefill_time = self.current_time
                
                if stage == "prefill":
                    attn_ops = 0
                    tot_len = 0
                    for prefill_len in prefill_len_list:
                        attn_ops += prefill_len * (prefill_len+1) / 2 * self.config.hidden_dim * self.config.hidden_dim
                        tot_len += prefill_len
                    qkv_ops = 3 * self.config.hidden_dim * self.config.hidden_dim * tot_len
                    tot_compute_power = self.config.chiplet_num * self.config.compte_power_per_die
                    layer_attn_time = (qkv_ops + attn_ops) / (tot_compute_power * 0.7)
                
                else:
                    qkv_ops = 3 * self.config.hidden_dim * self.config.hidden_dim * self.tot_batch_size
                    attn_ops = self.config.hidden_dim * self.config.hidden_dim
                    tot_compute_power = self.config.chiplet_num * self.config.compte_power_per_die
                    layer_attn_time = (qkv_ops + attn_ops) / (tot_compute_power * 0.7)
                    # operations = self.config.hidden_dim
                    # print("attn time: ", tot_time)
                layer_attn_time = 0
                    
                event = SimulationEvent(
                    time = self.current_time,
                    batch_size=0,
                    event_type=EventType.ATTENTION,
                    chiplet_id=None,
                    address=None,
                    resources=[],
                    callback=lambda e: self._process_attn_request(e, layer_attn_time),
                    priority=getattr(self, "_current_event_priority", 0.0),
                )
                self._add_event(event)
                self.tot_attn_time += layer_attn_time
                self.operation = "moe"
                
            else:
                expert_dict, layer, stage, _ = next(self.trace_generator)
                self.iter_info = [expert_dict, layer, stage]
                self.current_experts = expert_dict
                self.current_layer = layer  
                              
                # update current stage and last iter stage
                self.last_iter_stage = self.current_stage
                self.current_stage = stage
                if self.current_stage == "decode" and self.last_iter_stage == "prefill":
                    # print("iter id: %d, layer id: %d" % (self.iter_id, self.layer_id))
                    self.prefill_time = self.current_time 
                # expert_dict, layer, stage = self.iter_info # Use this if attention is considered

                # Allocate all requests to different dies 
                _allo_t0 = time.perf_counter()
                if self.config.allocation_strategy == self.config.AllocationStrategy.OURS_ORG:
                    exp_allo_plan = self._allocate_experts_org(expert_dict, layer, stage)
                else:
                    exp_allo_plan = self._allocate_experts(expert_dict, layer, stage)
                self._prof['allocate_experts'] += time.perf_counter() - _allo_t0

                # Group the plan by expert: every die executing an expert it
                # does not host must pull the same slices from the one die that
                # does. That requester set is the multicast group.
                if MEASURE_FANOUT:
                    groups = defaultdict(set)
                    for expert_id, _req, die_id in exp_allo_plan:
                        home = self.expert_chiplet_map.get(int(expert_id))
                        if home is not None and die_id != home:
                            groups[int(expert_id)].add(die_id)
                    slices_per_expert = (1 if self.config.whole_expert_as_one_slice
                                         else self.config.expert_tot_slice_num)
                    for requesters in groups.values():
                        self.fanout_hist[len(requesters)] += slices_per_expert

                if PROGRESS_EVERY_ITERS:
                    # Trigger on the iteration changing, NOT on layer_id == 0:
                    # models with dense layers (deepseek 0-2, llama4) never yield
                    # layer 0, so that condition could never fire for them.
                    every = min(PROGRESS_EVERY_ITERS,
                                max(1, getattr(self, "max_iter_num", 0) // 8))
                    if (self.iter_id != getattr(self, "_last_progress_iter", -1)
                            and self.iter_id % every == 0):
                        self._last_progress_iter = self.iter_id
                        self._print_progress()

                # Open a communication accounting window for this epoch
                # before any event is created.
                self.comm.begin_epoch(self.iter_id, self.layer_id, stage,
                                      self.current_time)

                # Record the logical routing demand and the selected execution
                # chiplet before one event is created for each expert slice.
                for expert_id, req_num in expert_dict.items():
                    self.expert_selection_records[(stage, int(layer), int(expert_id))] += req_num
                for expert_id, req_num, chiplet_id in exp_allo_plan:
                    key = (stage, int(layer), int(expert_id), int(chiplet_id))
                    self.expert_execution_records[key] += req_num
                    
                # for line in exp_allo_plan:
                #     expert_id, req_num, chiplet_id = line
                #     org_chiplet_id = self.expert_chiplet_map[expert_id]
                #     print("exp=%d, req_num=%d, allo_chip_id=%d, org_chip_id=%d" \
                #         % (expert_id, req_num, chiplet_id, org_chiplet_id))
                # print("exp allo plan")
                # print(exp_allo_plan)
                self.exp_allo_plan = exp_allo_plan

                # Multicast: every die executing an expert it does not host pulls
                # the same slices from the one die that does. Deliver each slice
                # to that whole requester set as a single tree; groups of one (and
                # the hosting die's own local read) stay on the unicast path.
                mcast_done = set()
                if MULTICAST_ENABLED:
                    grp = defaultdict(list)
                    for expert_id, req_num, die_id in exp_allo_plan:
                        home = self.expert_chiplet_map.get(int(expert_id))
                        if home is not None and die_id != home:
                            grp[int(expert_id)].append((die_id, req_num))
                    m_range = 1 if self.config.whole_expert_as_one_slice else self.config.expert_matrix_num
                    s_range = 1 if self.config.whole_expert_as_one_slice else self.config.expert_matrix_slice_num
                    for expert_id, members in grp.items():
                        if len(members) < 2:
                            continue            # no sharing to exploit
                        home = self.expert_chiplet_map[expert_id]
                        dies = [d for d, _ in members]
                        nums = [n for _, n in members]
                        ok = True
                        for matrix_id in range(m_range):
                            for slice_id in range(s_range):
                                addr = Addr_expert(expert_id=expert_id, layer_id=int(layer),
                                                   matrix_id=matrix_id, slice_id=slice_id)
                                ok &= self._emit_multicast_delivery(addr, home, dies, nums)
                                # one record per slice, matching the unicast path
                                for d, rn in members:
                                    self._record_allocation(d, rn)
                        if ok:
                            for d in dies:
                                mcast_done.add((expert_id, d))

                # print(self.exp_allo_plan)
                for line in exp_allo_plan:
                    expert_id, req_num, chiplet_id = line
                    if (int(expert_id), chiplet_id) in mcast_done:
                        continue
                    matrix_range = 1 if self.config.whole_expert_as_one_slice else self.config.expert_matrix_num
                    slice_range = 1 if self.config.whole_expert_as_one_slice else self.config.expert_matrix_slice_num
                    for matrix_id in range(matrix_range):
                        for slice_id in range(slice_range):
                            # print("event")
                            addr_expert = Addr_expert(
                                expert_id=int(expert_id),
                                layer_id=int(layer),
                                matrix_id=matrix_id,
                                slice_id=slice_id
                            )
                            _rec_t0 = time.perf_counter()
                            self._record_allocation(chiplet_id, req_num)
                            self._prof['record_allocation'] += time.perf_counter() - _rec_t0
                            
                            
                            # give remote access requests higher priority, so that they can be processed first
                            priority = 0.0
                            org_chiplet_id = self._expert_placement_map(addr_expert)
                            if org_chiplet_id != chiplet_id:
                                priority = 1.0
                            event = SimulationEvent(
                                time=self.current_time,
                                batch_size=req_num,
                                event_type=EventType.CACHE_ACCESS,
                                chiplet_id=chiplet_id,
                                address=addr_expert,
                                resources=[self.res_mgr.resources[(ResourceType.CACHE_PORT, chiplet_id)]],
                                callback=lambda e: self._process_request(e),
                                priority=priority   
                            )

                            self._add_event(event)
                            self.pending_requests += 1
                
                # self.operation = "attention" # Use this if attention is considered
            
        except StopIteration:
            # print("stop iteration")
            pass

    def _process_attn_request(self, event: SimulationEvent, layer_attn_time):
        # layer_attn_time = 1
        attn_latency = layer_attn_time

        # 注册完成回调
        complete_event = SimulationEvent(
            time=self.current_time + attn_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPLETION_SIGNAL,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[],
            callback=lambda e: self._handle_attn_request_completion(
                e,
                start_time = 0
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, complete_event)
        self._add_event(complete_event)
        self.current_time = self.current_time + attn_latency
        
        
        
        return

    def _process_request(self, event: SimulationEvent):
        # self._process_request_new(event)
        # return # FIXME
        self.stats['total_access'] += 1
        if self.config.pred_next_token:
            self._process_request_new(event)
        else:
            # print("cache strategy: ", self.config.cache_strategy)
            if (self.config.cache_strategy == self.config.CacheStrategy.CACHE_LOCAL) or \
               (self.config.cache_strategy == self.config.CacheStrategy.DISABLED):
                # print("in base cache local")
                self._processs_request_base_cache_local(event)
            elif self.config.cache_strategy == self.config.CacheStrategy.CACHE_ALL:
                self._process_request_base_cache_all(event)


    def _process_request_new(self, event: SimulationEvent):
        '''
            our proposed prediction based cache strategy
        '''
        start_time = self.current_time
        allocated_chiplet_id = event.chiplet_id
        allocated_chiplet = self.interconnect.chiplets[event.chiplet_id]
        address_chiplet_id = self._expert_placement_map(event.address)
        address_chiplet = self.interconnect.chiplets[address_chiplet_id]
        
        local_cache_hit = allocated_chiplet.cache.check(event.address)
        local_dram_hit = allocated_chiplet.dram.check(event.address)
        remote_cache_hit = address_chiplet.cache.check(event.address)
        # print("local cache / local DRAM / remote cache hit: ", local_cache_hit, local_dram_hit, remote_cache_hit)
        if local_cache_hit:
            self._handle_read_local_llc(event, start_time)
        else:
            if local_dram_hit:
                # print("local dram hit")
                self._handle_read_local_dram(event, start_time)
            else:
                # print("remote cache hit: ", remote_cache_hit)
                if remote_cache_hit:
                    self._handle_read_remote_llc(event, start_time, address_chiplet)
                else:
                    self._handle_read_remote_dram(event, start_time, address_chiplet)
        return

    def _processs_request_base_cache_local(self, event: SimulationEvent):
        '''
            process flow for cache local strategy, where the llc will only cache data form local DRAM
            first find the associate chiplet using address
            then check the cache and DRAM
        '''

        start_time = self.current_time
        # print("req start time: %d" % (start_time))
        allocated_chiplet_id = event.chiplet_id
        allocated_chiplet = self.interconnect.chiplets[allocated_chiplet_id]
        address_chiplet_id = self._expert_placement_map(event.address)
        address_chiplet = self.interconnect.chiplets[address_chiplet_id]
        is_local_address = allocated_chiplet_id == address_chiplet_id
        local_cache_hit = allocated_chiplet.cache.check(event.address)
        remote_cache_hit = address_chiplet.cache.check(event.address)
        # print("event info: exp_id=%s,\ttime=%s,\tchiplet_id=%s,\taddress_chiplet_id=%s,\tis_local=%s,\tremote_cache_hit=%s" \
        #       % (event.address.expert_id, event.time, event.chiplet_id, address_chiplet_id, is_local_address, remote_cache_hit))
        # print("cache hit info: is_local_address=%s, local_cache_hit=%s" % (is_local_address, local_cache_hit))
        if is_local_address:
            if local_cache_hit:
                # allocated_chiplet.cache.store(event.address)
                self._handle_read_local_llc(event, start_time)
            else:
                # print("is local dram? ", is_local_address)
                self._handle_read_local_dram(event, start_time)
        else:
            if remote_cache_hit:
                self._handle_read_remote_llc(event, start_time, address_chiplet)
            else:
                self._handle_read_remote_dram(event, start_time, address_chiplet)
        return

    def _process_request_base_cache_all(self, event: SimulationEvent):
        '''
            process flow for cache all strategy, where the llc cache data from both local and remote DRAM
            first check LLC from all chiplets
            then access DRAM if LLC miss
        '''
        # Local LLC hit
        start_time = self.current_time
        allocated_chiplet_id = event.chiplet_id
        allocated_chiplet = self.interconnect.chiplets[event.chiplet_id]
        address_chiplet_id = self._expert_placement_map(event.address)
        address_chiplet = self.interconnect.chiplets[address_chiplet_id]
        is_local_address = allocated_chiplet_id == address_chiplet_id
        if allocated_chiplet.cache.check(event.address):
            self._handle_read_local_llc(event, start_time)
        else: # not in local llc, search remote llc
            remote_llc_hit = self._search_remote_llc(event, start_time)
            if remote_llc_hit:
                self._handle_read_remote_llc(event, start_time, address_chiplet)
            else: # not in llc, fetch from dram
                if is_local_address:
                    self._handle_read_local_dram(event, start_time)
                else:
                    self._handle_read_remote_dram(event, start_time, address_chiplet)
        return

    def _handle_read_local_llc(self, event:SimulationEvent, start_time):
        # print("handle read local llc")
        # 命中处理
        cache_accesss_latency = self.timing_config.cache_hit_latency
        src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        self.stats['local_cache_hit'] += 1
        
        # 注册完成回调
        complete_event = SimulationEvent(
            time=self.current_time + cache_accesss_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPUTATION,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[self.res_mgr.resources[(ResourceType.COMPUTE_UNIT, src_chiplet.id)]],
            callback=lambda e: self._handle_computation(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, complete_event)
        self._add_event(complete_event)
        self.current_time = self.current_time + cache_accesss_latency

    def _search_remote_llc(self, event: SimulationEvent, start_time):

        # search remote llc
        for remote_chiplet in self.interconnect.chiplets:
            if remote_chiplet.id == event.chiplet_id: continue
            
            if remote_chiplet.cache.check(event.address):
                return True
        return False

    def _handle_read_remote_llc(self, event: SimulationEvent, start_time, remote_chiplet: Chiplet):
        src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        
        # 本地未命中惩罚
        base_latency = self.timing_config.cache_miss_penalty
        if remote_chiplet.cache.check(event.address):
            # print("find in remote LLC")
            
            path = self.interconnect.route(src_chiplet, remote_chiplet)
            path_link_resources = self._parse_link_resources(path)
            # send req to remote LLC
            issue_ns = self.current_time + base_latency
            transfer_event = SimulationEvent(
                time=issue_ns,
                batch_size=event.batch_size,
                event_type=EventType.NET_TRANS_REQ,
                chiplet_id=event.chiplet_id,
                address=event.address,
                resources=path_link_resources, 
                callback=lambda e: self._send_req_to_remote_llc(
                    e,
                    start_time,
                    src_chiplet,
                    remote_chiplet,
                    issue_ns
                ),
                priority=getattr(self, "_current_event_priority", 0.0),
            )
            # heapq.heappush(self.event_queue, transfer_event)
            self._add_event(transfer_event)
            self.current_time = self.current_time + base_latency
        else:
            raise ValueError("fake remote LLC hit")
        
        return

    def _handle_read_local_dram(self, event: SimulationEvent, start_time):
        '''
            try to find in local dram.
            If find in local dram, next step is to fetch data from DRAM
        '''
        self.stats['local_dram_hit'] += 1
        # print("handle read local dram")
        src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        # 本地未命中惩罚
        base_latency = self.timing_config.cache_miss_penalty
        
        found = False
        if src_chiplet.dram.check(event.address):
            mem_event = SimulationEvent(
                time=self.current_time + base_latency,
                batch_size=event.batch_size,
                event_type=EventType.MEMORY_ACCESS,
                chiplet_id=event.chiplet_id,
                address=event.address,
                resources=[self.res_mgr.resources[(ResourceType.DRAM_PACKAGE, src_chiplet.id)]],
                callback=lambda e: self._read_data_from_local_dram(
                    e,
                    start_time
                ),
                priority=getattr(self, "_current_event_priority", 0.0),
            )
            # heapq.heappush(self.event_queue, mem_event)
            # print("ready to add event")
            self._add_event(mem_event)
            self.current_time = self.current_time + base_latency
            found = True
        else:
            raise ValueError("fake local DRAM hit")
        return found

    def _handle_read_remote_dram(self, event: SimulationEvent, start_time, remote_chiplet: Chiplet):
        '''
            find in remote dram and next step is to transfer command through d2d link
        '''

        src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        # 本地未命中惩罚
        base_latency = self.timing_config.cache_miss_penalty
        
        if remote_chiplet.dram.check(event.address):
            # print("in remote DRAM, exp_id=%s" %(event.address.expert_id))

            path = self.interconnect.route(src_chiplet, remote_chiplet)
            path_link_resources = self._parse_link_resources(path)
            # print("path: ", path)
            # print("path link resources: ", len(path_link_resources))
            # create next event to transfer through d2d link
            issue_ns = self.current_time + base_latency
            d2d_trans_event = SimulationEvent(
                time=issue_ns,
                batch_size=event.batch_size,
                event_type=EventType.NET_TRANS_REQ,
                chiplet_id=event.chiplet_id,
                address=event.address,
                resources=path_link_resources,
                callback=lambda e: self._send_req_to_remote_dram(
                    e,
                    start_time, 
                    src_chiplet,
                    remote_chiplet,
                    issue_ns
                ),
                priority=getattr(self, "_current_event_priority", 0.0),
            )
            # print("read remote dram priority: ", getattr(self, "_current_event_priority", 0.0))
            # heapq.heappush(self.event_queue, mem_event)
            self._add_event(d2d_trans_event)
            self.current_time = self.current_time + base_latency
        else:
            raise ValueError("fake remote DRAM hit")
        return 

    def _record_comm(self, kind, path, data_size, service_ns, issue_ns):
        """Hand one D2D message to the recorder.

        Called from the four sites that actually put a message on the links, at
        the moment it starts moving -- so self.current_time is the post-contention
        start and (current_time - issue_ns) is how long it waited on a busy link.
        """
        if not COMM_TRACE_ENABLED:
            return
        start_ns = self.current_time
        self.comm.record(kind, path, data_size, service_ns, start_ns,
                         0.0 if issue_ns is None else (start_ns - issue_ns))

    def export_comm_analysis(self, output_dir, strategy_name, batch_size):
        """Write the D2D CSVs. Call after process_records()."""
        shape = f"{self.config.y_chiplets}x{self.config.x_chiplets}"
        stem = f"{self.config.model_name}_{strategy_name}_{shape}_batch{batch_size}"
        return self.comm.export(output_dir, stem, sim=self)

    def _emit_multicast_delivery(self, addr, supplier_id, requesters, req_nums):
        """One DRAM read at the supplier, one tree transfer, N computations.

        Replaces N independent unicast chains for the same slice. pending_requests
        is still incremented once per requester, so the layer barrier is unchanged
        -- only the delivery is shared.
        """
        supplier = self.interconnect.chiplets[supplier_id]
        dsts = [self.interconnect.chiplets[d] for d in requesters]
        links, depth, first_hops, _nodes = self.interconnect.multicast_tree(supplier, dsts)
        if not links:
            return False
        tree_res = self._parse_tree_link_resources(links)

        start_time = self.current_time
        event = SimulationEvent(
            time=self.current_time,
            batch_size=sum(req_nums),
            event_type=EventType.MEMORY_ACCESS,
            chiplet_id=supplier_id,
            address=addr,
            resources=[self.res_mgr.resources[(ResourceType.DRAM_PACKAGE, supplier_id)]],
            callback=lambda e: self._mcast_dram_read(
                e, start_time, supplier, requesters, req_nums,
                tree_res, links, depth, first_hops),
            priority=1.0,
        )
        self._add_event(event)
        self.pending_requests += len(requesters)
        return True

    def _mcast_dram_read(self, event, start_time, supplier, requesters, req_nums,
                         tree_res, links, depth, first_hops):
        """Supplier reads the slice from DRAM once, for all requesters."""
        data_size = self.config.expert_slice_size
        mem_latency = self._cal_mem_latency(data_size, event.batch_size, is_remote=True)
        self.remote_dram_read_size += data_size      # one read, not len(requesters)
        self._record_dram_access(supplier.id, data_size)

        issue_ns = self.current_time + mem_latency
        deliver = SimulationEvent(
            time=issue_ns,
            batch_size=event.batch_size,
            event_type=EventType.NET_TRANS_DATA,
            chiplet_id=supplier.id,
            address=event.address,
            resources=tree_res,
            callback=lambda e: self._mcast_deliver(
                e, start_time, supplier, requesters, req_nums,
                links, depth, first_hops, issue_ns),
            priority=1.0,
        )
        self._add_event(deliver)
        self.current_time = self.current_time + mem_latency

    def _mcast_deliver(self, event, start_time, supplier, requesters, req_nums,
                       links, depth, first_hops, issue_ns):
        """Tree transfer completes; every requester starts its computation."""
        data_size = self.config.expert_slice_size
        latency = self._cal_tree_latency(depth, data_size)
        self.stats['hop'] += len(links)
        if COMM_TRACE_ENABLED:
            self.comm.record_tree(supplier.id, requesters, links, first_hops,
                                  data_size, latency, self.current_time,
                                  self.current_time - issue_ns)
        for die_id, batch in zip(requesters, req_nums):
            comp = SimulationEvent(
                time=self.current_time + latency,
                batch_size=batch,
                event_type=EventType.COMPUTATION,
                chiplet_id=die_id,
                address=event.address,
                resources=[self.res_mgr.resources[(ResourceType.COMPUTE_UNIT, die_id)]],
                callback=lambda e: self._handle_computation(e, start_time),
                priority=1.0,
            )
            self._add_event(comp)
        self.current_time = self.current_time + latency

    def _record_dram_access(self, chiplet_id, data_size):
        if self.iter_id not in self.dram_access_records:
            self.dram_access_records[self.iter_id] = {}
        if self.layer_id not in self.dram_access_records[self.iter_id]:
            self.dram_access_records[self.iter_id][self.layer_id] = np.zeros(self.config.chiplet_num)
        self.dram_access_records[self.iter_id][self.layer_id][chiplet_id] += data_size
        return
        # pass
        

    def _read_data_from_local_dram(self, event: SimulationEvent, start_time):
        '''
            begin read data from local dram, next step is to complete request
        '''
        # mem_latency = self.timing_config.mem_access_latency
        weight_size = self.config.expert_slice_size
        activation_size = event.batch_size * (self.config.hidden_dim + self.config.expert_dim) * self.config.expert_matrix_num * self.config.data_type / self.config.expert_tot_slice_num
        data_size = weight_size + activation_size
        # print("weight size: ", weight_size, "activation size: ", activation_size, "data size: ", data_size)
        mem_latency = self._cal_mem_latency(data_size, event.batch_size)
        # print("local dram latency: ", mem_latency)
        
        src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        src_chiplet.dram.access(event.address)
        
        self.local_dram_read_size += data_size
        self._record_dram_access(event.chiplet_id, data_size)
        
        
        # duplicate in local LLC if Cache all
        if (self.config.cache_strategy == self.config.CacheStrategy.CACHE_ALL) or \
           (self.config.cache_strategy == self.config.CacheStrategy.CACHE_LOCAL):
            # print("cache in local llc")
            # create new event to store in local cache
            local_dram_event = SimulationEvent(
                time=self.current_time + mem_latency,
                batch_size=event.batch_size,
                event_type=EventType.CACHE_STORE,
                chiplet_id=event.chiplet_id,
                address=event.address,
                resources=[self.res_mgr.resources[(ResourceType.CACHE_PORT, event.chiplet_id)]],
                callback=lambda e: self._store_in_lcoal_llc(
                    e,
                    start_time
                ),
                priority=getattr(self, "_current_event_priority", 0.0),
            )
            # if event.address.expert_id == 64:       
            #     print("cache store event time before: %d" %(network_event.time))
            #     for event in self.res_mgr.resources[(ResourceType.CACHE_PORT, event.chiplet_id)].queue:
            #         print("\twait queue: expert=%d, type=%s, time=%d" \
            #             %(event.address.expert_id, event.event_type, event.time))
                # print(self.res_mgr.resources[(ResourceType.CACHE_PORT, event.chiplet_id)].queue)
            self._add_event(local_dram_event)
            # if event.address.expert_id == 64:    
            #     print("cache store event time after: %d" %(network_event.time))
            self.pending_requests += 1
            if PRINT_DEBUG:
                print("\t\t\tadd req: expert_id=%s" % (event.address.expert_id))

        # # cache data
        # chiplet = self.interconnect.chiplets[event.chiplet_id]
        # chiplet.cache.store(event.address)

        event = SimulationEvent(
            time=self.current_time + mem_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPUTATION,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[self.res_mgr.resources[(ResourceType.COMPUTE_UNIT, src_chiplet.id)]],
            callback=lambda e: self._handle_dram_read_computation(
                e,
                mem_latency,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        self._add_event(event)
        self.current_time = self.current_time + mem_latency
        # print("mem read latency: %d" %(mem_latency))
        return

    def _send_req_to_remote_llc(self, event: SimulationEvent, start_time, src_chiplet: Chiplet, remote_chiplet: Chiplet, issue_ns=None):
        '''
            begin transfer request through D2D link, next step is to read remote LLC
        '''
        path = self.interconnect.route(src_chiplet, remote_chiplet)
        network_latency = self._cal_network_latency(path, data_size=16)
        # print("network latency: ", network_latency)
        # path_list = self._parse_path(path)
        # print("path list: ", path_list)
        self.stats['hop'] += max(len(path) -1, 0)
        self._record_comm(KIND_REQ, path, 16, network_latency, issue_ns)

        # new event: read data from remote LLC
        mem_event = SimulationEvent(
            time=self.current_time + network_latency,
            batch_size=event.batch_size,
            event_type=EventType.CACHE_ACCESS,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[self.res_mgr.resources[(ResourceType.CACHE_PORT, remote_chiplet.id)]],
            callback=lambda e: self._remote_llc_read(
                e,
                start_time,
                src_chiplet,
                remote_chiplet
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(mem_event)
        self.current_time = self.current_time + network_latency     

    def _send_req_to_remote_dram(self, event: SimulationEvent, start_time, src_chiplet: Chiplet, remote_chiplet: Chiplet, issue_ns=None):
        '''
            begin transfer request through D2D link, next step is to read remote DRAM
        '''
        path = self.interconnect.route(src_chiplet, remote_chiplet)
        network_latency = self._cal_network_latency(path, data_size=16)
        # print("network latency: ", network_latency)
        # path_list = self._parse_path(path)
        # print("path list: ", path_list)
        self.stats['hop'] += max(len(path) -1, 0)
        self._record_comm(KIND_REQ, path, 16, network_latency, issue_ns)

        # new event: read data from remote DRAM
        mem_event = SimulationEvent(
            time=self.current_time + network_latency,
            batch_size=event.batch_size,
            event_type=EventType.MEMORY_ACCESS,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[self.res_mgr.resources[(ResourceType.DRAM_PACKAGE, remote_chiplet.id)]],
            callback=lambda e: self._remote_dram_read(
                e,
                start_time,
                src_chiplet,
                remote_chiplet
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(mem_event)
        self.current_time = self.current_time + network_latency        

    def _remote_llc_read(self, event: SimulationEvent, start_time, src_chiplet: Chiplet, remote_chiplet: Chiplet):
        '''
            begin to read from remote LLC, next step is to transfer data back to src
        '''
        
        llc_read_latency = self.timing_config.cache_hit_latency
        
        path = self.interconnect.route(remote_chiplet, src_chiplet)
        path_link_resources = self._parse_link_resources(path)
        
        # create new event to transfer data from remote to src
        issue_ns = self.current_time + llc_read_latency
        d2d_trans_event = SimulationEvent(
            time=issue_ns,
            batch_size=event.batch_size,
            event_type=EventType.NET_TRANS_DATA,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=path_link_resources,
            callback=lambda e: self._trans_remote_llc_data_back(
                e,
                start_time, 
                src_chiplet,
                remote_chiplet,
                issue_ns
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(d2d_trans_event)
        self.current_time = self.current_time + llc_read_latency

    def _remote_dram_read(self, event: SimulationEvent, start_time, src_chiplet: Chiplet, remote_chiplet: Chiplet):
        '''
            begin to read from remote DRAM, next step is to transfer data back to src
        '''
        # mem_latency = self.timing_config.mem_access_latency
        data_size = self.config.expert_slice_size
        mem_latency = self._cal_mem_latency(data_size, event.batch_size, is_remote=True)
        # mem_latency = 10
        self.remote_dram_read_size += data_size
        
        # update the LRU table
        # remote_chiplet.dram.access(event.address)
        self._record_dram_access(remote_chiplet.id, data_size)

        # # cache remote dram data in remote LLC if CACHE LOCAL
        if self.config.cache_strategy == self.config.CacheStrategy.CACHE_LOCAL:
            # print("cache data in remote cache")
            # create new event to store in remote cache
            network_event = SimulationEvent(
                time=self.current_time + mem_latency,
                batch_size=event.batch_size,
                event_type=EventType.CACHE_STORE,
                chiplet_id=event.chiplet_id,
                address=event.address,
                resources=[self.res_mgr.resources[(ResourceType.CACHE_PORT, remote_chiplet.id)]],
                callback=lambda e: self._store_in_remote_llc(
                    e,
                    start_time,
                    remote_chiplet
                ),
                priority=getattr(self, "_current_event_priority", 0.0),
            )       
            self._add_event(network_event)
            self.pending_requests += 1 # store in remote cache while trans back to local chiplet
            if PRINT_DEBUG:
                print("\t\t\tadd req: expert_id=%s" % (event.address.expert_id))
            # print("add one req, expert_id=%s" % (event.address.expert_id))
        path = self.interconnect.route(remote_chiplet, src_chiplet)
        path_link_resources = self._parse_link_resources(path)
        # create new event to transfer data from remote to src
        # print("create d2d trans event: expert_id=%s" %(event.address.expert_id))
        # print(path_link_resources)
        issue_ns = self.current_time + mem_latency
        d2d_trans_event = SimulationEvent(
            time=issue_ns,
            batch_size=event.batch_size,
            event_type=EventType.NET_TRANS_DATA,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=path_link_resources,
            callback=lambda e: self._trans_remote_dram_data_back(
                e,
                start_time, 
                src_chiplet,
                remote_chiplet,
                issue_ns
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        self._add_event(d2d_trans_event)
        self.current_time = self.current_time + mem_latency
        # print("resources d2d: ", path_link_resources) 
        return

    def _trans_remote_llc_data_back(self, event: SimulationEvent, start_time, src_chiplet: Chiplet, remote_chiplet: Chiplet, issue_ns=None):
        '''
            begin to transfer data back to src, next step is to finialize the request 
        '''
        path = self.interconnect.route(remote_chiplet, src_chiplet)
        data_size = self.config.expert_slice_size
        network_latency = self._cal_network_latency(path, data_size)
        self._record_comm(KIND_DATA_LLC, path, data_size, network_latency, issue_ns)
        # print("data size: ", data_size, network_latency)
        # print("trans remote llc data back")

        # duplicate in local dram if pred_next_token
        if self.config.pred_next_token:
            if self._pred_next_token(event): # need to store in local dram
                # print("cache in local dram")
                # create new event to store in local cache
                mem_event = SimulationEvent(
                    time=self.current_time + network_latency,
                    batch_size=event.batch_size,
                    event_type=EventType.DRAM_STORE,
                    chiplet_id=event.chiplet_id,
                    address=event.address,
                    resources=[self.res_mgr.resources[(ResourceType.DRAM_PACKAGE, src_chiplet.id)]],
                    callback=lambda e: self._store_in_lcoal_dram(
                        e,
                        start_time
                    ),
                    priority=getattr(self, "_current_event_priority", 0.0),
                )       
                self._add_event(mem_event)
                self.pending_requests += 1
                if PRINT_DEBUG:
                    print("\t\t\tadd req: expert_id=%s" % (event.address.expert_id))

        # finaliza the request if no further action
        compute_event = SimulationEvent(
            time=self.current_time + network_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPUTATION,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[self.res_mgr.resources[(ResourceType.COMPUTE_UNIT, src_chiplet.id)]],
            callback=lambda e: self._handle_computation(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(compute_event)
        self.current_time = self.current_time + network_latency

        return




    def _trans_remote_dram_data_back(self, event: SimulationEvent, start_time, src_chiplet: Chiplet, remote_chiplet: Chiplet, issue_ns=None):
        '''
            begin to transfer data back to src, next step is to finialize the request 
        '''
        # print("begin to trans remote dram data back")
        path = self.interconnect.route(remote_chiplet, src_chiplet)
        data_size = self.config.expert_slice_size
        network_latency = self._cal_network_latency(path, data_size)
        self._record_comm(KIND_DATA, path, data_size, network_latency, issue_ns)
        # print("data network latency: ", network_latency)
        # duplicate in local dram if pred_next_token
        if self.config.pred_next_token:
            if self._pred_next_token(event): # need to store in local dram
                # src_chiplet = self.interconnect.chiplets[event.chiplet_id]
                # src_chiplet.dram.store(event.address, self.config.expert_slice_size, is_hm_data=True)
                # print("duplicate in local dram")
                # create new event to store in local dram
                mem_event = SimulationEvent(
                    time=self.current_time + network_latency,
                    batch_size=event.batch_size,
                    event_type=EventType.DRAM_STORE,
                    chiplet_id=event.chiplet_id,
                    address=event.address,
                    resources=[self.res_mgr.resources[(ResourceType.DRAM_PACKAGE, src_chiplet.id)]],
                    callback=lambda e: self._store_in_lcoal_dram(
                        e,
                        start_time
                    ),
                    priority=getattr(self, "_current_event_priority", 0.0),
                )       
                self._add_event(mem_event)
                # self.current_time = self.current_time + network_latency
                # return
                self.pending_requests += 1
                if PRINT_DEBUG:
                    print("\t\t\tadd req: expert_id=%s" % (event.address.expert_id))
                # print("wrong event")

        # duplicate in local LLC if Cache all
        if self.config.cache_strategy == self.config.CacheStrategy.CACHE_ALL:

            # create new event to store in local cache
            cache_event = SimulationEvent(
                time=self.current_time + network_latency,
                batch_size=event.batch_size,
                event_type=EventType.CACHE_STORE,
                chiplet_id=event.chiplet_id,
                address=event.address,
                resources=[self.res_mgr.resources[(ResourceType.CACHE_PORT, src_chiplet.id)]],
                callback=lambda e: self._store_in_lcoal_llc(
                    e,
                    start_time
                ),
                priority=getattr(self, "_current_event_priority", 0.0),
            )       
            self._add_event(cache_event)
            # self.current_time = self.current_time + network_latency
            # return
            self.pending_requests += 1
            if PRINT_DEBUG:
                print("\t\t\tadd req: expert_id=%s" % (event.address.expert_id))
            # print("wrong event")

        # create new event to finalize the access
        # print("create new computation event: expert_id=%s" %(event.address.expert_id))
        network_event = SimulationEvent(
            time=self.current_time + network_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPUTATION,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[self.res_mgr.resources[(ResourceType.COMPUTE_UNIT, src_chiplet.id)]],
            callback=lambda e: self._handle_computation(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(network_event)
        # print("event_queue_after_new_event", len(self.event_queue))
        self.current_time = self.current_time + network_latency
        
        return


    def _store_in_lcoal_dram(self, event: SimulationEvent, start_time):
        src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        src_chiplet.dram.store(event.address, self.config.expert_slice_size, is_hm_data=True)
        # print("---------------------save remote data in local dram")
        data_size = self.config.expert_slice_size
        dram_store_latency = self._cal_mem_latency(data_size, event.batch_size)
        # dram_store_latency = 0 # FIXME
        
        self.local_dram_write_size += data_size
        self._record_dram_access(event.chiplet_id, data_size)
        
        completion_event = SimulationEvent(
            time=self.current_time + dram_store_latency,
            batch_size=event.batch_size,
            event_type=EventType.DRAM_STORE_COMPLETION_SIGNAL,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[],
            callback=lambda e: self._handle_request_completion(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # print("new DRAM store event: ", completion_event.event_type)
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(completion_event)
        # print("event queue after add: ------------------")
        # for event0 in self.event_queue:
        #     print("\texpert:%d, event type:%s, time:%d" % (event0.address.expert_id, event0.event_type, event0.time))
        #     for resource in event0.resources:
        #         print("\t\t resource: ", resource.type, resource.id)
        # # print("---------------------------------------------", event.event_type, event.time, self.current_time)
        self.current_time = self.current_time + dram_store_latency

    def _store_in_lcoal_llc(self, event: SimulationEvent, start_time):
        src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        src_chiplet.cache.store(event.address)
        llc_store_latency = self.timing_config.cache_store_latency
        event = SimulationEvent(
            time=self.current_time + llc_store_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPLETION_SIGNAL,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[],
            callback=lambda e: self._handle_request_completion(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(event)
        self.current_time = self.current_time + llc_store_latency

    def _store_in_remote_llc(self, event: SimulationEvent, start_time, remote_chiplet: Chiplet):
        # src_chiplet = self.interconnect.chiplets[event.chiplet_id]
        remote_chiplet.cache.store(event.address)
        llc_store_latency = self.timing_config.cache_store_latency
        event = SimulationEvent(
            time=self.current_time + llc_store_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPLETION_SIGNAL,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[],
            callback=lambda e: self._handle_request_completion(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # heapq.heappush(self.event_queue, mem_event)
        self._add_event(event)
        self.current_time = self.current_time + llc_store_latency

    def _handle_computation(self, event: SimulationEvent, start_time):
        
        # calculate compute latency
        operations = 2 * self.config.expert_matrix_num * self.config.hidden_dim * self.config.expert_dim * event.batch_size / self.config.expert_tot_slice_num
        # compute_latency = math.ceil(operations / (self.config.compte_power_per_die * 1000)) # ns
        compute_latency = self._cal_compute_latency(operations, event.batch_size)
        # print("comput_latency: %d" % (compute_latency), "batch_size: ", event.batch_size)
        # print("cccccc, create new finish req")
        # create 
        event = SimulationEvent(
            time=self.current_time + compute_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPLETION_SIGNAL,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[],
            callback=lambda e: self._handle_request_completion(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        # print("ccccccc, queue before add: ", len(self.event_queue))
        self._add_event(event)
        # print("ccccccc, queue after add: ", len(self.event_queue))
        self.current_time = self.current_time + compute_latency

        # self._record_latency(event.chiplet_id, start_time)
        return

    def _handle_dram_read_computation(self, event: SimulationEvent, data_load_time, start_time):
        '''
            handle computation for dram read
        '''
        # calculate compute latency
        operations = 2 * self.config.expert_matrix_num * self.config.hidden_dim * self.config.expert_dim * event.batch_size / self.config.expert_tot_slice_num
        # compute_latency = math.ceil(operations / (self.config.compte_power_per_die * 1000)) # ns
        compute_latency = self._cal_compute_latency(operations, event.batch_size)
        # print("operations: ", operations)
        
        # print("data_load_time: ", data_load_time)
        # print("compute_latency: ", compute_latency)
        
        # consider the overlap of dram read and computation
        # if data_load_time >= compute_latency:
        #     compute_latency = 10
        # else:
        #     compute_latency = compute_latency - data_load_time

        # create 
        event = SimulationEvent(
            time=self.current_time + compute_latency,
            batch_size=event.batch_size,
            event_type=EventType.COMPLETION_SIGNAL,
            chiplet_id=event.chiplet_id,
            address=event.address,
            resources=[],
            callback=lambda e: self._handle_request_completion(
                e,
                start_time
            ),
            priority=getattr(self, "_current_event_priority", 0.0),
        )
        self._add_event(event)
        self.current_time = self.current_time + compute_latency

        return
 
    def _resource_wait_queue_empty(self):
        for resource in self.res_mgr.resources.values():
            if len(resource.queue) != 0:
                return False
        return True

    def _handle_attn_request_completion(self, event: SimulationEvent, start_time):
        # self._record_latency(event.chiplet_id, start_time)
        # self.pending_requests -= 1
        # print("-----------pending requests", self.pending_requests)
        # print("\t\tcompletion: expert_id=%s, callback=%s" % (event.address.expert_id,event.callback))
        # print("event: ", event.event_type)
        # print("pend=%s, event_queue=%s, wait queue=%s" % (self.pending_requests, len(self.event_queue), self._resource_wait_queue_empty()))
        if (self.pending_requests == 0) and \
            (len(self.event_queue) == 0) and \
            (self._resource_wait_queue_empty()):
            # 前序请求全部完成，调度下一个
            # print("schedule_next")
            self._schedule_next_request()

    def _handle_request_completion(self, event: SimulationEvent, start_time):
        self._record_latency(event.chiplet_id, start_time)
        self.pending_requests -= 1
        # print("-----------pending requests", self.pending_requests)
        # print("\t\tcompletion: expert_id=%s, callback=%s" % (event.address.expert_id,event.callback))
        # print("event: ", event.event_type)
        # print("pend=%s, event_queue=%s, wait queue=%s" % (self.pending_requests, len(self.event_queue), self._resource_wait_queue_empty()))
        if (self.pending_requests == 0) and \
            (len(self.event_queue) == 0) and \
            (self._resource_wait_queue_empty()):
            # 前序请求全部完成，调度下一个
            # print("schedule_next")
            self._schedule_next_request()

    # def _finalize_access(self, start_time, added_latency):
    #     total_latency = self.current_time - start_time + added_latency
    #     self._record_latency(start_time, total_latency)
    #     complete_event = SimulationEvent(
    #         time=self.current_time + added_latency,
    #         batch_size=event.batch_size,
    #         event_type=EventType.COMPLETION_SIGNAL,
    #         chiplet_id=-1,
    #         address=0,
    #         resources=[],
    #         callback=lambda e: self._handle_request_completion(e)
    #     )
    #     # heapq.heappush(self.event_queue, complete_event)
    #     self._add_event(complete_event)

    def _record_latency(self, chiplet_id, start_time):
        # total_latency = self.current_time - start_time + latency
        # self.stats['total_cycles'] = max(self.stats['total_cycles'], self.current_time + latency)
        # self.stats['max_latency'] = max(self.stats['max_latency'], total_latency)
        # self.stats['latency_histogram'][total_latency // 10] += 1
        if self.iter_id not in self.access_records:
            self.access_records[self.iter_id] = {}
        if self.layer_id not in self.access_records[self.iter_id]:
            self.access_records[self.iter_id][self.layer_id] = {}
            # for chiplet_id in range(self.config.chiplet_num):
            #     self.access_records[self.iter_id][self.layer_id][chiplet_id] = []
        # Only the earliest start and latest end are ever read (process_records
        # takes min/max over this), so keep a running span instead of one tuple
        # per completed chain. At 24 dies x 256 experts x 3 slices x 58 layers
        # x 128 iterations that was ~137M tuples (>15 GB) for deepseek, which
        # exceeded the memory cap before the run could finish.
        layer_rec = self.access_records[self.iter_id][self.layer_id]
        span = layer_rec.get(chiplet_id)
        if span is None:
            layer_rec[chiplet_id] = [start_time, self.current_time]
        else:
            if start_time < span[0]:
                span[0] = start_time
            if self.current_time > span[1]:
                span[1] = self.current_time
        return
                

    def _pred_next_token(self, event: SimulationEvent):
        # print("hhihi: ", self.current_stage)
        # return True
        if self.current_stage == "prefill":
            return False
        # return True
        
        # summarize all activated experts in this chiplet 
        local_chiplet_current_experts = []
        for line in self.exp_allo_plan:
            expert_id, req_num, chiplet_id = line
            if chiplet_id == event.chiplet_id:
                if chiplet_id not in local_chiplet_current_experts:
                    local_chiplet_current_experts.append(expert_id)

        # top_frequent_pairs = [key for key, _ in list(self.token_pred_table[self.current_layer].items())[:3000]]
        # for expert_id in local_chiplet_current_experts:
        #     expert_pair = (expert_id, event.address.expert_id)
        #     if expert_pair in top_frequent_pairs:
        #         return True
        
        # see if the current expert will be used later
        for expert_id in local_chiplet_current_experts:
            # print(expert_id, type(expert_id), self.current_layer, type(self.current_layer))
            layer_id = int(self.current_layer)
            freq_for_one_exp = self.token_pred_heatmap[layer_id][expert_id]
            top_n = int(np.ceil(0.2 * self.config.expert_num))
            # print(top_n, type(top_n))
            top_n_indices = np.argsort(freq_for_one_exp)[-top_n:][::-1]
            # print(freq_for_one_exp)
            # print(freq_for_one_exp[top_n_indices])
            if event.address.expert_id in top_n_indices:
                return True
        #     high_prob_experts.extend(top_n_indices)
        # if event.address.expert_id in high_prob_experts:
        #     return True
        
        # print("hhi")
        return False
        # local_chipelt_experts = []

    def _pred_next_layer(self, event: SimulationEvent):
        return False
