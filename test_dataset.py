import sys
from collections import defaultdict, deque
from enum import Enum
import argparse
from typing import List, Dict, Set, Tuple
import heapq
from dataclasses import dataclass, field
import json
import random
import math
import numpy as np
import os

from toy_chiplet_sim import SystemConfig, TimingConfig, Simulator
from itertools import zip_longest



def main():


    dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/abstract_algebra"
    trace_path_list = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    trace_path_list = trace_path_list[:6]
    print("File paths:", trace_path_list)

    # all_data = []
    # for file_path in trace_path_list:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #         all_data.append(data)

    # config = SystemConfig()
    config = SystemConfig(
        y_chiplets=3,
        x_chiplets=3,

        allocation_strategy=SystemConfig.AllocationStrategy.NEAREST,
        # allocation_strategy=SystemConfig.AllocationStrategy.EVEN,
        # allocation_strategy=SystemConfig.AllocationStrategy.RANDOM,

        # exe_strategy=SystemConfig.ExeStrategy.BASELINE_CACHE_LOCAL,
        # exe_strategy=SystemConfig.ExeStrategy.BASELINE_CACHE_ALL,
        exe_strategy=SystemConfig.ExeStrategy.PRED_NEXT_TOKEN,
    )
    sim = Simulator(config)
    trace_file = "/workspace/zhongkai/profiling_result_bp/mmlu/abstract_algebra/0.json"
    # sim.process_trace(trace_file)
    sim.process_trace(trace_path_list)

    print("=== Simulation Results ===")
    # print(f"Total Time: {sim.current_time} ns = {sim.current_time / 1e6} ms")
    print(f"Total Time: {sim.current_time / 1e6:.2f} ms")
    # print(f"Total Accesses: {sim.stats['total_access']}")
    # print(f"Local Cache Hit Rate: {sim.stats['local_cache_hit']/sim.stats['total_access']:.2%}")
    print(f"Local DRAM Hit Rate: {sim.stats['local_dram_hit']/sim.stats['total_access']:.2%}")
    hop_time = sim.stats['hop'] * (sim.timing_config.link_latency + sim.config.expert_slice_size / sim.config.d2d_link_bw) / 1e6
    print(f"Total Hops: {sim.stats['hop']}")
    # print(f"Hop time consumption: {hop_time} ms")
    compute_time = sim.stats['total_access'] * sim.timing_config.compute_latency / 1e6
    # print(f"Total Compute time: {compute_time} ms")

    sim.process_records()
    print(f"chiplet usage: {sim.avg_chiplet_usage:.2f}")

    # print(f"DRAM size of each chiplet: ")
    # for chiplet in sim.interconnect.chiplets:
    #     dram_size = len(chiplet.dram.data) * sim.config.expert_slice_size
    #     dram_size_GB = dram_size / (1024 * 1024 * 1024)
    #     print(f"chiplet: {chiplet.id}, dram size: {dram_size_GB:.0f} GB")
    # # print(sim.access_records)

    print(f"req num of each chiplet: ")
    for chiplet in sim.interconnect.chiplets:
        req_num = chiplet.req_num
        print(f"chiplet: {chiplet.id}, req_num: {req_num:.0f}")
    # print(sim.access_records)

def gen_trace_data_list():
    
    full_trace_path_list = []
    
    dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "/workspace/zhongkai/profiling_result/mmlu/%s" % (subject_list[0])
    sub_trace_path_list1 = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list1.sort()
    
    dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "/workspace/zhongkai/profiling_result/mmlu/%s" % (subject_list[1])
    sub_trace_path_list2= [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list2.sort()
    
    dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "/workspace/zhongkai/profiling_result/mmlu/%s" % (subject_list[2])
    sub_trace_path_list3 = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list3.sort()
    
    dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "/workspace/zhongkai/profiling_result/mmlu/%s" % (subject_list[3])
    sub_trace_path_list4 = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list4.sort()
    
    dataset_dir = "/workspace/zhongkai/profiling_result/livecodebench/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "%s/%s" % (dataset_dir, subject_list[0])
    sub_trace_path_list5 = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list5.sort()

    dataset_dir = "/workspace/zhongkai/profiling_result/HuggingFaceH4/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "%s/%s" % (dataset_dir, subject_list[0])
    sub_trace_path_list6 = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list6.sort()

    dataset_dir = "/workspace/zhongkai/profiling_result/Chinese-SimpleQA/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "%s/%s" % (dataset_dir, subject_list[0])
    sub_trace_path_list7 = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list7.sort()
    
    dataset_dir = "/workspace/zhongkai/profiling_result/Chinese-SimpleQA/"
    subject_list = sorted(os.listdir(dataset_dir))
    dataset_subject_dir = "%s/%s" % (dataset_dir, subject_list[1])
    sub_trace_path_list8 = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
    sub_trace_path_list8.sort()

    # print(sub_trace_path_list1)


    sub_lists = [
        sub_trace_path_list1,
        sub_trace_path_list5,
        sub_trace_path_list2,
        sub_trace_path_list6,
        sub_trace_path_list3,
        sub_trace_path_list7,
        sub_trace_path_list4,
        sub_trace_path_list8,
    ]
    result = [item for group in zip_longest(*sub_lists, fillvalue=None) for item in group if item is not None]

    # print(result)
    return result


def gen_full_trace_data_list():
    dataset_root_dir = "/workspace/zhongkai/profiling_result/" 
    full_trace_path_list = []
    dataset_list = sorted(os.listdir(dataset_root_dir))
    for dataset in dataset_list:
        dataset_dir = "%s/%s" % (dataset_root_dir, dataset)
        # dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/"
        subject_list = sorted(os.listdir(dataset_dir))
        for subject in subject_list:
            dataset_subject_dir = "%s/%s" % (dataset_dir, subject)
            subject_trace_path_list = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
            full_trace_path_list += subject_trace_path_list   
    return full_trace_path_list


def test_data():
    dataset_dir = "/workspace/zhongkai/profiling_result/mmlu/"
    full_trace_path_list = []
    subject_list = sorted(os.listdir(dataset_dir))
    for subject in subject_list:
        dataset_subject_dir = "/workspace/zhongkai/profiling_result/mmlu/%s" % (subject)
        subject_trace_path_list = [os.path.join(dataset_subject_dir, f) for f in os.listdir(dataset_subject_dir) if os.path.isfile(os.path.join(dataset_subject_dir, f))]
        full_trace_path_list += subject_trace_path_list        
        
    # full_trace_path_list = gen_trace_data_list()
    full_trace_path_list = gen_full_trace_data_list()
    print("tot trace path length", len(full_trace_path_list))
    full_trace_path_list = full_trace_path_list[0: 100] 
    all_data = []
    output_len_list = []
    input_len_list = []
    for file_path in full_trace_path_list:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.append(data)
            output_len_list.append(len(data))
            input_len = len(data[0]['3'])
            input_len_list.append(input_len)
    # for name in full_trace_path_list:
    #     print(name)
    # print(output_len_list)
    # print(input_len_list)


if __name__ == "__main__":
    # main()
    test_data()
