"""
Data loading utilities for trace files.

This module provides functions to load and organize trace file paths from various datasets.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any
from itertools import zip_longest


def load_trace_files(path_list: List[str], max_workers: int = 32) -> List[Any]:
    """
    Load multiple trace JSON files in parallel.

    Use this to load once per (model, batch_size) and pass the result to
    Simulator.process_trace() to avoid repeated file I/O when running
    multiple strategies on the same batch.

    Args:
        path_list: List of paths to trace JSON files
        max_workers: Max threads for parallel loading (default 32)

    Returns:
        List of parsed trace dicts in the same order as path_list
    """
    # Unwrap if caller passed a single-element list by mistake (e.g. [paths] or [trace_data])
    if path_list and isinstance(path_list[0], list):
        path_list = path_list[0]
    # If already pre-loaded trace data (list of dicts), return as-is
    if path_list and isinstance(path_list[0], dict):
        return path_list

    def _load_one(idx_path: tuple) -> tuple:
        idx, path = idx_path
        if not isinstance(path, str):
            raise TypeError(
                "load_trace_files expects a list of path strings; got list of %s. "
                "Did you pass pre-loaded data? Use process_trace(trace_data) or run_simulation(trace_data, batch_size) instead."
                % type(path).__name__
            )
        with open(path, "r", encoding="utf-8") as f:
            return (idx, json.load(f))

    n = len(path_list)
    workers = min(max_workers, n)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_load_one, enumerate(path_list)))
    results.sort(key=lambda x: x[0])
    
    results = [r[1] for r in results]
    return results

def get_trace_paths_from_directory(dataset_dir: str) -> List[str]:
    """
    Get all trace file paths from a directory.
    
    Args:
        dataset_dir: Path to the directory containing trace files
        
    Returns:
        List of trace file paths, sorted
    """
    trace_path_list = [
        os.path.join(dataset_dir, f) 
        for f in os.listdir(dataset_dir) 
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]
    trace_path_list.sort()
    return trace_path_list


def gen_trace_data_list() -> List[str]:
    """
    Generate a mixed trace data list from multiple datasets.
    
    This function combines traces from multiple datasets in an interleaved manner.
    
    Returns:
        List of trace file paths
    """
    # MMLU dataset subjects
    mmlu_dir = "/workspace/zhongkai/profiling_result/mmlu/"
    subject_list = sorted(os.listdir(mmlu_dir))
    
    sub_trace_path_list1 = get_trace_paths_from_directory(
        os.path.join(mmlu_dir, subject_list[0])
    )
    sub_trace_path_list2 = get_trace_paths_from_directory(
        os.path.join(mmlu_dir, subject_list[1])
    )
    sub_trace_path_list3 = get_trace_paths_from_directory(
        os.path.join(mmlu_dir, subject_list[2])
    )
    sub_trace_path_list4 = get_trace_paths_from_directory(
        os.path.join(mmlu_dir, subject_list[3])
    )
    
    # LiveCodeBench dataset
    livecodebench_dir = "/workspace/zhongkai/profiling_result/livecodebench/"
    livecodebench_subject = sorted(os.listdir(livecodebench_dir))[0]
    sub_trace_path_list5 = get_trace_paths_from_directory(
        os.path.join(livecodebench_dir, livecodebench_subject)
    )
    
    # HuggingFaceH4 dataset
    hf_dir = "/workspace/zhongkai/profiling_result/HuggingFaceH4/"
    hf_subject = sorted(os.listdir(hf_dir))[0]
    sub_trace_path_list6 = get_trace_paths_from_directory(
        os.path.join(hf_dir, hf_subject)
    )
    
    # Chinese-SimpleQA dataset
    simpleqa_dir = "/workspace/zhongkai/profiling_result/Chinese-SimpleQA/"
    simpleqa_subjects = sorted(os.listdir(simpleqa_dir))
    sub_trace_path_list7 = get_trace_paths_from_directory(
        os.path.join(simpleqa_dir, simpleqa_subjects[0])
    )
    sub_trace_path_list8 = get_trace_paths_from_directory(
        os.path.join(simpleqa_dir, simpleqa_subjects[1])
    )
    
    # Interleave all lists
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
    
    result = [
        item 
        for group in zip_longest(*sub_lists, fillvalue=None) 
        for item in group 
        if item is not None
    ]
    
    return result


MODEL_TRACE_DIRS = {
    "deepseek": "cognitivecomputations/DeepSeek-R1-AWQ",
    "qwen":     "Qwen/Qwen3-235B-A22B-FP8",
    "kimi":     "moonshotai/Kimi-K2-Thinking",
    "llama4":   "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
}

DEFAULT_TRACE_ROOT = os.environ.get("MOE_TRACE_ROOT", "/workspace/zhongkai/profiling_result")


def gen_full_trace_data_list(model: str = "deepseek", root_override: str = None) -> List[str]:
    """
    Generate full trace data list for a specific model.

    Args:
        model: Model name ("deepseek", "qwen", "kimi", or "llama4")
        root_override: If provided, use this as the root directory instead of
                       the default. The model subdirectory is appended automatically.

    Returns:
        List of trace file paths

    Raises:
        ValueError: If model name is not supported
    """
    if model not in MODEL_TRACE_DIRS:
        raise ValueError(f"Undefined model name: {model}. Supported: {list(MODEL_TRACE_DIRS.keys())}")

    base = root_override if root_override else DEFAULT_TRACE_ROOT
    dataset_root_dir = os.path.join(base, MODEL_TRACE_DIRS[model])
    
    full_trace_path_list = []
    dataset_list = sorted(os.listdir(dataset_root_dir))
    
    for dataset in dataset_list:
        # Only process mmlu datasets
        if dataset not in ["mmlu", "mmlu_ZH_CN"]:
            continue
            
        dataset_dir = os.path.join(dataset_root_dir, dataset)
        subject_list = sorted(os.listdir(dataset_dir))
        
        for subject in subject_list:
            dataset_subject_dir = os.path.join(dataset_dir, subject)
            # Hugging Face snapshots can include per-subject ``.tar.gz``
            # archives beside the extracted subject directories.  Only the
            # extracted directories contain JSON routing traces.
            if not os.path.isdir(dataset_subject_dir):
                continue
            query_list = sorted(os.listdir(dataset_subject_dir))
            subject_trace_path_list = [
                os.path.join(dataset_subject_dir, f) 
                for f in query_list 
                if os.path.isfile(os.path.join(dataset_subject_dir, f))
            ]
            full_trace_path_list.extend(subject_trace_path_list)
    
    return full_trace_path_list


def rotate_trace_list(trace_list: List[str], start_index: int) -> List[str]:
    """
    Rotate trace list so that start_index becomes the first element, wrapping around.

    Example: rotate_trace_list([a,b,c,d,e], 3) -> [d,e,a,b,c]

    Args:
        trace_list: Original list of trace file paths
        start_index: Index to use as the new starting point

    Returns:
        Rotated list of the same length
    """
    n = len(trace_list)
    if n == 0:
        return trace_list
    start_index = start_index % n
    return trace_list[start_index:] + trace_list[:start_index]
