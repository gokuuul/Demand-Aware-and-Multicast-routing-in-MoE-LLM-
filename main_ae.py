#!/usr/bin/env python3
"""
One-click Artifact Evaluation (AE) script for ISCA paper.

Usage:
    python main_ae.py                          # default: qwen, (8,3), batch 4096/8192/16384
    python main_ae.py --models qwen,deepseek   # multiple models
    python main_ae.py --models all             # all 4 models
    python main_ae.py --models all --parallel  # all 4 models in parallel
    python main_ae.py --archs 8,3;5,5          # multiple architectures
    python main_ae.py --batches 4096,8192      # custom batch sizes

Steps:
    1. Check / download trace files from HuggingFace
    2. Backup results/ to results_bp/
    3. Run simulator (optionally in parallel per model)
    4. Generate figures in figures_ae/
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Dependency check ────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
}

def check_and_install_deps():
    """Check core dependencies and instruct the user to install missing ones.

    On "externally managed" systems (PEP 668) this script will not attempt to
    install packages automatically. Instead it prints clear instructions to
    create a virtual environment or install distribution packages and exits.
    """
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[AE] Missing dependencies: {missing}")
        print("Please install them in a virtual environment, or install the ")
        print("distribution packages. Recommended steps:")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  python -m pip install --upgrade pip")
        print("  pip install " + " ".join(missing))
        print("\nAlternatively, to install system packages (Debian/Ubuntu):")
        print("  sudo apt update && sudo apt install python3-numpy python3-matplotlib python3-pandas python3-venv")
        print("\nAfter installing dependencies, re-run: python main_ae.py")
        sys.exit(1)

check_and_install_deps()

SCRIPT_DIR = Path(__file__).resolve().parent
TRACE_DIR = SCRIPT_DIR / "moe_profiling_trace"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_BP_DIR = SCRIPT_DIR / "results_bp"
FIGURES_DIR = SCRIPT_DIR / "figures_ae"

HF_REPO = "core12345/MoE_expert_selection_trace"

# Model name -> HuggingFace subdirectory
MODEL_HF_PATHS = {
    "deepseek": "cognitivecomputations/DeepSeek-R1-AWQ",
    "qwen":     "Qwen/Qwen3-235B-A22B-FP8",
    "kimi":     "moonshotai/Kimi-K2-Thinking",
    "llama4":   "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
}

# Datasets used by gen_full_trace_data_list
REQUIRED_DATASETS = ["mmlu", "mmlu_ZH_CN"]

ALL_STRATEGIES = ["base", "norm_ep", "allo_only", "pred_only", "allo_and_pred"]


def parse_args():
    parser = argparse.ArgumentParser(description="ISCA Artifact Evaluation")
    parser.add_argument("--models", type=str, default="qwen",
                        help="Comma-separated model names or 'all' (default: qwen)")
    parser.add_argument("--archs", type=str, default="8,3",
                        help="Semicolon-separated y,x pairs (default: '8,3'). "
                             "Example: '8,3;5,5'")
    parser.add_argument("--batches", type=str, default="4096,8192,16384",
                        help="Comma-separated batch sizes (default: 4096,8192,16384)")
    parser.add_argument("--strategies", type=str, default=",".join(ALL_STRATEGIES),
                        help="Comma-separated strategies (default: all except allo_only_org)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run models in parallel (one subprocess per model)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip trace download even if missing")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation (use existing results)")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip plot generation")
    # Internal flag: used by parallel subprocess to run a single model
    parser.add_argument("--_worker-model", type=str, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_archs(s):
    """Parse '8,3;5,5' -> [(8,3), (5,5)]"""
    archs = []
    for part in s.split(";"):
        y, x = part.strip().split(",")
        archs.append((int(y), int(x)))
    return archs


# ── Step 1: Trace download ──────────────────────────────────────────────

def _find_trace_root(model):
    """Return the trace root directory for a model, checking default then local."""
    from data_loader import DEFAULT_TRACE_ROOT, MODEL_TRACE_DIRS
    subdir = MODEL_TRACE_DIRS[model]

    # Check default path first
    default_path = os.path.join(DEFAULT_TRACE_ROOT, subdir)
    for ds in REQUIRED_DATASETS:
        ds_path = os.path.join(default_path, ds)
        if os.path.isdir(ds_path) and any(path.is_file() for path in Path(ds_path).rglob("*")):
            return DEFAULT_TRACE_ROOT

    # Check local moe_profiling_trace/
    local_path = os.path.join(TRACE_DIR, subdir)
    for ds in REQUIRED_DATASETS:
        ds_path = os.path.join(local_path, ds)
        if os.path.isdir(ds_path) and any(path.is_file() for path in Path(ds_path).rglob("*")):
            return str(TRACE_DIR)

    return None


def _ensure_huggingface_hub():
    """Ensure huggingface_hub is available; if not, print instructions and exit."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[AE] Missing dependency: huggingface_hub")
        print("Please install it in your environment (recommended: virtualenv):")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install huggingface_hub")
        print("\nAfter installing, re-run: python main_ae.py")
        sys.exit(1)


def download_traces(models):
    """Download trace files from HuggingFace for models that are missing."""
    models_to_download = []
    trace_roots = {}

    for model in models:
        root = _find_trace_root(model)
        if root:
            trace_roots[model] = root
            print(f"[AE] Traces for '{model}' found at {root}")
        else:
            models_to_download.append(model)

    if not models_to_download:
        return trace_roots

    _ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    for model in models_to_download:
        subdir = MODEL_HF_PATHS[model]
        allow = [f"{subdir}/{ds}/**" for ds in REQUIRED_DATASETS]

        print(f"[AE] Downloading traces for '{model}' from HuggingFace ...")
        print(f"      Patterns: {allow}")
        t0 = time.time()

        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=allow,
            local_dir=str(TRACE_DIR),
        )

        elapsed = time.time() - t0
        print(f"      Done in {elapsed:.1f}s")
        trace_roots[model] = str(TRACE_DIR)

    return trace_roots


# ── Step 2: Backup results ──────────────────────────────────────────────

def backup_results():
    """Backup existing results/ to results_bp/, then create fresh results/."""
    if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()):
        # If results_bp already exists, remove it first
        if RESULTS_BP_DIR.exists():
            shutil.rmtree(str(RESULTS_BP_DIR))
        print(f"[AE] Backing up results/ -> results_bp/")
        shutil.move(str(RESULTS_DIR), str(RESULTS_BP_DIR))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 3: Run simulation ──────────────────────────────────────────────

def run_simulations(models, archs, batches, strategies, trace_roots):
    """Patch experiment_config and run the benchmark suite (sequential)."""
    import experiment_config
    experiment_config.MODELS = list(models)
    experiment_config.CHIPLET_ARCHITECTURES = list(archs)
    experiment_config.BATCH_SIZES = sorted(batches)
    experiment_config.STRATEGIES = list(strategies)

    import data_loader
    _original_gen = data_loader.gen_full_trace_data_list

    def _patched_gen(model="deepseek", root_override=None):
        override = trace_roots.get(model)
        return _original_gen(model, root_override=override)

    data_loader.gen_full_trace_data_list = _patched_gen

    import main as main_module
    main_module.MODELS = experiment_config.MODELS
    main_module.CHIPLET_ARCHITECTURES = experiment_config.CHIPLET_ARCHITECTURES
    main_module.BATCH_SIZES = experiment_config.BATCH_SIZES
    main_module.STRATEGIES = experiment_config.STRATEGIES
    main_module.gen_full_trace_data_list = _patched_gen

    print(f"\n[AE] Running simulations (sequential):")
    print(f"      Models:     {experiment_config.MODELS}")
    print(f"      Archs:      {experiment_config.CHIPLET_ARCHITECTURES}")
    print(f"      Batches:    {experiment_config.BATCH_SIZES}")
    print(f"      Strategies: {experiment_config.STRATEGIES}")
    print()

    t0 = time.time()
    main_module.run_benchmark_suite()
    elapsed = time.time() - t0
    print(f"\n[AE] Simulation completed in {elapsed:.1f}s")


def run_simulations_parallel(models, archs, batches, strategies, trace_roots):
    """Launch one subprocess per model, wait for all to finish."""
    print(f"\n[AE] Running simulations (parallel, {len(models)} processes):")
    for m in models:
        print(f"      - {m}")
    print()

    # Build common args
    archs_str = ";".join(f"{y},{x}" for y, x in archs)
    batches_str = ",".join(str(b) for b in batches)
    strategies_str = ",".join(strategies)

    processes = {}
    log_files = {}
    for model in models:
        log_path = SCRIPT_DIR / f"ae_{model}.log"
        log_f = open(log_path, "w")
        log_files[model] = (log_path, log_f)

        cmd = [
            sys.executable, str(SCRIPT_DIR / "main_ae.py"),
            "--models", model,
            "--archs", archs_str,
            "--batches", batches_str,
            "--strategies", strategies_str,
            "--skip-download",  # traces already downloaded by parent
            "--skip-plot",      # parent will plot after all finish
            "--_worker-model", model,
        ]
        print(f"[AE] Starting {model} (log: {log_path.name})")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                             cwd=str(SCRIPT_DIR))
        processes[model] = p

    # Wait for all
    t0 = time.time()
    failed = []
    for model, p in processes.items():
        p.wait()
        log_path, log_f = log_files[model]
        log_f.close()
        if p.returncode != 0:
            failed.append(model)
            print(f"[AE] ERROR: {model} failed (exit code {p.returncode}). "
                  f"See {log_path.name}")
        else:
            print(f"[AE] {model} completed successfully.")

    elapsed = time.time() - t0
    print(f"\n[AE] All simulations finished in {elapsed:.1f}s")

    if failed:
        print(f"[AE] WARNING: {len(failed)} model(s) failed: {failed}")
        print(f"     Check log files: {', '.join(f'ae_{m}.log' for m in failed)}")


# ── Step 4: Plot ────────────────────────────────────────────────────────

def generate_plots():
    """Generate the stacked e2e figure from results/."""
    from draw_e2e_ae import process_ae_results
    print(f"\n[AE] Generating figures ...")
    process_ae_results(data_dir=RESULTS_DIR, out_dir=FIGURES_DIR)
    print(f"[AE] Figures saved to {FIGURES_DIR}/")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Parse arguments
    if args.models.lower() == "all":
        models = list(MODEL_HF_PATHS.keys())
    else:
        models = [m.strip() for m in args.models.split(",")]
    archs = parse_archs(args.archs)
    batches = [int(b.strip()) for b in args.batches.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]

    # If this is a worker subprocess, run single model and exit
    if args._worker_model:
        trace_roots = download_traces(models)  # will find existing traces
        run_simulations(models, archs, batches, strategies, trace_roots)
        return

    print("=" * 60)
    print("  ISCA Artifact Evaluation")
    print("=" * 60)
    print(f"  Models:       {models}")
    print(f"  Architectures:{archs}")
    print(f"  Batch sizes:  {batches}")
    print(f"  Strategies:   {strategies}")
    print(f"  Parallel:     {args.parallel}")
    print("=" * 60)

    # Step 1: Download traces
    if not args.skip_download:
        trace_roots = download_traces(models)
    else:
        trace_roots = {m: _find_trace_root(m) for m in models}
        print("[AE] Skipping trace download (--skip-download)")

    # Step 2 & 3: Backup results and run simulation
    if not args.skip_sim:
        backup_results()
        if args.parallel and len(models) > 1:
            run_simulations_parallel(models, archs, batches, strategies,
                                     trace_roots)
        else:
            run_simulations(models, archs, batches, strategies, trace_roots)
    else:
        print("[AE] Skipping simulation (--skip-sim)")

    # Step 4: Plot
    if not args.skip_plot:
        generate_plots()
    else:
        print("[AE] Skipping plot generation (--skip-plot)")

    print("\n" + "=" * 60)
    print("  AE completed successfully!")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
