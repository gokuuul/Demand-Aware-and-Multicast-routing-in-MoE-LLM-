# Project notes for AI agents

## Traces live OUTSIDE this repo. Do not move them back in.
- 72 GB of expert-routing JSON (~20k files per model) lives at `/home/gokul/moe_traces`.
- The path comes from `MOE_TRACE_ROOT` in `.env`, read by `DEFAULT_TRACE_ROOT` in `data_loader.py`.
- It was moved out because 72 GB of small files inside the workspace froze VS Code's
  file watcher, search index, and agent indexing.
- NEVER run `main_ae.py` without `--skip-download`. Without it, `snapshot_download`
  re-fetches into `./moe_profiling_trace/` and recreates the problem.
- Do not create a symlink at `./moe_profiling_trace`. Watchers follow it.

## Never run the simulator yourself
- `main_ae.py` takes 8-12 hours and caches tens of GB in RAM. A tool call on it
  blocks indefinitely.
- The user runs it manually in tmux under a memory cap. Suggest the command; don't execute it.
- Safe to run: `--skip-sim --skip-plot` for path checks, or the plotting scripts on
  existing `results/`.

## Machine limits
- 64 GiB RAM, 28-thread i7-14700, no GPU. Batch 4096 is safe; 16384 needs ~50 GB
  and the editor closed. `--parallel` across models needs ~200 GB: not possible here.
- `main_ae.py` overwrites `results_bp/` with `results/` on every startup.

## Don't open trace JSON in the editor
Inspect with a python one-liner instead.
