# Wafer-Scale MoE Chiplet Simulator — Technical Reference

A trace-driven, discrete-event simulator for Mixture-of-Experts inference on a
2D mesh of chiplets. It replays recorded expert-routing decisions from real MoE
models, decides which die executes each expert, walks every resulting weight
fetch through a cache/DRAM/D2D hierarchy, and reports end-to-end time,
inter-die hop count, DRAM traffic, and load balance.

The simulator does **not** compute activations. It models *time* and *traffic*:
who reads what, from where, over which link, and how long each shared resource
is occupied.

---

## 1. Repository map

| Path | Role |
|---|---|
| `main_ae.py` | Artifact-Evaluation driver: trace discovery/download, results backup, simulation sweep, plotting |
| `main.py` | Benchmark suite: the `model x strategy x batch x arch` loop, CSV writer |
| `experiment_config.py` | What to sweep: `MODELS`, `CHIPLET_ARCHITECTURES`, `STRATEGIES`, `BATCH_SIZES`, `STRATEGY_MAP` |
| `data_loader.py` | Trace path discovery (`gen_full_trace_data_list`), parallel JSON loading, `MOE_TRACE_ROOT` |
| `simulator_runner.py` | One-run wrapper + metric collection + CSV line formatting |
| `plotting.py` | Decode-time-vs-batch validation plots (verify modes only) |
| `draw_e2e_ae.py` | AE figures: `e2e_stacked`, `available_results`, `e2e_summary.csv` |
| `ep_balancer.py` | Standalone DeepSeek-style expert-replication balancer (torch). Not wired into the simulator |
| `test_dataset.py` | Scratch single-run harness with hard-coded paths |
| `config.py`, `sim.py` | Backward-compat re-exports of `SystemConfig` / `Simulator` |
| `setup_env.sh` | Creates `.venv` and installs numpy/matplotlib/pandas/huggingface_hub |
| `toy_chiplet_sim/` | The simulator package |

Inside the package:

```
toy_chiplet_sim/
  config.py                    SystemConfig (topology, model, strategy), TimingConfig (latencies)
  models/
    address.py                 Addr_expert (expert, layer, matrix, slice), Addr_chip
    expert.py                  Expert, Expert_slice
  hardware/
    chiplet.py                 One die: Cache + DRAM + mesh position
    cache.py                   Set-associative LLC with per-set LRU
    dram.py                    Per-die DRAM as a set of resident Addr_expert
    interconnect.py            Mesh construction, XY routing, ring routing
  simulation/
    events.py                  EventType, ResourceType, Resource, SimulationEvent
    resource_manager.py        Resource grant/release + per-resource wait queues
    simulator.py               Everything else (~2.7k lines): the event loop,
                               allocation strategies, cache/DRAM policies,
                               prediction, statistics
```

---

## 2. The machine being modelled

### Topology

`Interconnect` builds `y_chiplets * x_chiplets` dies in a 2D mesh.
Die id is `y + x * y_chiplets`, so ids run down each column. Neighbours are
connected north–south and east–west; there is no wraparound. Routing is
**dimension-order XY** (`_xy_routing`): move in X to the destination column,
then in Y. A ring router exists (`_ring_routing`) but `InterconnectType` is
hard-set to `MESH` in `SystemConfig.__init__`, and `CROSSBAR` is unimplemented.

The AE sweeps two shapes: `(8,3)` = 24 dies and `(5,5)` = 25 dies.

### Per-die resources

Each die owns four things the event loop can contend for:

| Resource | Count | Held by |
|---|---|---|
| `COMPUTE_UNIT` | 1 per die | `COMPUTATION` events |
| `CACHE_PORT` | 1 per die | `CACHE_ACCESS`, `CACHE_STORE` |
| `DRAM_PACKAGE` | 1 per die | `MEMORY_ACCESS`, `DRAM_STORE` |
| `D2D_LINK` | 1 per mesh edge | `NET_TRANS_REQ`, `NET_TRANS_DATA` — **every link on the route at once** |

A resource is a binary semaphore, not a bandwidth pool. Holding all links on a
path simultaneously models wormhole-style circuit reservation: a long transfer
blocks every hop it crosses for its whole duration.

### Hardware constants (`SystemConfig`, `TimingConfig`)

```
compte_power_per_die  = 1000        # TFLOP/s  -> 1e6 ops/ns
dram_bw_per_die       = 3350        # bytes/ns (= GB/s), HBM-class
d2d_link_bw           = 1741        # bytes/ns per link
cache_size            = 64 MiB      # LLC per die
cache_assoc           = 2
line_size             = one expert slice
dram_size_per_die     = 1000        # GB — recorded, never enforced

cache_hit_latency     = 100 ns
cache_miss_penalty    = 110 ns
cache_store_latency   =  30 ns
link_latency          = 200 ns      # per hop, per direction
mem_access_latency    = 300 ns
```

`dram_size_per_die` and `hardware_manage_dram_size_per_die` are stored but no
capacity check or eviction ever runs — DRAM is effectively unbounded. The LRU
eviction path in `hardware/dram.py` is present but commented out for speed.

### Models

Model geometry is hard-coded in `SystemConfig.__init__`:

| Model | Layers | Hidden | FFN dim | Experts | top-k | Shared |
|---|---:|---:|---:|---:|---:|---:|
| `deepseek` | 61 | 7168 | 2048 | 256 | 8 | 1 |
| `qwen` | 94 | 4096 | 1536 | 128 | 8 | 0 |
| `kimi` | 61 | 7168 | 2048 | 384 | 8 | 1 |
| `llama4` | 48 | 5120 | 8192 | 128 | 1 | 0 |

`routed_expert_num` and `shared_expert_num` are informational only — the actual
top-k comes from the trace, and shared experts are never simulated.

### Expert sizing and slicing

An expert is three matrices (`expert_matrix_num = 3`, i.e. gate/up/down) at
`data_type = 2` bytes:

```
expert_size       = 3 * hidden_dim * expert_dim * 2
expert_slice_size = expert_size / 3            # expert_matrix_slice_num = 1
expert_tot_slice_num = 3
```

The slice is the unit of addressing, caching, transfer and event generation.
One allocated `(expert, die)` pair produces **3 independent events**, one per
matrix.

| Model | Expert | Slice | All experts, all layers | LLC sets | Slices resident in LLC |
|---|---:|---:|---:|---:|---:|
| `deepseek` | 84 MB | 28 MB | 1281 GB | 1 | 2 |
| `qwen` | 36 MB | 12 MB | 423 GB | 2 | 4 |
| `kimi` | 84 MB | 28 MB | 1922 GB | 1 | 2 |
| `llama4` | 240 MB | 80 MB | 1440 GB | 1 | 2 |

Note the consequence: `num_sets = cache_size // (assoc * line_size)`, and a
slice is 12–80 MB against a 64 MiB LLC. **The cache holds 2–4 slices.** At
batch sizes where hundreds of experts are live per layer, the LLC is
effectively a small victim buffer, not a working-set cache. Cache-index is
`expert_id % num_sets`, so with 1 set every slice collides.

### Expert placement

`_expert_placement_map(address)` maps expert → home die using **only the expert
id**, so an expert lives on the same die in every layer:

```
expert_per_die = floor(expert_num / chiplet_num)
die            = floor(expert_id / expert_per_die)
if expert_id > expert_per_die * chiplet_num - 1:   # leftover experts
    die = expert_id % chiplet_num
```

For qwen on 24 dies: 5 experts per die covers ids 0–119; ids 120–127 wrap onto
dies 0–7, so those eight dies hold 6 experts. This is the static "EP placement"
baseline that every strategy starts from. At init, every slice of every expert
of every layer is written into its home die's DRAM set.

---

## 3. Traces

### Source and layout

Traces are per-query JSON files of recorded expert routing, one directory tree
per model:

```
$MOE_TRACE_ROOT/<hf_org>/<hf_model>/{mmlu,mmlu_ZH_CN}/<subject>/<query>.json
```

`MOE_TRACE_ROOT` comes from the environment (`.env` in this checkout points at
`/home/gokul/moe_traces`); the fallback in `data_loader.py` is the original
authors' `/workspace/zhongkai/profiling_result`. Only `mmlu` and `mmlu_ZH_CN`
are read; other datasets in the tree are skipped, and non-directory entries
(HuggingFace `.tar.gz` archives) are ignored.

`gen_full_trace_data_list(model)` walks that tree in sorted order and returns
a flat list of ~20k file paths. `main.py` then applies
`rotate_trace_list(..., start_index=8192)`, which moves the first 8192 paths to
the end so a run does not always begin at the same MMLU subject.

### File format

One file = one query = a list of iterations.

```jsonc
[
  { "0": [[105,7,63,...], [35,69,...], ...],   // iteration 0 = prefill, one row per prompt token
    "1": [[...], ...],
    ...
    "93": [[...], ...] },
  { "0": [[69,10,125,...]],                    // iteration 1..N = decode, exactly one row
    ... },
  ...
]
```

Keys are layer ids as strings. A value of `null` means the layer is not an MoE
layer. Each row is the list of expert ids selected for one token. A qwen trace
is 129 iterations x 94 layers; iteration 0 carries the full prompt
(e.g. 25 tokens), iterations 1+ carry one token each.

### Batch semantics

**`batch_size` = number of trace files** = number of concurrent requests. The
simulator does not track individual requests. `_trace_reader_core` merges the
whole batch into one counter per `(iteration, layer)`:

```
aggregated[iter][layer] = Counter{expert_id: total_selections_across_the_batch}
```

`stage_map[(iter, layer)]` is `"prefill"` if iteration 0 has more than one token
row, `"decode"` if exactly one row. Iteration count is **hard-capped at 128** in
`_trace_reader_core` (`max_iter_num = 128`), so every run is 1 prefill + 127
decode steps regardless of trace length. `max_layer_num` is taken from
iteration 1 of the first trace.

The generator then yields `(expert_dict, layer, stage, prefill_len_list)` once
per `(iteration, layer)` in order.

---

## 4. Execution model

### Layer lockstep

The simulator advances **one `(iteration, layer)` at a time**, with a global
barrier between them. `_schedule_next_request()` pulls the next `(iter, layer)`
from the trace generator, runs allocation, and emits every event for that layer.
The next layer is only scheduled when `_handle_request_completion` observes

```
pending_requests == 0  and  event_queue empty  and  every resource wait queue empty
```

So there is no cross-layer or cross-token pipelining: layer *n+1* starts after
the last die finishes layer *n*. Attention is not modelled — the
`self.operation = "attention"` toggle is commented out in
`_schedule_next_request`, and `layer_attn_time` is forced to `0` regardless.

### The event loop

```
while event_queue:
    event = heappop(event_queue)          # earliest time first
    prev = current_time
    current_time = event.time
    event.callback(event)                 # advances current_time, enqueues successors
    end = current_time
    current_time = max(prev, end)
    release event.resources at `end`
    for each released resource: try to admit its highest-priority waiter
```

`Simulator.current_time` therefore ends up as the maximum event end-time seen —
that is the reported total simulated time.

### Admission and wait queues

`_add_event` checks whether *all* of an event's resources are free. If so it
claims them and pushes the event onto the time-ordered event queue; if not, the
event goes onto the wait queue of each of its resources. A resource is held from
the moment the event is admitted until its callback returns, so the reservation
covers both the queueing delay and the transfer itself.

Wait queues are heaps ordered by `(-priority, time)`. Events for an expert whose
allocated die differs from its home die are created with `priority = 1.0`; local
ones get `0.0`. Remote work therefore jumps the queue, because a remote fetch has
a longer critical path. Priority is inherited by every successor event in the
chain via `_current_event_priority`.

`_try_swap_into_event_queue` (preemption of an already-admitted lower-priority
event) exists but its call site is commented out.

### Request lifecycle

One event chain per `(expert, die, matrix)` triple. All chains end in
`COMPLETION_SIGNAL -> _handle_request_completion`, which records the chain's
`(start, end, duration)` against its die and decrements `pending_requests`.

```
CACHE_ACCESS  (cache port)
  |
  +-- local LLC hit ........ +100ns ................................ COMPUTATION
  |
  +-- local DRAM hit ....... +110ns -> MEMORY_ACCESS (dram) ........ COMPUTATION
  |                                       \-> CACHE_STORE (local LLC)
  |
  +-- remote LLC hit ....... +110ns -> NET_TRANS_REQ (all links)
  |                                 -> CACHE_ACCESS (remote port)
  |                                 -> NET_TRANS_DATA (all links) .. COMPUTATION
  |                                       \-> DRAM_STORE (if predicted)
  |
  +-- remote DRAM .......... +110ns -> NET_TRANS_REQ (all links)
                                    -> MEMORY_ACCESS (remote dram)
                                          \-> CACHE_STORE (remote LLC, CACHE_LOCAL)
                                    -> NET_TRANS_DATA (all links) .. COMPUTATION
                                          \-> DRAM_STORE (if predicted)
                                          \-> CACHE_STORE (local LLC, CACHE_ALL)
```

Side-effect events (`CACHE_STORE`, `DRAM_STORE`) are independent chains: they
increment `pending_requests` too, so the layer barrier waits for write-backs and
predicted replications to land before the next layer starts.

### Latency formulas used during execution

```
cache hit                 100 ns
miss penalty (any miss)   110 ns
cache store                30 ns

DRAM access   _cal_mem_latency(size, batch, is_remote)
              = 300 + ceil(size / (3350 * u))          u = 0.9, or 0.85 for llama4
              cold_start_latency is present in the code but set to 0 everywhere

Network       _cal_network_latency(path, size)
              = 200 * (len(path) - 1) + ceil(size / (1741 * 0.9))
              request messages use size = 16 bytes; data transfers use one slice

Compute       _cal_compute_latency(ops, batch)
              = ceil(ops / (1e6 * u(model, batch)))
              ops = 2 * 3 * hidden * expert_dim * batch / 3   (i.e. one slice)
              u is a per-model, per-batch utilisation table (0.5-0.72), calibrated
              against measured GEMM throughput
```

Local DRAM reads charge for weights **plus activations**:

```
weight     = expert_slice_size
activation = batch * (hidden_dim + expert_dim) * 3 * 2 / 3
```

Remote DRAM reads charge for the **weight slice only** — the activation term is
not applied on the remote path.

---

## 5. Cache and DRAM strategies

`ExeStrategy` selects both a cache policy and whether prediction is on.

| `ExeStrategy` | Cache policy | Lookup order |
|---|---|---|
| `BASELINE_CACHE_LOCAL` | `CACHE_LOCAL` | local LLC (only if the die is the home die) -> local DRAM, else remote LLC -> remote DRAM |
| `BASELINE_CACHE_ALL` | `CACHE_ALL` | local LLC -> **all** remote LLCs -> local/remote DRAM |
| `BASELINE_DISABLE_CACHE` | `DISABLED` | same routing as `CACHE_LOCAL`, but no `CACHE_STORE` events are ever created |
| `PRED_NEXT_LAYER` | `CACHE_LOCAL` | `_pred_next_layer` returns `False` unconditionally — a stub |
| `PRED_NEXT_TOKEN` | `CACHE_LOCAL` | `_process_request_new`: local LLC -> local DRAM -> remote LLC -> remote DRAM |

Where the data is cached differs by policy:

* `CACHE_LOCAL` — an LLC only caches lines its **own** DRAM served. A remote
  DRAM read populates the *remote* (home) die's LLC, not the requester's.
* `CACHE_ALL` — the requester also caches remote data locally, and a miss
  searches every other die's LLC (`_search_remote_llc` is an O(dies) scan with
  no modelled cost).

`PRED_NEXT_TOKEN` is the interesting one. It is the only mode where the local
DRAM lookup can succeed on a die that is not the expert's home die, because
prediction **replicates expert slices into local DRAM**.

### Prediction (`_pred_next_token`)

Built from a co-activation heatmap, `token_pred_heatmap[layer][e_prev][e_next]`:

* Prefill contributes intra-prompt pairs: for consecutive tokens in iteration 0,
  every (expert at token *t*, expert at token *t+1*) pair is counted
  (`_stat_prefill_pairs`).
* Decode contributes cross-iteration pairs: for iteration *i >= 2*, every
  (expert at *i-1*, expert at *i*) pair, accumulated as a running cumulative sum
  over iterations (`_aggregate_decode_cross_token_pairs`). The heatmap presented
  at iteration *i* is the cumulative count through *i*.

When a remote fetch returns, the decision to replicate is:

> For each expert already allocated to *this* die in *this* layer, take the top
> 20% of experts by co-activation frequency with it. If the fetched expert is in
> any of those sets, store it in local DRAM.

Prediction never fires during prefill. Replication is permanent for the rest of
the run (no capacity limit, no eviction), so local DRAM hit rates rise
monotonically and `local_dram_write` accumulates.

Because replication changes which dies hold an expert,
`_gen_exp_chiplet_distribution(layer)` — which the allocator calls to find
"local" candidate dies — sees a growing set as the run proceeds. Prediction and
allocation compound.

---

## 6. Allocation strategies

Allocation answers: *for this layer, which die executes which expert's requests?*
It runs once per `(iteration, layer)`, before any event is created, and returns

```
exp_allo_plan = [[expert_id, req_num, chiplet_id], ...]
```

An expert may appear multiple times with different dies — its requests are split.

| `AllocationStrategy` | Rule |
|---|---|
| `NEAREST` | Home die. Classic expert parallelism; zero remote weight traffic, load follows routing skew |
| `EXP_EVEN` | `(expert_index + 7) % chiplet_num` — round-robin over experts in dict order |
| `EVEN` | `allocated_req_num % chiplet_num` — round-robin over the running request total |
| `RANDOM` | `expert_id % chiplet_num` |
| `OURS` | Greedy, overlap-aware, load-balanced (below) |
| `OURS_ORG` | Earlier version of the same idea, kept for comparison (`_allocate_experts_org`) |

Named strategies in `experiment_config.STRATEGY_MAP`:

| Name | Execution | Allocation |
|---|---|---|
| `base` | `BASELINE_CACHE_LOCAL` | `EXP_EVEN` |
| `norm_ep` | `BASELINE_CACHE_LOCAL` | `NEAREST` |
| `allo_only` | `BASELINE_CACHE_LOCAL` | `OURS` |
| `pred_only` | `PRED_NEXT_TOKEN` | `EXP_EVEN` |
| `allo_and_pred` | `PRED_NEXT_TOKEN` | `OURS` |
| `allo_only_org` | `BASELINE_CACHE_LOCAL` | `OURS_ORG` (not in the default sweep) |

### `OURS` — `_allocate_experts_ours`

**Die state.** Six per-die accumulators, in nanoseconds:

```
compute_local    compute for experts whose weights are already on this die
compute_remote   compute for experts whose weights must arrive over D2D
dram_local       DRAM reads feeding local compute
dram_remote      DRAM reads feeding weights shipped to other dies
d2d_in           link time receiving weights
d2d_out          link time sending weights
```

**Estimated die time** (`_die_estimated_time`) folds those into one number,
modelling what overlaps and what serialises:

```
die_time = max( max(compute_local, dram_local + dram_remote), d2d_in ) + compute_remote
```

Local compute overlaps with DRAM reads (different hardware); the two DRAM
streams share one controller and serialise; remote compute cannot start until
its weights have arrived; the single compute unit runs local then remote work
back to back. `d2d_out` is accumulated but does **not** enter the formula —
the `memory_path` term in the docstring is commented out in the code.

**The planner's cost model is coarser than the executed one.** It works on
whole experts, not slices, with a flat 0.7 compute utilisation and no per-model
tables:

```
_compute_lat(batch)   = ceil(2 * hidden * expert_dim * batch * 3 / (1e6 * 0.7))
_dram_read_lat()      = 300 + expert_size / (3350 * 0.9)
_d2d_transfer_lat(h)  = 2 * h * 200 + expert_size / (1741 * 0.9)
```

**Algorithm.**

1. Pre-charge `dram_local[home] += _dram_read_lat()` for every active expert —
   the assumption that each live expert's weights get read once at home.
2. Sort experts **ascending by request count**, so the hottest experts are
   placed last, against the most complete picture of load.
3. For each expert:
   - *Local candidates*: every die that already holds this expert for this layer.
   - *Remote candidates*: dies within Manhattan distance 1 of the home die,
     costed with a `_d2d_transfer_lat(hops)` penalty.
   - Rank all candidates by estimated die time, keep the best
     `clamp(max(5, req_num // max(mean_count, 512)), <= len(candidates))`.
   - Greedily assign requests in **512-request blocks**: for each block, trial
     every candidate, score it as the max estimated time over the two dies the
     block touches (executing die and data-source die), take the minimum. Ties
     go to a local die.
4. Optionally run `_rebalance_pass` (off by default): rebuild state from the
   plan, find the bottleneck die, and try migrating one of its entries to any
   die within `rebalance_distance` (default 2) that lowers the global max.
   Repeat up to `rebalance_max_rounds` (default 3).

Two vestiges in this function: `remote_helps` is computed and never read, and
`allow_remote` is computed and then unconditionally overwritten with `True`.

---

## 7. Metrics and outputs

### `results/<model>_results.csv`

20 columns, one row per `(strategy, arch, batch)`:

| # | Column | Meaning |
|---:|---|---|
| 1–5 | `model, strategy, y_chips, x_chips, batch` | configuration |
| 6 | `usage` | mean die utilisation: `sum(die busy spans) / (max span * n_dies)`, averaged over layers |
| 7 | `die_max_over_min` | per-layer slowest/fastest die span, averaged |
| 8 | `die_max_over_avg` | per-layer slowest/mean die span, averaged |
| 9 | `allo_exp_skew` | mean/max experts assigned per die — **1.0 = perfectly even** |
| 10 | `allo_req_skew` | mean/max requests assigned per die — **1.0 = perfectly even** |
| 11 | `dram_access_skew` | mean/max DRAM bytes per die — **1.0 = perfectly even** |
| 12 | `dram` | total resident weight footprint across all dies (GB) |
| 13–15 | `local_dram_read`, `remote_dram_read`, `local_dram_write` | traffic (GB) |
| 16 | `hop` | total link traversals, **counted on request legs only** (`_send_req_to_remote_*`), not on data returns |
| 17–19 | `time`, `prefill`, `decode` | ms; `prefill` is the clock at the prefill→decode transition |
| 20 | `throughput` | `batch * 128 / decode_ms * 1000` tokens/s |

Columns 9–11 are named `*_skew` but are **avg/max ratios in (0, 1] where higher
is more even**. Reading them as skew inverts the conclusion.

Die spans in columns 6–8 only count dies that did work in that layer; an idle
die is absent from the statistic rather than contributing a zero.

### `results/expert_analysis/`

`export_expert_analysis` writes two CSVs per configuration, stemmed
`<model>_<strategy>_<y>x<x>_batch<b>`:

* `..._expert_usage.csv` — `stage, layer, expert_id, selection_count, selection_share, home_chiplet`.
  How often the router picked each expert. Independent of allocation.
* `..._expert_allocation.csv` — `stage, layer, expert_id, executing_chiplet, request_count, request_share_for_expert, home_chiplet, is_remote`.
  Where those requests actually ran, and whether that meant crossing a die boundary.

### Figures

`draw_e2e_ae.process_ae_results` reads `results/*_results.csv` and writes to
`figures_ae/`:

* `e2e_stacked.{png,pdf,svg}` — 4 rows x 4 cols. Rows are
  (throughput, 5x5), (throughput, 8x3), (hop, 5x5), (hop, 8x3); columns are the
  four models; each subplot has three batch groups x five strategy bars.
  Throughput bars are speedup vs `base`; hop bars are `base_hop / strategy_hop`
  (reduction). Missing models are left blank rather than crashing.
* `available_results.{png,pdf}` — same data restricted to what actually ran.
* `e2e_summary.csv` — mean speedup and hop reduction per strategy, grouped
  overall / by batch / by shape / by model.

`plotting.plot_decode_time_vs_batch` is only invoked under `GEMM_VERIFY_MODE`
or `P2P_VERIFY_MODE`; it overlays simulated decode time against measured GEMM
numbers from `verify_ref/moe_bench_results.csv` (not present in this checkout).

---

## 8. Running it

### Environment

```bash
bash setup_env.sh                 # creates .venv, installs numpy/matplotlib/pandas/huggingface_hub
source .venv/bin/activate
set -a; source .env; set +a       # exports MOE_TRACE_ROOT
echo "$MOE_TRACE_ROOT"            # must print /home/gokul/moe_traces
```

### The sweep

```bash
python main_ae.py --models qwen --batches 4096 --skip-download
python main_ae.py --models all --archs "8,3;5,5" --skip-download
python main_ae.py --skip-download --skip-sim          # re-plot existing results/
python main_ae.py --skip-download --skip-sim --skip-plot   # path check only
```

`main_ae.py` monkey-patches `experiment_config` and `main` at import time with
the CLI values, then calls `main.run_benchmark_suite()`.

### Constraints on this machine

* **Never launch a full run from a tool call or the VS Code terminal.** A sweep
  takes 8–12 hours per model and holds tens of GB. Use tmux, as in `runfile.txt`.
* **Always pass `--skip-download`.** Without it, `snapshot_download` re-fetches
  72 GB of small files into `./moe_profiling_trace/` and freezes the editor's
  file watcher and search index.
* **`main_ae.py` destroys `results_bp/` on startup**, replacing it with the
  current `results/`. Copy anything you want to keep elsewhere first.
* 64 GiB RAM: batch 4096 is comfortable, 16384 needs ~50 GB with the editor
  closed. `--parallel` needs ~200 GB and is not viable here.

The memory comes from `main.py`'s trace cache: `trace_cache` is rebuilt **per
strategy** and grows to `max(BATCH_SIZES)` parsed JSON traces, all held live.
The architecture loop is innermost, so both mesh shapes share one load.

### Recommended invocation

```bash
tmux new -s sim
cd "<repo>" && source .venv/bin/activate && set -a; source .env; set +a
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
  python3 -u main_ae.py --models qwen --batches 4096 --skip-download 2>&1 | tee ~/ae_qwen.log
# Ctrl+b, d to detach
```

### Library use

```python
from toy_chiplet_sim import SystemConfig, Simulator

config = SystemConfig(
    y_chiplets=8, x_chiplets=3,
    allocation_strategy=SystemConfig.AllocationStrategy.OURS,
    exe_strategy=SystemConfig.ExeStrategy.PRED_NEXT_TOKEN,
    model_name="qwen",
)
sim = Simulator(config)
sim.process_trace(list_of_trace_paths)   # or a pre-loaded list of trace dicts
sim.process_records()                    # must be called before reading metrics

print(sim.current_time / 1e6, "ms")
print(sim.stats["hop"], "hops")
print(sim.avg_chiplet_usage)
```

`process_records()` is what computes `avg_chiplet_usage`, the duration ratios,
the allocation-evenness ratios and `dram_access_avg_over_max`. Reading those
attributes before calling it gives zeros.

---

## 9. Behaviours worth knowing before trusting a number

**Modelling scope**

1. Attention, KV cache, all-to-all dispatch/combine, and shared experts are not
   modelled. Only MoE expert weight movement and expert GEMM time.
2. The per-layer global barrier means no overlap between layers or tokens. Real
   systems pipeline; this is a pessimistic serialisation.
3. Iteration count is fixed at 128 regardless of trace length, and throughput
   uses `batch * 128 / decode_time` even though only 127 iterations are decode.
4. DRAM has no capacity limit and no eviction, so `PRED_NEXT_TOKEN` replication
   grows without bound over a run.
5. `hop` counts request legs only. The data return leg traverses the same links
   and does cost time, but is not in the counter — so `hop` is a proxy for
   remote-access *frequency x distance*, not total link traversals.

**Divergences between the plan and the execution**

6. The allocator's cost model (whole experts, flat 0.7 utilisation, distance-1
   candidates) is deliberately coarser than the event model (per-slice, per-model
   utilisation tables, real queueing). A plan that looks balanced can still
   execute unevenly.
7. `d2d_out` is tracked in the die state but excluded from `_die_estimated_time`,
   so a die that is shipping weights to many neighbours is not penalised for the
   outbound link time.

**Dead or disabled code**

8. `ExeStrategy.PRED_NEXT_LAYER` is a stub (`_pred_next_layer` returns `False`).
9. `InterconnectType.RING` has a router but is unreachable; `CROSSBAR` is unimplemented.
10. `_try_swap_into_event_queue` is implemented but never called.
11. DRAM LRU/hardware-managed-region eviction is commented out in `hardware/dram.py`.
12. `Chiplet.req_num` and `Chiplet.comm_stats` are initialised and never updated;
    `main.py`'s standalone `main()` prints `req_num` and will always show 0.
13. `SystemConfig.routing_table` is an unused empty dict.
14. `main.main()` and `test_dataset.py` reference `/workspace/zhongkai/...`
    paths from the original authors' machine and will not run here.
15. `should_skip_configuration` guards `NEAREST + PRED_NEXT_TOKEN`, a combination
    no entry in `STRATEGY_MAP` produces.

**Cache realism**

16. With a 64 MiB LLC and 12–80 MB slices, `num_sets` is 1 or 2 and the cache
    holds 2–4 slices. Cache hits are rare by construction; the interesting
    locality effect in this simulator is DRAM-level replication under
    `PRED_NEXT_TOKEN`, not the LLC.

**Debug switches** (module-level, top of `simulation/simulator.py`)

```
DEBUG, PRINT_DEBUG      verbose event tracing
DEBUG_ALLO              per-expert allocation decisions
SUM_TABLE_EN            per-layer allocation summary table
ENABLE_PROFILING        wall-clock breakdown via _print_profiling()
SECURITY_CHECK          assert the trace ran to completion
LIMIT_MAX_ITER_NUM      >0 truncates the run to that many iterations (fast smoke test)
```
