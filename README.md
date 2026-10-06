# Toy Chiplet Simulator for MoE

Multi-chiplet GPU simulator for **MoE (Mixture-of-Experts)** model execution. Simulates cycle-accurate inference on a 2D mesh of chiplets with configurable expert placement, allocation strategies, and cache/execution policies.

## Features

- **Topology**: 2D mesh (or ring) interconnect; configurable `y_chiplets` x `x_chiplets`.
- **Models**: Predefined configs for DeepSeek, Qwen, Kimi, and Llama-4 (layer count, experts, hidden dims, etc.).
- **Allocation**: NEAREST, EVEN, EXP_EVEN, RANDOM, OURS (load-balanced).
- **Execution**: Baseline cache-local / cache-all, or prediction-based (next-token / next-layer).
- **Trace-driven**: JSON trace files (per-token expert selection) drive the simulation.

## Layout

```
toy_chiplet_sim/
├── main.py              # Entry: run single trace or batch benchmark
├── main_ae.py           # One-click Artifact Evaluation script
├── draw_e2e_ae.py       # AE plotting (handles partial data)
├── data_loader.py       # Trace file loading utilities
├── experiment_config.py # Experiment configuration
├── simulator_runner.py  # Simulation runner helpers
├── test_dataset.py      # Dataset / trace list helpers
├── ep_balancer.py       # Optional: expert load balancing (torch)
├── toy_chiplet_sim/     # Main package
│   ├── config.py        # SystemConfig, TimingConfig
│   ├── models/          # Addresses, Expert, Expert_slice
│   ├── hardware/        # DRAM, Cache, Chiplet, Interconnect
│   └── simulation/      # Events, ResourceManager, Simulator
├── results/             # Simulation outputs (gitignored)
├── figures_ae/          # AE-generated figures
├── moe_profiling_trace/ # Downloaded trace files (auto-created by AE)
├── requirements.txt
└── README.md
```

---

## Artifact Evaluation (AE)

### Dependencies

#### Hardware
- **CPU**: Any modern x86-64 processor (simulation is single-threaded, no GPU required)
- **RAM**: >= 64 GB for single model (batch size 16384 caches ~50 GB of trace data in memory); >= 200 GB if running all 4 models with `--parallel`
- **Disk**: >= 80 GB free space for 1 model; >= 300 GB for all 4 models (trace files are large)

#### Software
- **OS**: Linux (tested on Ubuntu 22.04)
- **Python**: >= 3.10

| Python Package   | Min Version | Purpose                              |
|------------------|-------------|--------------------------------------|
| numpy            | >= 1.20     | Core simulation computation          |
| matplotlib       | >= 3.5      | Figure generation                    |
| pandas           | >= 1.3      | CSV result processing                |
| huggingface_hub  | latest      | Trace download (auto-installed only if traces not present locally) |

> All Python dependencies are **auto-installed** by `main_ae.py` if missing.

#### Data
- **MoE expert selection traces**: Profiled token-level expert routing decisions on MMLU / MMLU_ZH_CN benchmarks.
- **Source**: [HuggingFace Dataset](https://huggingface.co/datasets/core12345/MoE_expert_selection_trace)
- **Models covered**:
  | Model | Parameters | HuggingFace Subdirectory |
  |-------|-----------|--------------------------|
  | DeepSeek-R1-AWQ | 671B (4-bit) | `cognitivecomputations/DeepSeek-R1-AWQ` |
  | Qwen3-235B-A22B-FP8 | 235B (FP8) | `Qwen/Qwen3-235B-A22B-FP8` |
  | Kimi-K2-Thinking | 1T | `moonshotai/Kimi-K2-Thinking` |
  | Llama-4-Maverick | 17B (128E) | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |
- **Format**: JSON files, one per query, containing per-layer expert selections.
- The AE script **auto-downloads** required traces if not found locally.

### Quick Start (Recommended)

Run the default AE configuration (Qwen model, 8x3 chiplet, batch 4096/8192/16384):

```bash
pip install -r requirements.txt
python main_ae.py
```

This will:
1. Check for trace files (download from HuggingFace if missing)
2. Backup existing `results/` to `results_bp/` and run simulations
3. Generate figures in `figures_ae/`

**Estimated time**: 8-12 hours for the default configuration (1 model, 1 architecture, 3 batch sizes, 5 strategies).

### Full Evaluation

Run all 4 models on both chiplet architectures:

```bash
python main_ae.py --models all --archs "8,3;5,5"
```

Run all 4 models **in parallel** (one process per model, significantly faster):

```bash
python main_ae.py --models all --archs "8,3;5,5" --parallel
```

Per-model logs are saved to `ae_{model}.log`. All models finish before plotting.

**Estimated time**: ~96 hours sequentially; 18-36 hours with `--parallel` (4 models x 2 architectures x 3 batch sizes x 5 strategies = 120 configurations).

### Custom Configurations

```bash
# Specific models
python main_ae.py --models qwen,deepseek

# Specific architecture
python main_ae.py --archs "5,5"

# Both architectures
python main_ae.py --archs "8,3;5,5"

# Custom batch sizes
python main_ae.py --batches 4096,8192

# Parallel execution (one process per model)
python main_ae.py --models all --parallel

# Skip steps (useful for re-plotting existing results)
python main_ae.py --skip-download --skip-sim     # only re-generate figures
python main_ae.py --skip-plot                     # only run simulation
```

### AE Output

- **`results/`**: Per-model CSV files (`qwen_results.csv`, etc.) with 20 columns including throughput, hop count, DRAM access stats, and load balance metrics.
- **`figures_ae/e2e_stacked.png`**: Stacked bar chart (4 rows x 4 cols) showing throughput speedup and hop reduction across models, batch sizes, and die shapes. Missing models are left blank.
- **`figures_ae/e2e_summary.csv`**: Aggregate statistics grouped by model, batch size, and die shape.

### Expected Results

The AE reproduces the end-to-end simulation results from the paper. Key claims:
- The proposed allocation strategy (Allo Only / Allo+Pred) achieves higher throughput and lower inter-chiplet hops compared to baseline and standard EP placement.
- The improvement is consistent across different MoE models, batch sizes, and chiplet topologies.

---

## Developer Usage

### Install / Run

```bash
pip install -r requirements.txt
python main.py              # run with current experiment_config.py settings
python main.py qwen         # run single model
```

### API

```python
from toy_chiplet_sim import SystemConfig, TimingConfig, Simulator

config = SystemConfig(
    y_chiplets=3,
    x_chiplets=3,
    allocation_strategy=SystemConfig.AllocationStrategy.NEAREST,
    exe_strategy=SystemConfig.ExeStrategy.PRED_NEXT_TOKEN,
    model_name="deepseek",
)
sim = Simulator(config)
sim.process_trace(["path/to/trace0.json", "path/to/trace1.json"])
sim.process_records()

print("Total time (ms):", sim.current_time / 1e6)
print("Local DRAM hit rate:", sim.stats["local_dram_hit"] / sim.stats["total_access"])
print("Chiplet usage:", sim.avg_chiplet_usage)
```

### Trace Format

JSON: list of iterations; each iteration is a dict keyed by layer id (`"0"`, `"1"`, ...). Value is either `null` (no MoE) or a 2D list of selected expert ids per token (prefill: multiple tokens; decode: one token per list).

## Phase 1 — Token-Parallel Placement (`token_par`)

Added to enable research on **multicast delivery of expert weights**. This section
documents what changed and why.

### Why it was needed

The simulator has no per-die token ownership. `_trace_reader_core` merges every
token in the batch into one global counter per `(iteration, layer)`, and the
allocator then hands each expert to a **single** executing die. Measured fanout —
the number of dies requesting the same expert slice — was therefore exactly 1:

| batch | mean req/expert | mean fanout | max fanout |
|---|---:|---:|---:|
| 4096 | 256 | 1.00 | 1 |
| 8192 | 512 | 1.00 | 1 |
| 16384 | 1024 | 1.57 | 2 |

With fanout 1 every transfer is point-to-point, so there is nothing to multicast.
The fix models the architecture the multicast work assumes: **tokens stay on their
home die, and each die runs the union of experts its own tokens selected.** Because
a die's token population covers nearly the whole expert set, one expert must then
reach nearly every die — a broadcast.

### What changed

Three additive changes. `TOKEN_PARALLEL` is a **new** allocation strategy rather
than a replacement, so `base`, `norm_ep`, `allo_only`, `pred_only` and
`allo_and_pred` all behave exactly as before.

| File | Change | Why |
|---|---|---|
| `toy_chiplet_sim/config.py` | `AllocationStrategy.TOKEN_PARALLEL = 7` | New strategy; leaves existing enum values untouched so saved results stay comparable |
| `toy_chiplet_sim/simulation/simulator.py` — `_trace_reader_core` | Builds `self.die_expert_counts`, an `(iter, layer, die, expert)` int32 array | The global counter cannot say *which die* wants an expert. Requests are split into contiguous blocks, one per die, so die `d` owns trace files `[d·B/N, (d+1)·B/N)` |
| `toy_chiplet_sim/simulation/simulator.py` — `_allocate_experts_token_parallel` | Emits one plan entry per `(expert, die)` with a non-zero count | Makes no placement *decision* — reads the plan straight off per-die demand. This is what produces fanout |
| `experiment_config.py` | Registers `"token_par"` | Makes it selectable via `--strategies token_par` |

Two implementation notes:

- The per-die array is **only allocated when `TOKEN_PARALLEL` is active**, so other
  strategies pay no memory or time cost.
- Counts are accumulated with **one batched `np.add.at` per trace** rather than one
  call per token row. The naive form would be ~50M numpy calls at batch 4096; the
  batched form takes ~1.4 minutes.

### Verification

Fanout after the change, on an 8×3 mesh (24 dies), qwen:

| tokens/die | mean fanout | max | experts per die |
|---:|---:|---:|---:|
| 10 | 14.4 | 24 | 76.9 / 128 |
| 50 | 23.3 | 24 | 124.4 / 128 |
| 170 (≈ batch 4096) | **24.0** | 24 | **128.0 / 128** |

Correctness: per-die token counts reconcile exactly with the global counter at
every `(iteration, layer)`, and `base` / `norm_ep` produce unchanged results.

End-to-end on a reduced synthetic run, showing the unicast cost this creates:

```
      strategy   time ms        hop  remote GB   usage  fanout
          base      15.5      21618       69.2    0.86     1.0
       norm_ep       1.4          0        0.0    0.89     1.0
     token_par     275.5     409194     1303.6    0.98    18.9
```

The ~19× traffic increase over `base` is expected and is the point: it is the
*unicast broadcast* that multicast tree delivery is intended to compress.

### Usage

```bash
python main_ae.py --models qwen --archs "8,3" --batches 4096 \
  --strategies token_par --skip-download --skip-plot
```

Or directly:

```python
config = SystemConfig(
    y_chiplets=8, x_chiplets=3,
    allocation_strategy=SystemConfig.AllocationStrategy.TOKEN_PARALLEL,
    exe_strategy=SystemConfig.ExeStrategy.BASELINE_CACHE_LOCAL,
    model_name="qwen",
)
```

### Costs and caveats

- **Memory**: `die_expert_counts` is a fixed 148 MB for qwen on 24 dies
  (`128 × 94 × 24 × 128` int32), 192 MB for deepseek. Independent of batch size.
- **Time**: ~1.4 min of extra aggregation at batch 4096.
- **Runtime**: expect `token_par` runs to be far slower than `base` — roughly 19×
  the link traffic, all of it serialising on the same links.
- **Coverage figures above are from synthetic traces with uniform expert
  selection**, the most favourable case. Real traces have popularity skew, so cold
  experts will be missed on some dies and per-die coverage will be below 128/128.
  Re-measure on real traces before quoting these numbers.

## Phase 2 — D2D Instrumentation and Bounded Replication

Two additions needed before multicast can be evaluated: the simulator must
**measure** link and egress traffic, and replication must be **capacity-bounded**
so it is a fair comparison rather than a free win.

### Why it was needed

**Nothing about communication was retained.** The event loop computes the route,
message size and link occupancy for every remote slice access, then discards all
of it except one scalar, `stats['hop']` — which counts only the *request* legs,
not the payload returns that carry the bytes. Peak per-die egress, the bottleneck
the multicast work targets, was not measurable at all.

**Replication was free and unbounded.** `dram_size_per_die` was recorded but never
checked, and the LRU eviction in `dram.py` was commented out. So `PRED_NEXT_TOKEN`
could replicate experts into every die's DRAM without limit — which makes any
"multicast vs. replication" comparison meaningless, because replication wins
trivially when its cost is not modelled.

### What changed

| File | Change | Why |
|---|---|---|
| `toy_chiplet_sim/hardware/dram.py` | Replica region is capacity-bounded with LRU eviction; `deque` replaced by `OrderedDict` | The eviction logic existed but was commented out **and** used `deque.remove()`, which is O(n) per store — ~51 billion operations over a full run. `OrderedDict.move_to_end` is O(1) |
| `toy_chiplet_sim/simulation/comm_recorder.py` | **New file.** Per-link, per-die, per-pair and per-epoch D2D accumulators plus CSV export | Everything the multicast study measures. Per-die **egress** is first-class because that is the bottleneck no routing policy can relieve |
| `toy_chiplet_sim/simulation/simulator.py` | `COMM_TRACE_ENABLED` switch, recorder construction, `begin_epoch` call, `_record_comm`, `export_comm_analysis`, and `issue_ns` threaded through four callbacks | Hooks the recorder into the only four sites that put a message on a link |
| `main.py` | `EXPORT_COMM_ANALYSIS` flag and export call | Writes `results/comm_analysis/` alongside the main results CSV |

**Two design points worth knowing:**

*Record in the callback, not at event creation.* `ResourceManager.request_resource`
pushes `event.time` forward when links are busy, so the callback fires
*post-contention*. Recording there means `current_time - issue_ns` is the
queueing delay, for free. That is why four callback signatures gained an
`issue_ns` parameter.

*Home data is never evicted.* The bounded region covers replicas only
(`is_hm_data=True`, written by `_store_in_lcoal_dram`). Slices a die owns under
the static placement map are written with `is_hm_data=False` and stay resident.

### Verification

Four reconciliation checks, both on `base` and on `token_par`:

| Check | Meaning |
|---|---|
| `Σ(req_count × hops) == stats['hop']` | No link site missed or double-counted |
| `recorder DRAM payload == remote_dram_read_size` | Byte accounting agrees with the simulator |
| `Σ per-die egress == payload` | Egress attribution is complete |
| `Σ per-die ingress == payload` | Ingress attribution is complete |

All four pass. Home-slice eviction count is 0.

The second check initially failed on `token_par` by 3.13 GB and exposed a real
distinction: a **remote LLC hit still crosses the network but is not a DRAM
read**. With fanout ~23, the first fetch populates the supplier's LLC and later
requests in the same layer hit it — the cache performs a small amount of
*accidental multicast*. It is now reported separately as
`payload_from_remote_llc_GB` (0.33% of payload, consistent with an LLC that holds
only 2–4 slices). Credit it to the baseline rather than to multicast.

### Output

`results/comm_analysis/<model>_<strategy>_<shape>_batch<N>_comm_*.csv`

| File | One row per | Carries |
|---|---|---|
| `_comm_dies.csv` | die | egress/ingress GB, peak per-epoch egress, requesters served, replica GB, evictions |
| `_comm_links.csv` | mesh edge | GB, busy_ns, utilisation, queueing delay |
| `_comm_pairs.csv` | (requester, supplier) | hops, transfers, GB, wait |
| `_comm_timeline.csv` | (iteration, layer) | bytes, hops, wait, peak die egress in that epoch |
| `_comm_summary.csv` | metric | roll-up incl. `die_egress_peak_GB`, `link_util_max_over_mean`, `total_link_crossings` |

Column-naming caution: `payload_GB` excludes the 16-byte request messages;
`link_traffic_GB` includes them. And `mean_requesters_per_supplier` counts
distinct requester dies per supplier die — it is **not** expert fanout (dies per
expert), which is a per-expert quantity measured from the allocation plan.

### Costs and caveats

- **Recorder overhead**: under 5% of event-loop time, measured on `token_par`
  (40.6/42.5 s disabled vs 4tmux new -s baseline2.6/41.4 s enabled — within run-to-run noise). Flat
  Python lists and a per-die-pair route cache are what keep it cheap; naive numpy
  scalar indexing costs ~40%. Set `COMM_TRACE_ENABLED = False` to disable.
- **The default replica budget does not bind.** `hardware_manage_dram_size_per_die`
  is 50 GB, and `allo_and_pred` only adds ~15–20 GB per die, so enforcement
  produces **0 evictions** at the default. This is a knob to sweep, not a bug: the
  useful question is *how much replica capacity replication needs to match
  multicast's traffic reduction*, given multicast needs none.

## Phase 3 — Multicast Tree Delivery

Replaces N independent unicast deliveries of the same expert slice with **one
tree**: the supplier sends a single copy and routers fork it only where
destinations diverge.

### Why it works here

Under `token_par`, each die runs the union of experts its own tokens selected,
so nearly every die needs nearly every expert. Measured at batch 4096 on 8×3:

| model | mean fanout | fanout = 1 | fanout = 23 |
|---|---:|---:|---:|
| deepseek | 20.84 | 0.0% | 53.9% |
| qwen | 17.84 | 2.4% | 41.7% |
| llama4 | 9.42 | 10.0% | 3.7% |

Half of all slice broadcasts go to **all 23 remote dies**. Under unicast the
supplier sends that slice 23 times down largely overlapping routes and reads it
from DRAM 23 times.

### The tree needs no construction algorithm

The tree is simply the **union of the XY paths** from supplier to each
destination. XY routing is deterministic, so every die has exactly one path from
a given source; if two paths touch the same node they share the identical prefix,
so the union cannot contain a cycle. Verified: `|links| == |nodes| - 1` held on
**3000/3000** random cases on both meshes, and at full fanout the tree uses
exactly `n-1` links — i.e. it is **Steiner-optimal**, so there is no better tree
to find.

This is also what a destination-bitmap header produces when each router forwards
one copy per distinct output port, so the simulator models the result of the
bitmap scheme without simulating per-router bitmap logic.

### What changed

| File | Change | Why |
|---|---|---|
| `hardware/interconnect.py` | `multicast_tree(src, dsts)` → `(links, depth, first_hops, nodes)` | One call gives the resource list, the latency term, and the egress count |
| `simulation/simulator.py` | `_cal_tree_latency(depth, size)` | Unicast is `200·hops + size/bw`; a tree pays the hop term along its **longest branch** and serialises once |
| `simulation/simulator.py` | `_parse_tree_link_resources(links)` | Tree links reserve once each — works unchanged because links are keyed undirected |
| `simulation/simulator.py` | `_emit_multicast_delivery`, `_mcast_dram_read`, `_mcast_deliver` | The three-stage chain: one DRAM read → one tree transfer → N computations |
| `simulation/simulator.py` | grouping pass in `_schedule_next_request` | Groups plan entries by expert; groups of ≥2 go by tree, the rest fall through to the untouched unicast path |
| `simulation/simulator.py` | `fanout_hist` + `MEASURE_FANOUT` | Records the fanout distribution. Measurement only — runs even with multicast off |
| `simulation/comm_recorder.py` | `record_tree(...)` | Charges each tree link **once**, and supplier egress per **distinct first hop** rather than per destination |
| `plot_multicast.py` | **New.** Baseline vs multicast figure + table | Reads both runs' summaries; works part-way through a sweep |

**Event model.** One delivery satisfies many requests, but `pending_requests` is
still incremented once per requester, so the per-layer barrier is unchanged:

```
unicast  (per requester):  CACHE_ACCESS → NET_TRANS_REQ → MEMORY_ACCESS
                           → NET_TRANS_DATA → COMPUTATION → COMPLETION
multicast (per group):     MEMORY_ACCESS (once) → NET_TRANS_DATA (tree)
                           → N × COMPUTATION → N × COMPLETION
```

Multicast is **push-based**: the request leg is skipped entirely, matching the
"dies announce, supplier pushes" model. The announcement itself is a few hundred
bytes per die per layer and is not modelled — note it as overhead in any writeup.

### Usage

Off by default, so every existing baseline is unaffected:

```bash
MOE_MULTICAST=1 python main_ae.py --models qwen --archs "8,3" --batches 4096 \
  --strategies token_par --skip-download --skip-plot

python plot_multicast.py --baseline-dir <dir with baselines> --out <figure dir>
```

### Results (8×3, batch 4096, 128 iterations)

| model | link crossings | predicted | peak die egress | source DRAM reads | runtime |
|---|---:|---:|---:|---:|---:|
| deepseek | **3.42×** | 3.48× | 5.27× | 20.84× | 5.18× |
| qwen | **3.23×** | 3.26× | 5.83× | 17.80× | 4.64× |
| llama4 | **2.41×** | 2.36× | 3.66× | 9.42× | 2.78× |

Crossings land within 2% of a geometric prediction made independently from the
measured fanout — the strongest evidence the implementation is correct.

Three distinct magnitudes, one mechanism:

* **~3.4× link traffic** — the geometric result, robust.
* **~5× peak die egress** — the bottleneck no routing policy can relieve.
* **~20× source DRAM reads** — the second win: the supplier reads each slice
  **once** instead of once per requester, so DRAM pressure falls by the full
  fanout factor, not the tree factor.

### What got *worse* — report these too

| metric | qwen | llama4 | deepseek |
|---|---:|---:|---:|
| `link_util_max` | 0.75 → 0.95 | 0.72 → 0.86 | 0.73 → 0.99 |
| `link_util_mean` | +36.9% | +14.1% | +48.2% |
| `die_egress_max_over_mean` | +8.1% | +11.1% | +29.4% |

Utilisation rises because the **job finishes sooner**, not because more is sent:
runtime fell 4.64× while traffic fell 3.23×, so `busy ÷ runtime` rises by
4.64/3.23 = 1.44×. The real finding is that `link_util_max` reaches 0.95–0.99 —
**multicast moves the bottleneck onto the links**, which is the motivation for
demand-aware tree selection in Phase 4.

Egress balance worsens because a tree charges egress per distinct first hop, and
how many ports a tree needs depends on where the supplier sits in the mesh;
unicast averaged that out over many separate transfers. Absolute egress still
falls 3.7–5.8× on every die.

### Validation

| check | result |
|---|---|
| tree invariant `\|links\| == \|nodes\|-1` | PASS, 3000/3000 both meshes |
| allocated requests identical to unicast | PASS |
| allocated experts identical to unicast | PASS |
| trace runs to completion, 0 pending requests | PASS |
| fanout-1 groups fall back to unicast | PASS (excluded by construction) |
| byte accounting reconciles (tree + fallback) | PASS, to the GB |
| supplier-LLC path | conservative — see below |

**Three bugs this caught.** `_record_allocation` was called once per expert in
the multicast path but once per *slice* (3×) in the unicast path — work
conservation failed until fixed. `total_link_crossings` was computed as
`data_count × pair_hops`, which is meaningless for a tree, so it silently
reported the *unicast* crossing count; it is now an explicit counter. And the
plot script aliased a missing `link_bytes_GB` to `link_traffic_GB`, a differently
defined field, which turned a 3.2× win into a reported 0.88× loss.

> **Any multicast summary written before 2026-10-06 has a wrong
> `total_link_crossings`. Use `sim_hop_counter`, which is correct in both modes.**

**The LLC path is conservative, not flattering.** Multicast always sources from
DRAM and never populates the supplier's LLC, so it is charged a full DRAM read
(~4,474 ns) where unicast would occasionally have been served from cache at
100 ns. Magnitude: 0.045% of unicast payload. It costs multicast; it does not
help it.

### Scaling fixes needed to get here

`token_par` generates ~24× the event chains of `base`, which exposed two latent
problems that fanout-1 strategies never hit:

| File | Change | Why |
|---|---|---|
| `simulation/resource_manager.py`, `events.py` | Wait-queue entries removed **lazily** (`peek_waiter`) instead of an O(n) scan plus O(n) `heapify` per admission | Quadratic in contention. `token_par` was 7× slower *per event* than `base`; the fix is **10× faster** |
| `simulation/events.py` | Explicit FIFO tiebreak (`seq`) in `_ResQueueItem` | Ties between equal `(priority, time)` were previously broken by whatever `heapify` produced — undefined and non-reproducible. Now total and deterministic |
| `simulation/simulator.py` | `access_records` stores a running `[min_start, max_end]` instead of a tuple per completed chain | ~137M tuples (>15 GB) for deepseek, exceeding the memory cap. Now ~50 MB. `process_records` only ever read min/max, so results are **byte-identical** |
| `simulation/simulator.py` | Progress line triggers on the **iteration changing**, not `layer_id == 0` | deepseek layers 0–2 and llama4's dense layers never yield layer 0, so it could never fire — both runs looked hung |
| `simulation/simulator.py` | `MOE_MAX_ITERS`, `MOE_PROGRESS` env overrides | Truncated smoke runs without editing source |

The lazy wait-queue changes simulated **time** by −1% to −4.4% (`hop` is
unaffected) because it breaks ties differently from the old undefined order. All
three baselines were produced *after* this change, so they are mutually
comparable; results from before it are not.

## License

Use as needed for research or open-source projects.
# Demand-Aware-and-Multicast-routing-in-MoE-LLM-
