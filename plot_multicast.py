#!/usr/bin/env python3
"""
Baseline (XY unicast) vs multicast tree delivery.

Reads the comm_analysis summaries both runs produce and draws the four numbers
the project turns on. Works with whatever models are present, so it can be run
part-way through a sweep.

    python plot_multicast.py
    python plot_multicast.py --models qwen deepseek
    python plot_multicast.py --baseline-dir ~/b --mcast-dir ~/m --out figures_mcast
"""
import argparse, csv, os, sys
from pathlib import Path

MODELS = ["qwen", "llama4", "deepseek"]
# default baseline locations (qwen's differs for historical reasons)
BASE_DEFAULT = {
    "qwen":     "~/Downloads/sys for ml/project/results/Baseline_XY_unicast_QWEN",
    "llama4":   "~/Downloads/sys for ml/project/results/Baseline_XY_unicast_Llama",
    "deepseek": "~/baseline_deepseek_results",
}
MCAST_DEFAULT = "~/mcast_results/{model}"

# validated categorical palette (dataviz reference instance, light surface)
C_BASE, C_MCAST = "#2a78d6", "#eb6834"
C_MODEL = {"qwen": "#2a78d6", "llama4": "#eb6834", "deepseek": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dddcd6", "#fcfcfb"

# (summary key, axis label, unit scale, unit)
# sim_hop_counter is the one crossing metric that is correct in BOTH modes:
# unicast counts request-leg traversals (= data-leg traversals by symmetry),
# multicast counts tree links. total_link_crossings was broken under multicast
# in runs produced before 2026-10-06 -- do not use it for comparison.
METRICS = [
    ("sim_hop_counter",      "Link crossings",    1e6,  "M",  "Link\ncrossings"),
    ("__link_tb",            "Link traffic",      1,    "TB", "Link\ntraffic"),
    ("die_egress_peak_GB",   "Peak die egress",   1024, "TB", "Peak die\negress"),
    ("payload_from_dram_GB", "Source DRAM reads", 1024, "TB", "Source DRAM\nreads"),
]
SLICE_MB = {"qwen": 12, "llama4": 80, "deepseek": 28}


def read_summary(d, model):
    """Find this model's summary under `d`. Returns (dict, path) so the caller
    can show which file was used -- a mislabelled folder is otherwise invisible."""
    d = Path(os.path.expanduser(str(d)))
    hits = sorted(d.glob(f"**/{model}_token_par_*_comm_summary.csv"))
    if not hits:
        return None, None
    if len(hits) > 1:
        print(f"    note: {len(hits)} matches for {model} under {d}, using {hits[0]}")
    with open(hits[0], newline="", encoding="utf-8") as f:
        return {r["metric"]: r["value"] for r in csv.DictReader(f)}, hits[0]


def fnum(d, key, model=None):
    """Read one metric. No cross-field aliasing: silently substituting a
    differently-defined field is how a 3.2x win got reported as 0.88x."""
    if d is None:
        return None
    if key == "__link_tb":                       # crossings x slice size
        x = fnum(d, "sim_hop_counter")
        mb = SLICE_MB.get(model)
        return (x * mb / 1024 / 1024) if (x and mb) else None
    if key in d:
        try:
            return float(d[key])
        except ValueError:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--baseline-dir", default=None,
                    help="one dir holding all baselines (else per-model defaults)")
    ap.add_argument("--mcast-dir", default=None)
    ap.add_argument("--out", default="figures_mcast")
    a = ap.parse_args()

    data = {}
    for m in a.models:
        b, bp = read_summary(a.baseline_dir or BASE_DEFAULT.get(m, ""), m)
        c, cp = read_summary(a.mcast_dir or MCAST_DEFAULT.format(model=m), m)
        if b and c:
            data[m] = (b, c)
            print(f"  {m:>9}: baseline {bp}")
            print(f"  {'':>9}  mcast    {cp}")
        else:
            print(f"  {m:>9}: skipped ({'no baseline' if not b else 'no multicast run yet'})")
    if not data:
        print("\nNothing to plot. Run the baseline and multicast sweeps first.", file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    models = list(data)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    fig.subplots_adjust(hspace=0.42, wspace=0.26)

    # ---- (0,0) the headline: reduction factor -----------------------------
    ax = axes[0][0]
    w = 0.8 / len(models)
    xs = np.arange(len(METRICS))
    for i, m in enumerate(models):
        b, c = data[m]
        vals = []
        for key, _lbl, _sc, _u, _short in METRICS:
            bv, cv = fnum(b, key, m), fnum(c, key, m)
            vals.append(bv / cv if (bv and cv) else 0.0)
        pos = xs - 0.4 + w * (i + 0.5)
        ax.bar(pos, vals, w * 0.86, color=C_MODEL.get(m, C_BASE), label=m, zorder=3)
        for p, v in zip(pos, vals):
            if v:
                ax.text(p, v, f"{v:.1f}×", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    ax.axhline(1.0, color=INK2, lw=1, ls="--", zorder=2)
    ax.text(-0.46, 1.0, "no change", va="bottom", ha="left", fontsize=7, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[4] for m in METRICS], fontsize=8)
    ax.set_ylabel("baseline ÷ multicast  (higher is better)")
    ax.set_title("Reduction from multicast tree delivery", fontsize=10.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, ncol=len(models))

    # ---- (0,1) and (1,0): absolutes for the two headline metrics ----------
    for ax, (key, label, scale, unit, _s) in zip((axes[0][1], axes[1][0]), METRICS[:1] + METRICS[2:3]):
        xs2 = np.arange(len(models))
        bv = [(fnum(data[m][0], key, m) or 0) / scale for m in models]
        cv = [(fnum(data[m][1], key, m) or 0) / scale for m in models]
        ax.bar(xs2 - 0.21, bv, 0.40, color=C_BASE, label="XY unicast", zorder=3)
        ax.bar(xs2 + 0.21, cv, 0.40, color=C_MCAST, label="multicast", zorder=3)
        for x, v in zip(xs2 - 0.21, bv):
            ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        for x, v in zip(xs2 + 0.21, cv):
            ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(xs2); ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(label, fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.legend(frameon=False, fontsize=8)

    # ---- (1,1) why: the fanout distribution -------------------------------
    ax = axes[1][1]
    drew = False
    for m in models:
        h = data[m][1].get("fanout_hist") or data[m][0].get("fanout_hist")
        if not h:
            continue
        pairs = [p.split(":") for p in h.split()]
        k = np.array([int(a) for a, _ in pairs]); v = np.array([float(b) for _, b in pairs])
        ax.plot(k, 100 * v / v.sum(), lw=2, color=C_MODEL.get(m, C_BASE),
                marker="o", ms=3.5, label=m)
        drew = True
    if drew:
        ax.set_xlabel("dies requesting the same slice (fanout)")
        ax.set_ylabel("% of slice broadcasts")
        ax.set_title("Why: multicast saving tracks fanout", fontsize=10.5, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "fanout histogram not in these summaries\n"
                          "(re-run with the current code to record it)",
                ha="center", va="center", fontsize=8.5, color=INK2)

    fig.suptitle("Expert-weight delivery: XY unicast vs multicast tree  ·  8×3 mesh, batch 4096",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.985)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"multicast_vs_baseline.{ext}", dpi=170, bbox_inches="tight")
    print(f"\nwrote {out}/multicast_vs_baseline.png (+ .pdf)")

    # the same numbers as text, for the writeup
    print(f"\n{'model':>10} {'metric':>22} {'baseline':>14} {'multicast':>14} {'ratio':>8}")
    for m in models:
        for key, label, scale, unit, _s in METRICS:
            bv, cv = fnum(data[m][0], key, m), fnum(data[m][1], key, m)
            if bv and cv:
                print(f"{m:>10} {label:>22} {bv/scale:>13,.1f}{unit:<1} "
                      f"{cv/scale:>13,.1f}{unit:<1} {bv/cv:>7.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
