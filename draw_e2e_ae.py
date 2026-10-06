"""
Plotting script for AE (Artifact Evaluation).
Copied from paper_figs/draw_e2e.py with modifications to handle partial data
(missing models / die shapes are left blank instead of crashing).
Only produces the stacked figure (e2e_stacked) and summary CSV.
"""
import re
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'DengXian', 'Microsoft YaHei'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ── Config ──────────────────────────────────────────────────────────────
MODELS = ['deepseek', 'kimi', 'llama4', 'qwen']
MODEL_LABELS = ['DeepSeek-V3', 'Kimi-K2', 'Llama-4-Maverick', 'Qwen3-235B']

STRATEGIES = ['base', 'norm_ep', 'pred_only', 'allo_only', 'allo_and_pred']
STRATEGY_LABELS = ['Base(w/o proposed HW)', 'EP (w/ proposed HW)',
                   'Pred Only (w/ proposed HW)', 'Allo Only (w/ proposed HW)',
                   'Allo+Pred (w/ proposed HW)']

BATCHES = [4096, 8192, 16384]
SHAPES = [(5, 5), (8, 3)]
SHAPE_TITLES = ['5\u00d75 Wafer', '8\u00d73 Wafer']

TP_COLORS = ['#4A3728', '#8B5A3C', '#B27B56', '#D2B189', '#E7D7B6']
TP_EDGES  = ['#2E2317', '#6D452E', '#8D6042', '#AF916C', '#C4B699']
HP_COLORS = ['#214334', '#33665A', '#45799B', '#6FA4C4', '#A0C7DD']
HP_EDGES  = ['#162F24', '#274B42', '#345A78', '#587F9C', '#7EA2B4']

ANNOT_SIZE = 8.0
BAR_W = 0.18


# ── Helpers ─────────────────────────────────────────────────────────────
def _load_dfs(data_dir: Path) -> dict:
    dfs = {}
    for m in MODELS:
        csv_path = data_dir / f'{m}_results.csv'
        if csv_path.exists():
            dfs[m] = pd.read_csv(csv_path)
    return dfs


def get(df, strat, batch, col, y_chips, x_chips):
    row = df[
        (df['strategy'] == strat)
        & (df['batch'] == batch)
        & (df['y_chips'] == y_chips)
        & (df['x_chips'] == x_chips)
    ]
    return row[col].values[0] if len(row) else np.nan


def _save_svg(fig, path):
    fig.savefig(path, bbox_inches='tight')
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    content = re.sub(
        r'(width|height)="([\d.]+)pt"',
        lambda m: f'{m.group(1)}="{float(m.group(2)) * 96 / 72:.2f}px"',
        content,
    )
    path.write_text(content, encoding='utf-8')


def fmt(v, is_base=False):
    if is_base:
        return '1'
    if abs(v) < 1e-12:
        return '0'
    return f'{v:.1f}'


_MIN_GAP_TP = 1.1
_MIN_GAP_HP = 0.28


def _annotate_subplot(ax, bars_by_s, ratios_by_s, metric):
    max_y = 0.0
    for bi in range(len(BATCHES)):
        items = []
        for sj, (bars, ratios) in enumerate(zip(bars_by_s, ratios_by_s)):
            r = ratios[bi]
            if np.isnan(r) or r <= 0:
                continue
            items.append((sj, STRATEGIES[sj], bars[bi], r))
        if not items:
            continue
        items_sorted = sorted(items, key=lambda t: t[3])

        label_pos = []
        for _rank, (sj, strat, bar, r) in enumerate(items_sorted):
            cx = bar.get_x() + bar.get_width() / 2
            if metric == 'throughput':
                y = r + 0.12
                fs = ANNOT_SIZE
            else:
                y = r * 1.15 if r > 0 else 0.65
                fs = ANNOT_SIZE - 0.5
            label_pos.append([sj, strat, cx, y, r, fs])

        if metric == 'throughput':
            for i in range(1, len(label_pos)):
                if label_pos[i][3] < label_pos[i - 1][3] + _MIN_GAP_TP:
                    label_pos[i][3] = label_pos[i - 1][3] + _MIN_GAP_TP
        else:
            for i in range(1, len(label_pos)):
                yp, yn = label_pos[i - 1][3], label_pos[i][3]
                if yp > 0 and yn > 0:
                    if np.log10(yn) - np.log10(yp) < _MIN_GAP_HP:
                        label_pos[i][3] = yp * (10 ** _MIN_GAP_HP)

        center_rank = (len(label_pos) - 1) / 2.0
        for rank, (sj, strat, cx, y, r, fs) in enumerate(label_pos):
            dx = (rank - center_rank) * 0.007
            ax.text(cx + dx, y, fmt(r, strat == 'base'),
                    ha='center', va='bottom', fontsize=fs, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.05', fc='white',
                              ec='none', alpha=0.55))
            max_y = max(max_y, y)
    return max_y


def plot_stacked(dfs: dict, out_dir: Path):
    """Single stacked figure (4 rows x 4 cols).
    Missing models are left as blank subplots."""
    n_shapes = len(SHAPES)
    n_rows_total = 2 * n_shapes
    fig, axes = plt.subplots(n_rows_total, 4, figsize=(14.0, 6.0))
    plt.subplots_adjust(hspace=0.12, wspace=0.10,
                        top=0.88, bottom=0.06, left=0.07, right=0.95)

    x = np.arange(len(BATCHES))
    n = len(STRATEGIES)

    metric_valid = {'throughput': [], 'hop': []}
    metric_max_annot = {'throughput': 0.0, 'hop': 0.0}

    for metric_idx, (metric, colors, edges) in enumerate(
        [('throughput', TP_COLORS, TP_EDGES), ('hop', HP_COLORS, HP_EDGES)]
    ):
        row_offset = metric_idx * n_shapes

        for si, (shape, _shape_title) in enumerate(zip(SHAPES, SHAPE_TITLES)):
            y_chips, x_chips = shape
            for ci, (model, mlabel) in enumerate(zip(MODELS, MODEL_LABELS)):
                ax = axes[row_offset + si, ci]

                # ---- skip missing models: leave subplot blank ----
                if model not in dfs:
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(b) for b in BATCHES])
                    ax.grid(axis='y', ls='--', lw=0.3, alpha=0.5, zorder=0)
                    if metric == 'throughput' and si == 0:
                        ax.set_title(mlabel, fontsize=10, fontweight='bold',
                                     pad=3, color='#AAAAAA')
                    continue

                df = dfs[model]
                base_col = 'throughput' if metric == 'throughput' else 'hop'
                base_vals = [get(df, 'base', b, base_col, y_chips, x_chips)
                             for b in BATCHES]

                bars_by_s, ratios_by_s = [], []
                for sj, strat in enumerate(STRATEGIES):
                    raw = [get(df, strat, b, metric, y_chips, x_chips)
                           for b in BATCHES]
                    if metric == 'throughput':
                        ratios = [v / bv if (not np.isnan(v) and not np.isnan(bv) and bv > 0)
                                  else np.nan for v, bv in zip(raw, base_vals)]
                    else:
                        ratios = [bv / v if (not np.isnan(v) and not np.isnan(bv) and v > 0)
                                  else np.nan for bv, v in zip(base_vals, raw)]
                    metric_valid[metric].extend(
                        r for r in ratios if not np.isnan(r) and r > 0)

                    offset = (sj - (n - 1) / 2) * BAR_W
                    plot_ratios = [0 if np.isnan(r) else r for r in ratios]
                    bars = ax.bar(x + offset, plot_ratios, BAR_W * 0.88,
                                  color=colors[sj], edgecolor=edges[sj],
                                  linewidth=0.5, zorder=3)
                    bars_by_s.append(bars)
                    ratios_by_s.append(ratios)

                my = _annotate_subplot(ax, bars_by_s, ratios_by_s, metric)
                metric_max_annot[metric] = max(metric_max_annot[metric], my)

                ax.set_xticks(x)
                ax.set_xticklabels([str(b) for b in BATCHES])
                ax.grid(axis='y', ls='--', lw=0.3, alpha=0.5, zorder=0)
                if metric == 'throughput' and si == 0:
                    ax.set_title(mlabel, fontsize=10, fontweight='bold', pad=3)

    # ── shared y limits ──────────────────────────────────────────────────
    tp_max = max(metric_valid['throughput']) if metric_valid['throughput'] else 3.0
    hop_max = max(metric_valid['hop']) if metric_valid['hop'] else 50.0
    tp_ylim = max(tp_max * 1.15, metric_max_annot['throughput'] * 1.10)
    hop_ylim = max(hop_max * 5.0, metric_max_annot['hop'] * 2.5)

    for row in range(n_rows_total):
        is_hop = row >= n_shapes
        for col in range(4):
            ax = axes[row, col]
            if not is_hop:
                ax.set_ylim(0, tp_ylim)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
            else:
                ax.set_yscale('log')
                ax.set_ylim(0.5, hop_ylim)
                ax.yaxis.set_major_locator(
                    ticker.LogLocator(base=10, numticks=6))
                ax.yaxis.set_minor_locator(
                    ticker.LogLocator(base=10, subs='auto', numticks=12))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # y tick labels only on col-0
    for row in range(n_rows_total):
        for col in range(1, 4):
            axes[row, col].tick_params(labelleft=False)

    # x-labels only on the very last row
    for row in range(n_rows_total - 1):
        for col in range(4):
            axes[row, col].tick_params(labelbottom=False)

    # ── vertical shift to open gap between throughput and hop blocks ─────
    shift = 0.030
    for row in range(n_rows_total):
        for col in range(4):
            box = axes[row, col].get_position()
            if row < n_shapes:
                box.y0 += shift; box.y1 += shift
            else:
                box.y0 -= shift; box.y1 -= shift
            axes[row, col].set_position(box)

    # die-shape labels on the RIGHT side
    for row_idx in range(n_rows_total):
        pos = axes[row_idx, -1].get_position()
        fig.text(pos.x1 + 0.02, pos.y0 + pos.height / 2.0,
                 SHAPE_TITLES[row_idx % n_shapes],
                 rotation=+90, va='center', ha='center',
                 fontsize=10, fontweight='bold')

    # metric labels on the LEFT side
    tp_top = axes[0, 0].get_position().y1
    tp_bottom = axes[n_shapes - 1, 0].get_position().y0
    hp_top = axes[n_shapes, 0].get_position().y1
    hp_bottom = axes[2 * n_shapes - 1, 0].get_position().y0
    fig.text(0.04, (tp_top + tp_bottom) / 2, 'Throughput',
             rotation=90, va='center', ha='center',
             fontsize=11, fontweight='bold')
    fig.text(0.04, (hp_top + hp_bottom) / 2, 'Hop Reduction',
             rotation=90, va='center', ha='center',
             fontsize=11, fontweight='bold')

    # legends
    legend_kw = dict(ncol=len(STRATEGIES), frameon=True, edgecolor='#AAAAAA',
                     fancybox=False, handlelength=1.2, handletextpad=0.4,
                     columnspacing=1.0, fontsize=10)
    tp_handles = [plt.Rectangle((0, 0), 1, 1, fc=TP_COLORS[i],
                  ec=TP_EDGES[i], lw=0.5) for i in range(len(STRATEGIES))]
    hp_handles = [plt.Rectangle((0, 0), 1, 1, fc=HP_COLORS[i],
                  ec=HP_EDGES[i], lw=0.5) for i in range(len(STRATEGIES))]
    fig.legend(tp_handles, STRATEGY_LABELS, loc='upper center',
               bbox_to_anchor=(0.51, 1), **legend_kw)
    gap_y = (tp_bottom + hp_top) / 2
    fig.legend(hp_handles, STRATEGY_LABELS, loc='center',
               bbox_to_anchor=(0.51, gap_y), **legend_kw)

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = 'e2e_stacked'
    stem = out_dir / fname
    fig.savefig(f'{stem}.png', bbox_inches='tight', dpi=300)
    _save_svg(fig, f'{stem}.svg')
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}.png / .svg / .pdf -> {out_dir}')


def plot_available_results(dfs: dict, out_dir: Path):
    """Plot the raw rows that are actually present in results/*.csv.

    Unlike the paper-style stacked figure, this plot accepts arbitrary batch
    sizes, chiplet shapes, and subsets of strategies.  It is useful for quick
    experiments such as a single batch-256 baseline run.
    """
    frames = [df.copy() for df in dfs.values()]
    data = pd.concat(frames, ignore_index=True)
    required = {'model', 'strategy', 'y_chips', 'x_chips', 'batch',
                'throughput', 'hop'}
    if not required.issubset(data.columns):
        print('Skipping available-results plot: result CSV has unexpected columns.')
        return

    strategy_order = [s for s in STRATEGIES if s in set(data['strategy'])]
    strategy_order.extend(s for s in sorted(set(data['strategy']))
                          if s not in strategy_order)
    configs = list(data[['model', 'y_chips', 'x_chips', 'batch']]
                   .drop_duplicates()
                   .sort_values(['model', 'y_chips', 'x_chips', 'batch'])
                   .itertuples(index=False, name=None))
    labels = [f'{model}\n{y}×{x}, batch {batch}'
              for model, y, x, batch in configs]

    fig, (ax_tp, ax_hop) = plt.subplots(1, 2, figsize=(max(9, len(configs) * 2.2), 4.8))
    x = np.arange(len(configs))
    width = 0.72 / max(len(strategy_order), 1)
    color_map = {strategy: TP_COLORS[i % len(TP_COLORS)]
                 for i, strategy in enumerate(strategy_order)}

    for idx, strategy in enumerate(strategy_order):
        throughput, hops = [], []
        for model, y_chips, x_chips, batch in configs:
            row = data[(data['model'] == model) & (data['strategy'] == strategy)
                       & (data['y_chips'] == y_chips) & (data['x_chips'] == x_chips)
                       & (data['batch'] == batch)]
            throughput.append(float(row['throughput'].iloc[0]) if not row.empty else np.nan)
            hops.append(float(row['hop'].iloc[0]) if not row.empty else np.nan)
        offset = (idx - (len(strategy_order) - 1) / 2) * width
        bars_tp = ax_tp.bar(x + offset, np.nan_to_num(throughput, nan=0.0), width,
                            label=strategy, color=color_map[strategy])
        ax_hop.bar(x + offset, np.nan_to_num(hops, nan=0.0), width,
                   label=strategy, color=color_map[strategy])
        for bar, value in zip(bars_tp, throughput):
            if not np.isnan(value):
                ax_tp.annotate(f'{value:,.0f}',
                               (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               ha='center', va='bottom', fontsize=8, xytext=(0, 3),
                               textcoords='offset points')

    for ax, title, ylabel in [
        (ax_tp, 'Raw simulated throughput', 'Throughput'),
        (ax_hop, 'Raw inter-chiplet hops', 'Total hops'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', ls='--', lw=0.5, alpha=0.5)
    ax_tp.legend(title='Strategy')
    fig.suptitle('Available simulator results', fontweight='bold')
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / 'available_results'
    fig.savefig(f'{stem}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved available_results.png / .pdf -> {out_dir}')


def save_summary_csv(dfs: dict, out_dir: Path):
    rows = []
    for model, mlabel in zip(MODELS, MODEL_LABELS):
        if model not in dfs:
            continue
        df = dfs[model]
        for shape, shape_title in zip(SHAPES, SHAPE_TITLES):
            y_chips, x_chips = shape
            for batch in BATCHES:
                base_tp = get(df, 'base', batch, 'throughput', y_chips, x_chips)
                base_hop = get(df, 'base', batch, 'hop', y_chips, x_chips)
                for strat, slabel in zip(STRATEGIES, STRATEGY_LABELS):
                    tp_raw = get(df, strat, batch, 'throughput', y_chips, x_chips)
                    hop_raw = get(df, strat, batch, 'hop', y_chips, x_chips)
                    tp_ratio = (tp_raw / base_tp
                                if (not np.isnan(tp_raw) and base_tp > 0)
                                else np.nan)
                    hop_ratio = (base_hop / hop_raw
                                 if (not np.isnan(hop_raw) and hop_raw > 0)
                                 else np.nan)
                    rows.append(dict(
                        model=mlabel, shape=shape_title, batch=batch,
                        strategy=slabel,
                        throughput_speedup=tp_ratio,
                        hop_reduction=hop_ratio,
                    ))

    def _mean(vals):
        v = [x for x in vals if not np.isnan(x)]
        return float(np.mean(v)) if v else float('nan')

    strategies_all = list(dict.fromkeys(r['strategy'] for r in rows))
    shapes_all = list(dict.fromkeys(r['shape'] for r in rows))
    batches_all = sorted({r['batch'] for r in rows})
    models_all = list(dict.fromkeys(r['model'] for r in rows))

    summary_rows = []
    for strat in strategies_all:
        sub = [r for r in rows if r['strategy'] == strat]
        summary_rows.append(dict(
            group='overall', segment='all', strategy=strat,
            avg_throughput_speedup=_mean([r['throughput_speedup'] for r in sub]),
            avg_hop_reduction=_mean([r['hop_reduction'] for r in sub]),
            n=len(sub),
        ))
    for batch in batches_all:
        for strat in strategies_all:
            sub = [r for r in rows
                   if r['batch'] == batch and r['strategy'] == strat]
            summary_rows.append(dict(
                group='batch_size', segment=str(batch), strategy=strat,
                avg_throughput_speedup=_mean(
                    [r['throughput_speedup'] for r in sub]),
                avg_hop_reduction=_mean([r['hop_reduction'] for r in sub]),
                n=len(sub),
            ))
    for shape in shapes_all:
        for strat in strategies_all:
            sub = [r for r in rows
                   if r['shape'] == shape and r['strategy'] == strat]
            summary_rows.append(dict(
                group='die_shape', segment=shape, strategy=strat,
                avg_throughput_speedup=_mean(
                    [r['throughput_speedup'] for r in sub]),
                avg_hop_reduction=_mean([r['hop_reduction'] for r in sub]),
                n=len(sub),
            ))
    for model in models_all:
        for strat in strategies_all:
            sub = [r for r in rows
                   if r['model'] == model and r['strategy'] == strat]
            summary_rows.append(dict(
                group='model', segment=model, strategy=strat,
                avg_throughput_speedup=_mean(
                    [r['throughput_speedup'] for r in sub]),
                avg_hop_reduction=_mean([r['hop_reduction'] for r in sub]),
                n=len(sub),
            ))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'e2e_summary.csv'
    fieldnames = ['group', 'segment', 'strategy',
                  'avg_throughput_speedup', 'avg_hop_reduction', 'n']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sr in summary_rows:
            writer.writerow({
                k: (f'{sr[k]:.4f}' if isinstance(sr[k], float) else sr[k])
                for k in fieldnames
            })
    print(f'Saved e2e_summary.csv -> {out_dir}')


def process_ae_results(data_dir: Path, out_dir: Path):
    """Entry point for AE: load CSVs, plot stacked figure, write summary."""
    dfs = _load_dfs(data_dir)
    if not dfs:
        print(f'No CSV files found in {data_dir}, skipping.')
        return
    available = list(dfs.keys())
    missing = [m for m in MODELS if m not in dfs]
    if missing:
        print(f'  Note: models {missing} not available (subplots left blank)')
    print(f'  Generating stacked plot for {len(dfs)} model(s): {available}')
    plot_stacked(dfs, out_dir)
    plot_available_results(dfs, out_dir)
    save_summary_csv(dfs, out_dir)
