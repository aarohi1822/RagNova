"""Render the README / LinkedIn figures from results/metrics.json.

  python eval/plots.py
Writes:
  assets/retrieval_ablation.png  Recall@5 and MRR@10 with 95% CI error bars
  assets/judge_confusion.png     human labels vs LLM-judge verdicts

Every number is read from results/metrics.json; nothing is typed in by hand.
Colors are the dataviz reference palette (light surface): categorical slots 1-2
for the two metrics, the single-hue blue ramp for the confusion-matrix counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics.json"
ASSETS = ROOT / "assets"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"recall@5": "#2a78d6", "mrr@10": "#eb6834"}  # categorical slots 1, 2
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CONFIGS = ["BM25 only", "FAISS only", "Hybrid (RRF)", "Hybrid + cross-encoder"]
TICK_LABELS = ["BM25\nonly", "FAISS\nonly", "Hybrid\n(RRF)", "Hybrid (RRF)\n+ cross-encoder"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
    "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.94, title, fontsize=15, fontweight="bold", color=TEXT, va="top")
    fig.text(0.04, 0.875, subtitle, fontsize=9.5, color=TEXT_2, va="top")


def ablation_chart(metrics: dict) -> Path:
    results = metrics["retrieval_ablation"]["results"]
    meta = metrics["retrieval_ablation"]["meta"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.17)

    width, gap = 0.36, 0.02  # 2-unit visual gap between paired bars
    best = max(CONFIGS, key=lambda c: results[c]["recall@5"])
    for j, (metric, label) in enumerate([("recall@5", "Recall@5"), ("mrr@10", "MRR@10")]):
        xs = [i + (j - 0.5) * (width + gap) for i in range(len(CONFIGS))]
        vals = [results[c][metric] for c in CONFIGS]
        cis = [results[c][f"{metric}_ci"] for c in CONFIGS]
        err = [[v - lo for v, (lo, _) in zip(vals, cis, strict=True)],
               [hi - v for v, (_, hi) in zip(vals, cis, strict=True)]]
        ax.bar(xs, vals, width, color=SERIES[metric], label=label, zorder=2)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=TEXT_2, elinewidth=1,
                    capsize=3, capthick=1, zorder=3)
        i = CONFIGS.index(best)  # label only the headline configuration
        ax.text(xs[i], cis[i][1] + 0.02, f"{vals[i]:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=TEXT)

    ax.set_xticks(range(len(CONFIGS)), TICK_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9.5, labelcolor=TEXT)

    header(fig, "Cross-encoder reranking gives the largest retrieval gain",
           f"{meta['queries']}-query benchmark over {meta['passages']:,} passages · "
           "error bars: 95% bootstrap CI · source: results/metrics.json")
    out = ASSETS / "retrieval_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def judge_confusion(metrics: dict) -> Path:
    agr = metrics["judge_reliability"]["agreement"]
    r = agr["vs_judge"]["human"]["app_truncated"]
    c = r["confusion"]
    grid = [[c["both_yes"], c["human_yes_judge_no"]],
            [c["human_no_judge_yes"], c["both_no"]]]
    notes = [["both: faithful", "only judge flags"],
             ["missed by judge", "both: unfaithful"]]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.subplots_adjust(left=0.27, right=0.80, top=0.74, bottom=0.12)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmax = max(max(row) for row in grid)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax)
    for i in range(2):
        for j in range(2):
            v = grid[i][j]
            ink = "#ffffff" if v > vmax * 0.55 else TEXT
            ax.text(j, i - 0.08, f"{v}", ha="center", va="center", fontsize=22,
                    fontweight="bold", color=ink)
            ax.text(j, i + 0.22, notes[i][j], ha="center", va="center", fontsize=8.5,
                    color=ink)
    for x in (0.5,):  # 2px-style surface gap between cells
        ax.axvline(x, color=SURFACE, linewidth=3)
        ax.axhline(x, color=SURFACE, linewidth=3)
    ax.set_xticks([0, 1], ["Judge: faithful", "Judge: unfaithful"])
    ax.set_yticks([0, 1], ["Human: faithful", "Human: unfaithful"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=10, labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)

    kappa = r["kappa"]
    header(fig, "The LLM judge rated every answer as faithful",
           f"Human labels vs LLM-judge verdicts on {r['n']} RAG answers "
           f"(judge score ≥ {agr['threshold']} = faithful)\n"
           f"Agreement {r['agreement']:.2f} · Cohen's κ = {kappa:.2f} · "
           "source: results/metrics.json")
    fig.text(0.04, 0.04, f"Judge: {agr['backend']}. Items the human marked unsure "
             "are excluded.", fontsize=8, color=TEXT_2)
    out = ASSETS / "judge_confusion.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    ASSETS.mkdir(exist_ok=True)
    for path in (ablation_chart(metrics), judge_confusion(metrics)):
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
