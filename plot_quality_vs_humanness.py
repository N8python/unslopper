import json
import os
from collections import Counter
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import numpy as np

PANGRAM_FILE = "unslopped_stories_pangram_with_control.jsonl"
QUALITY_FILE = "unslopped_stories_quality_with_control.jsonl"
BASELINE_QUALITY_FILE = "short_stories_bad_quality.jsonl"
MISTRAL_QUALITY_FILE = "short_stories_mistral_quality.jsonl"
OUTPUT_FILE = "quality_vs_humanness.png"


def load_pangram(path: str) -> dict[int, dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story_id = obj.get("story_id")
            if isinstance(story_id, int):
                data[story_id] = obj
    return data


def load_quality(path: str) -> dict[int, dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story_id = obj.get("story_id")
            if isinstance(story_id, int):
                data[story_id] = obj
    return data


def weakest_score(scores: dict) -> Optional[float]:
    if not scores:
        return None
    vals = []
    for key in ("coherence", "style", "general"):
        val = scores.get(key)
        if val is None:
            return None
        vals.append(float(val))
    return min(vals)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_baseline_weakest(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    weakest = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            scores = obj.get("story_eval", {}).get("scores", {})
            vals = []
            for key in ("coherence", "style", "general"):
                val = scores.get(key)
                if val is None:
                    vals = []
                    break
                vals.append(float(val))
            if vals:
                weakest.append(min(vals))
    return mean(weakest) if weakest else None


def main() -> None:
    pangram = load_pangram(PANGRAM_FILE)
    quality = load_quality(QUALITY_FILE)
    baseline_weakest = load_baseline_weakest(BASELINE_QUALITY_FILE)
    mistral_baseline = load_baseline_weakest(MISTRAL_QUALITY_FILE)

    original_labels = Counter()
    unslopped_labels = Counter()
    original_x = []
    original_y = []
    unslopped_x = []
    unslopped_y = []
    control_x = []
    control_y = []

    common_ids = sorted(set(pangram.keys()) & set(quality.keys()))
    for story_id in common_ids:
        p = pangram[story_id]
        q = quality[story_id]
        op = p.get("original_pangram", {})
        up = p.get("unslopped_pangram", {})
        cp = p.get("control_pangram", {})
        oq = q.get("original_eval", {}).get("scores", {})
        uq = q.get("unslopped_eval", {}).get("scores", {})
        cq = q.get("control_eval", {}).get("scores", {})

        if "fraction_ai" not in op or "fraction_ai" not in up:
            continue
        o_min = weakest_score(oq)
        u_min = weakest_score(uq)
        if o_min is None or u_min is None:
            continue

        original_x.append(1.0 - float(op["fraction_ai"]))
        original_y.append(o_min)
        unslopped_x.append(1.0 - float(up["fraction_ai"]))
        unslopped_y.append(u_min)
        if "prediction_short" in op:
            original_labels[op["prediction_short"]] += 1
        if "prediction_short" in up:
            unslopped_labels[up["prediction_short"]] += 1
        if "fraction_ai" in cp:
            c_min = weakest_score(cq)
            if c_min is not None:
                control_x.append(1.0 - float(cp["fraction_ai"]))
                control_y.append(c_min)

    plt.rcParams.update(
        {
            "figure.facecolor": "#fafbfc",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "axes.labelcolor": "#1e293b",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.color": "#1e293b",
            "legend.framealpha": 0.97,
            "legend.edgecolor": "#e2e8f0",
            "legend.fancybox": True,
            "legend.shadow": False,
            "font.family": "sans-serif",
        }
    )

    fig = plt.figure(figsize=(12, 6.5), facecolor="#fafbfc")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.2, 1.8],
        height_ratios=[1, 1],
        wspace=0.18,
        hspace=0.35,
    )
    ax = fig.add_subplot(gs[:, 0])
    orig_pie_ax = fig.add_subplot(gs[0, 1])
    uns_pie_ax = fig.add_subplot(gs[1, 1])

    # Gradient-like background for main plot
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#e5e7eb", linewidth=0.6, alpha=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)

    # Clean pie chart backgrounds
    for pie_ax in (orig_pie_ax, uns_pie_ax):
        pie_ax.set_facecolor("#fafbfc")
        for spine in pie_ax.spines.values():
            spine.set_visible(False)
        pie_ax.set_xticks([])
        pie_ax.set_yticks([])
        pie_ax.set_aspect("equal")

    original_mean_x = mean(original_x)
    original_mean_y = mean(original_y)
    unslopped_mean_x = mean(unslopped_x)
    unslopped_mean_y = mean(unslopped_y)
    control_mean_x = mean(control_x) if control_x else None
    control_mean_y = mean(control_y) if control_y else None

    # Refined color palette
    blue_main = "#3b82f6"
    blue_light = "#93c5fd"
    green_main = "#22c55e"
    green_light = "#86efac"
    purple_main = "#a855f7"
    purple_light = "#c4b5fd"

    # Plot full distributions with soft glow effect
    control_z = 1
    unslopped_z = 2
    original_z = 3
    ax.scatter(
        original_x,
        original_y,
        s=50,
        color=blue_light,
        alpha=0.25,
        edgecolors="none",
        zorder=original_z,
    )
    ax.scatter(
        original_x,
        original_y,
        s=22,
        color=blue_main,
        alpha=0.4,
        edgecolors="none",
        zorder=original_z,
    )
    ax.scatter(
        unslopped_x,
        unslopped_y,
        s=50,
        color=green_light,
        alpha=0.25,
        edgecolors="none",
        zorder=unslopped_z,
    )
    ax.scatter(
        unslopped_x,
        unslopped_y,
        s=22,
        color=green_main,
        alpha=0.4,
        edgecolors="none",
        zorder=unslopped_z,
    )
    if control_x and control_y:
        ax.scatter(
            control_x,
            control_y,
            s=50,
            color=purple_light,
            alpha=0.25,
            edgecolors="none",
            zorder=control_z,
        )
        ax.scatter(
            control_x,
            control_y,
            s=22,
            color=purple_main,
            alpha=0.4,
            edgecolors="none",
            zorder=control_z,
        )

    # Mean markers with glow effect
    mean_markers = [
        (original_mean_x, original_mean_y, blue_main, blue_light),
        (unslopped_mean_x, unslopped_mean_y, green_main, green_light),
    ]
    if control_mean_x is not None and control_mean_y is not None:
        mean_markers.append(
            (control_mean_x, control_mean_y, purple_main, purple_light)
        )
    for mx, my, color, light in mean_markers:
        # Outer glow
        ax.scatter([mx], [my], s=450, color=light, alpha=0.3, edgecolors="none", zorder=3)
        # Main marker
        ax.scatter(
            [mx], [my], s=280, color=color, edgecolors="white", linewidth=2.5, zorder=4
        )
    label_offset = 0.25
    ax.text(
        unslopped_mean_x,
        unslopped_mean_y - label_offset,
        "Unslop",
        ha="center",
        va="top",
        fontsize=10,
        color="#111827",
        zorder=5,
    )
    if control_mean_x is not None and control_mean_y is not None:
        ax.text(
            control_mean_x,
            control_mean_y - label_offset,
            "Control",
            ha="center",
            va="top",
            fontsize=10,
            color="#111827",
            zorder=5,
        )

    # Elegant curved arrow connecting means
    ax.annotate(
        "",
        xy=(unslopped_mean_x, unslopped_mean_y),
        xytext=(original_mean_x, original_mean_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#374151",
            linewidth=2.0,
            shrinkA=16,
            shrinkB=16,
            mutation_scale=14,
            connectionstyle="arc3,rad=-0.15",
        ),
    )
    if control_mean_x is not None and control_mean_y is not None:
        ax.annotate(
            "",
            xy=(control_mean_x, control_mean_y),
            xytext=(original_mean_x, original_mean_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#374151",
                linewidth=1.6,
                shrinkA=16,
                shrinkB=16,
                mutation_scale=12,
                connectionstyle="arc3,rad=0.15",
                linestyle=":",
            ),
        )
        ax.annotate(
            "",
            xy=(unslopped_mean_x, unslopped_mean_y),
            xytext=(control_mean_x, control_mean_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#ef4444",
                linewidth=1.8,
                shrinkA=16,
                shrinkB=16,
                mutation_scale=12,
                connectionstyle="arc3,rad=0.08",
            ),
        )

    ax.set_title(
        "Writing Quality vs. Humanness",
        fontsize=15,
        fontweight="bold",
        pad=16,
        color="#111827",
    )
    ax.set_xlabel("Pangram Humanness  (1 − AI Fraction)", fontsize=11, labelpad=10)
    ax.set_ylabel("Weakest-Point Writing Score", fontsize=11, labelpad=10)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(4.2, 9.5)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=10,
            markerfacecolor=blue_main,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="GPT-5.2 (Original)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=10,
            markerfacecolor=green_main,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="GPT-5.2 + Unslopper-30B-A3B",
        ),
    ]
    if control_mean_x is not None and control_mean_y is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=10,
                markerfacecolor=purple_main,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="GPT-5.2 + Qwen3 VL 30B A3B",
            )
        )

    # Baseline lines with softer styling
    if baseline_weakest is not None:
        ax.axhline(
            baseline_weakest,
            color="#fb923c",
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
            zorder=1,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#fb923c",
                linestyle="--",
                linewidth=1.8,
                label="GPT-4o Mini Baseline",
            )
        )
    if mistral_baseline is not None:
        ax.axhline(
            mistral_baseline,
            color="#38bdf8",
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
            zorder=1,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#38bdf8",
                linestyle="--",
                linewidth=1.8,
                label="Mistral Large 3 Baseline",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
        facecolor="white",
        edgecolor="#e5e7eb",
        borderpad=1,
        labelspacing=0.8,
    )

    # Refined pie chart palette
    palette = {"AI": "#f87171", "Mixed": "#fbbf24", "Human": "#34d399"}
    order = ["Human", "Mixed", "AI"]

    if original_labels:
        labels = [label for label in order if label in original_labels]
        labels += [label for label in original_labels if label not in labels]
        sizes = [original_labels[label] for label in labels]
        colors = [palette.get(label, "#94a3b8") for label in labels]
        total = sum(sizes)

        orig_pie_ax.set_title(
            "Original\nAI Detection",
            fontsize=11,
            fontweight="semibold",
            pad=12,
            color="#374151",
        )
        wedges, _ = orig_pie_ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            labels=None,
            wedgeprops={"linewidth": 2, "edgecolor": "#fafbfc", "width": 0.5},
        )
        if len(labels) == 1:
            pct = 100.0 * sizes[0] / total if total else 0.0
            orig_pie_ax.text(
                0,
                0,
                f"{labels[0]}\n{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#374151",
            )

    if unslopped_labels:
        labels = [label for label in order if label in unslopped_labels]
        labels += [label for label in unslopped_labels if label not in labels]
        sizes = [unslopped_labels[label] for label in labels]
        colors = [palette.get(label, "#94a3b8") for label in labels]
        total = sum(sizes)

        uns_pie_ax.set_title(
            "Unslopped\nAI Detection",
            fontsize=11,
            fontweight="semibold",
            pad=12,
            color="#374151",
        )
        wedges, texts, autotexts = uns_pie_ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            labels=labels,
            autopct="%1.0f%%",
            pctdistance=0.73,
            labeldistance=1.15,
            textprops={"fontsize": 9, "color": "#374151", "fontweight": "medium"},
            wedgeprops={"linewidth": 2, "edgecolor": "#fafbfc", "width": 0.5},
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(9)

    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.12)
    fig.savefig(OUTPUT_FILE, dpi=250, facecolor="#fafbfc", bbox_inches="tight", pad_inches=0.3)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
