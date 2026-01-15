import json
import math
import os
from collections import Counter

INPUT_FILE = "unslopped_stories_pangram_with_control.jsonl"
QUALITY_FILE = "unslopped_stories_quality_with_control.jsonl"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def percent_breakdown(counter: Counter, total: int) -> list[tuple[str, float, int]]:
    breakdown = []
    for label, count in counter.most_common():
        pct = (count / total) * 100 if total else 0.0
        breakdown.append((label, pct, count))
    return breakdown


def print_breakdown(title: str, breakdown: list[tuple[str, float, int]]) -> None:
    print(title)
    for label, pct, count in breakdown:
        print(f"  {label}: {pct:.1f}% ({count})")


def main() -> None:
    original_vals = []
    unslopped_vals = []
    control_vals = []
    original_labels = Counter()
    unslopped_labels = Counter()
    control_labels = Counter()
    original_headlines = Counter()
    unslopped_headlines = Counter()
    control_headlines = Counter()
    total = 0
    control_total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            op = obj.get("original_pangram")
            up = obj.get("unslopped_pangram")
            cp = obj.get("control_pangram")
            if not op or not up:
                continue
            if "fraction_ai" not in op or "fraction_ai" not in up:
                continue
            total += 1
            original_vals.append(float(op["fraction_ai"]))
            unslopped_vals.append(float(up["fraction_ai"]))
            if "prediction_short" in op:
                original_labels[op["prediction_short"]] += 1
            if "prediction_short" in up:
                unslopped_labels[up["prediction_short"]] += 1
            if "headline" in op:
                original_headlines[op["headline"]] += 1
            if "headline" in up:
                unslopped_headlines[up["headline"]] += 1
            if cp and "fraction_ai" in cp:
                control_total += 1
                control_vals.append(float(cp["fraction_ai"]))
                if "prediction_short" in cp:
                    control_labels[cp["prediction_short"]] += 1
                if "headline" in cp:
                    control_headlines[cp["headline"]] += 1

    print(f"Count: {total}")
    print(
        f"Original fraction_ai mean: {mean(original_vals):.6f} "
        f"(stderr {stderr(original_vals):.6f})"
    )
    print(
        f"Unslopped fraction_ai mean: {mean(unslopped_vals):.6f} "
        f"(stderr {stderr(unslopped_vals):.6f})"
    )
    if control_total:
        print(
            f"Control fraction_ai mean: {mean(control_vals):.6f} "
            f"(stderr {stderr(control_vals):.6f})"
        )

    print_breakdown(
        "Original prediction_short breakdown:",
        percent_breakdown(original_labels, total),
    )
    print_breakdown(
        "Unslopped prediction_short breakdown:",
        percent_breakdown(unslopped_labels, total),
    )
    if control_total:
        print_breakdown(
            "Control prediction_short breakdown:",
            percent_breakdown(control_labels, control_total),
        )
    print_breakdown(
        "Original headline breakdown:",
        percent_breakdown(original_headlines, total),
    )
    print_breakdown(
        "Unslopped headline breakdown:",
        percent_breakdown(unslopped_headlines, total),
    )
    if control_total:
        print_breakdown(
            "Control headline breakdown:",
            percent_breakdown(control_headlines, control_total),
        )

    if not os.path.exists(QUALITY_FILE):
        print(f"Quality file not found: {QUALITY_FILE}")
        return

    quality = {
        "original": {"coherence": [], "style": [], "general": []},
        "unslopped": {"coherence": [], "style": [], "general": []},
        "control": {"coherence": [], "style": [], "general": []},
    }
    min_scores = {"original": [], "unslopped": [], "control": []}
    quality_count = 0
    control_quality_count = 0

    with open(QUALITY_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            orig_scores = obj.get("original_eval", {}).get("scores", {})
            uns_scores = obj.get("unslopped_eval", {}).get("scores", {})
            ctrl_scores = obj.get("control_eval", {}).get("scores", {})
            if any(
                orig_scores.get(k) is None or uns_scores.get(k) is None
                for k in ("coherence", "style", "general")
            ):
                pass
            else:
                quality_count += 1
                for key in ("coherence", "style", "general"):
                    quality["original"][key].append(float(orig_scores[key]))
                    quality["unslopped"][key].append(float(uns_scores[key]))
                min_scores["original"].append(
                    min(
                        float(orig_scores["coherence"]),
                        float(orig_scores["style"]),
                        float(orig_scores["general"]),
                    )
                )
                min_scores["unslopped"].append(
                    min(
                        float(uns_scores["coherence"]),
                        float(uns_scores["style"]),
                        float(uns_scores["general"]),
                    )
                )
            if all(ctrl_scores.get(k) is not None for k in ("coherence", "style", "general")):
                control_quality_count += 1
                for key in ("coherence", "style", "general"):
                    quality["control"][key].append(float(ctrl_scores[key]))
                min_scores["control"].append(
                    min(
                        float(ctrl_scores["coherence"]),
                        float(ctrl_scores["style"]),
                        float(ctrl_scores["general"]),
                    )
                )

    print(f"\nQuality eval count: {quality_count}")
    if control_quality_count:
        print(f"Control quality eval count: {control_quality_count}")
    for key in ("coherence", "style", "general"):
        print(
            f"Original {key} mean: {mean(quality['original'][key]):.6f} "
            f"(stderr {stderr(quality['original'][key]):.6f})"
        )
        print(
            f"Unslopped {key} mean: {mean(quality['unslopped'][key]):.6f} "
            f"(stderr {stderr(quality['unslopped'][key]):.6f})"
        )
        if control_quality_count:
            print(
                f"Control {key} mean: {mean(quality['control'][key]):.6f} "
                f"(stderr {stderr(quality['control'][key]):.6f})"
            )
    print(
        "Original weakest-point mean: "
        f"{mean(min_scores['original']):.6f} "
        f"(stderr {stderr(min_scores['original']):.6f})"
    )
    print(
        "Unslopped weakest-point mean: "
        f"{mean(min_scores['unslopped']):.6f} "
        f"(stderr {stderr(min_scores['unslopped']):.6f})"
    )
    if control_quality_count:
        print(
            "Control weakest-point mean: "
            f"{mean(min_scores['control']):.6f} "
            f"(stderr {stderr(min_scores['control']):.6f})"
        )


if __name__ == "__main__":
    main()
