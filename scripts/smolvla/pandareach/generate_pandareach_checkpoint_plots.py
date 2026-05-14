from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CHECKPOINT_CSV = Path("evidence/smolvla_pandareach_checkpoint_comparison/checkpoint_comparison.csv")
BASELINE_CSV = Path("evidence/smolvla_pandareach_results/rollout_results_compact.csv")

OUT_DIR = Path("evidence/smolvla_pandareach_checkpoint_comparison/plots")


def read_checkpoint_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint comparison CSV: {path}")

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue

            checkpoint = row["checkpoint"]
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "step": int(checkpoint),
                    "success_rate": float(row["success_rate"]),
                    "mean_best_distance": float(row["mean_best_distance"]),
                    "mean_steps": float(row["mean_steps"]),
                }
            )

    rows.sort(key=lambda item: item["step"])
    return rows


def read_baselines(path: Path) -> dict[str, dict]:
    baselines: dict[str, dict] = {}

    if not path.exists():
        return baselines

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("run") != "pandareach_chunk1_500_eval50_scale1":
                continue

            policy = row.get("policy")
            if policy not in {"random", "zero"}:
                continue

            baselines[policy] = {
                "success_rate": float(row["success_rate"]),
                "mean_best_distance": float(row["mean_best_distance"]),
                "mean_steps": float(row["mean_steps"]),
            }

    return baselines


def annotate_points(xs: list[int], ys: list[float], fmt: str) -> None:
    for x, y in zip(xs, ys):
        plt.annotate(
            fmt.format(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )


def save_line_plot(
    *,
    xs: list[int],
    ys: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    annotation_format: str,
    baseline_values: dict[str, float] | None = None,
    ylim: tuple[float, float] | None = None,
    lower_is_better: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(xs, ys, marker="o", label="SmolVLA checkpoints")
    annotate_points(xs, ys, annotation_format)

    if baseline_values:
        for name, value in baseline_values.items():
            plt.axhline(value, linestyle="--", linewidth=1, label=f"{name} baseline")
            plt.annotate(
                annotation_format.format(value),
                (xs[-1], value),
                textcoords="offset points",
                xytext=(8, 0),
                va="center",
                fontsize=8,
            )

    subtitle = "lower is better" if lower_is_better else "higher is better"
    plt.title(f"{title}\n({subtitle})")
    plt.xlabel("Training checkpoint")
    plt.ylabel(ylabel)
    plt.xticks(xs, [str(x) for x in xs])

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_markdown_report(rows: list[dict], baselines: dict[str, dict]) -> None:
    best_success = max(rows, key=lambda row: row["success_rate"])
    best_distance = min(rows, key=lambda row: row["mean_best_distance"])
    best_steps = min(rows, key=lambda row: row["mean_steps"])

    lines = [
        "# SmolVLA PandaReach checkpoint plots",
        "",
        "Generated from:",
        "",
        f"- `{CHECKPOINT_CSV}`",
        f"- `{BASELINE_CSV}`",
        "",
        "## Plots",
        "",
        "### Success rate",
        "",
        "![Success rate by checkpoint](success_rate_by_checkpoint.png)",
        "",
        "### Mean best distance",
        "",
        "![Mean best distance by checkpoint](mean_best_distance_by_checkpoint.png)",
        "",
        "### Mean steps",
        "",
        "![Mean steps by checkpoint](mean_steps_by_checkpoint.png)",
        "",
        "## Best observed checkpoints",
        "",
        f"- Best success rate: checkpoint `{best_success['checkpoint']}` with `{best_success['success_rate']:.2f}`.",
        f"- Best mean best distance: checkpoint `{best_distance['checkpoint']}` with `{best_distance['mean_best_distance']:.4f}`.",
        f"- Best mean steps: checkpoint `{best_steps['checkpoint']}` with `{best_steps['mean_steps']:.2f}`.",
        "",
        "## Baselines",
        "",
    ]

    if baselines:
        for name, metrics in sorted(baselines.items()):
            lines.append(
                f"- `{name}`: success_rate=`{metrics['success_rate']:.2f}`, "
                f"mean_best_distance=`{metrics['mean_best_distance']:.4f}`, "
                f"mean_steps=`{metrics['mean_steps']:.2f}`."
            )
    else:
        lines.append("- No baseline rows found.")

    lines.append("")

    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_checkpoint_rows(CHECKPOINT_CSV)
    baselines = read_baselines(BASELINE_CSV)

    xs = [row["step"] for row in rows]

    success_values = [row["success_rate"] for row in rows]
    distance_values = [row["mean_best_distance"] for row in rows]
    steps_values = [row["mean_steps"] for row in rows]

    save_line_plot(
        xs=xs,
        ys=success_values,
        title="SmolVLA PandaReach success rate by checkpoint",
        ylabel="Success rate",
        output_path=OUT_DIR / "success_rate_by_checkpoint.png",
        annotation_format="{:.2f}",
        baseline_values={
            name: metrics["success_rate"]
            for name, metrics in baselines.items()
        },
        ylim=(0.0, 1.0),
        lower_is_better=False,
    )

    save_line_plot(
        xs=xs,
        ys=distance_values,
        title="SmolVLA PandaReach mean best distance by checkpoint",
        ylabel="Mean best distance",
        output_path=OUT_DIR / "mean_best_distance_by_checkpoint.png",
        annotation_format="{:.3f}",
        baseline_values={
            name: metrics["mean_best_distance"]
            for name, metrics in baselines.items()
        },
        lower_is_better=True,
    )

    save_line_plot(
        xs=xs,
        ys=steps_values,
        title="SmolVLA PandaReach mean steps by checkpoint",
        ylabel="Mean steps",
        output_path=OUT_DIR / "mean_steps_by_checkpoint.png",
        annotation_format="{:.1f}",
        baseline_values={
            name: metrics["mean_steps"]
            for name, metrics in baselines.items()
        },
        lower_is_better=True,
    )

    write_markdown_report(rows, baselines)

    print("Created:")
    for path in sorted(OUT_DIR.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()
