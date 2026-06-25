from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


STAGES = [
    {
        "stage": "1",
        "title": "Real county\nenvironment",
        "fill": "#e8eef5",
        "edge": "#2f425a",
    },
    {
        "stage": "2",
        "title": "Trajectory collection\n(random + greedy)",
        "fill": "#eef5fb",
        "edge": "#37648a",
    },
    {
        "stage": "3",
        "title": "Transition model\ntraining",
        "fill": "#edf7f0",
        "edge": "#3f7a53",
    },
    {
        "stage": "4",
        "title": "Learned county\nenvironment",
        "fill": "#f3f2fb",
        "edge": "#5a4f8a",
    },
    {
        "stage": "5",
        "title": "Policy optimization\n(optional calibration)",
        "fill": "#fdf2e7",
        "edge": "#9a5d1c",
    },
    {
        "stage": "6",
        "title": "Real-environment\nevaluation",
        "fill": "#eef0f2",
        "edge": "#4d5966",
    },
]


def draw_pipeline(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 6.6)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    box_width = 0.88
    box_height = 0.36
    y = 0.32
    xs = [0.12, 1.17, 2.22, 3.27, 4.32, 5.37]

    for idx, (x, stage) in enumerate(zip(xs, STAGES, strict=True)):
        patch = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            linewidth=1.4,
            facecolor=stage["fill"],
            edgecolor=stage["edge"],
        )
        ax.add_patch(patch)
        ax.text(
            x + box_width / 2.0,
            y + box_height * 0.78,
            f"STAGE {stage['stage']}",
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=stage["edge"],
        )
        ax.text(
            x + box_width / 2.0,
            y + box_height * 0.40,
            stage["title"],
            ha="center",
            va="center",
            fontsize=10.6,
            color="#19222d",
            linespacing=1.08,
        )

        if idx < len(xs) - 1:
            arrow = FancyArrowPatch(
                (x + box_width, y + box_height / 2.0),
                (xs[idx + 1], y + box_height / 2.0),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.45,
                color="#5b6774",
                shrinkA=0,
                shrinkB=0,
            )
            ax.add_patch(arrow)


def build_figure(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure_1_pipeline.pdf"
    png_path = output_dir / "figure_1_pipeline.png"

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(14.2, 2.95))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    draw_pipeline(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submission/ceus/05_figures"),
        help="Directory for figure_1_pipeline.pdf/png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path, png_path = build_figure(args.output_dir)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
