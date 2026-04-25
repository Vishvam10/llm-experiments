import argparse
import json
from pathlib import Path

import catppuccin
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from catppuccin.extras.matplotlib import get_colormap_from_list
from matplotlib.lines import Line2D

mpl.style.use(catppuccin.PALETTE.macchiato.identifier)


def build_colormap(categories):
    cmap = get_colormap_from_list(
        catppuccin.PALETTE.macchiato.identifier,
        ["blue", "mauve", "green", "peach", "red"],
    )

    cat_to_idx = {c: i for i, c in enumerate(categories)}

    def get_color(cat):
        return cmap(cat_to_idx[cat] / max(len(categories) - 1, 1))

    return get_color


def get_category(task):
    if task.startswith("math"):
        return "math"
    if task.startswith("code"):
        return "code"
    if task.startswith("logic"):
        return "logic"
    if task.startswith("lang"):
        return "language"
    return "adversarial"


categories = ["math", "code", "logic", "language", "adversarial"]


def build_legend(get_color):
    elements = [
        Line2D([0], [0], marker="o", linestyle="None", label="No Context"),
        Line2D([0], [0], marker="x", linestyle="None", label="With Context"),
    ]

    for cat in categories:
        elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                label=cat.capitalize(),
                color=get_color(cat),
            )
        )

    return elements


def plot_distance(distances, method, dim, out_dir):
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        range(len(distances)), distances, marker="o", label="Cosine Distance"
    )

    ax.set_title("Context Influence Across Layers", fontsize=20, pad=14)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Cosine Distance", fontsize=14)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)

    fig.text(
        0.5,
        0.92,
        "Lower means both prompt versions stay closer in representation space. Higher means context changes the representation more.",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_dir / f"{method}_{dim}d_distance.png", dpi=200)
    plt.close()


def plot_multi_embedding(layers, out_dir, get_color):
    num_layers = len(layers)
    dim = layers[0]["dim"]
    method = layers[0]["method"]

    cols = 4
    rows = int(np.ceil(num_layers / cols))

    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    axes = []

    for i in range(num_layers):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        axes.append(ax)

    for layer_data, ax in zip(layers, axes):
        for p in layer_data["points"]:
            marker = "o" if p["context"] == "no_ctx" else "x"
            color = get_color(get_category(p["task"]))

            ax.scatter(
                p["x"],
                p["y"],
                p["z"],
                marker=marker,
                color=color,
                s=22,
            )

        ax.set_title(f"L{layer_data['layer']}", fontsize=10, pad=2)

        # keep ticks tiny but visible
        ax.tick_params(axis="x", labelsize=5, pad=-2)
        ax.tick_params(axis="y", labelsize=5, pad=-2)
        ax.tick_params(axis="z", labelsize=5, pad=-2)

        # subtle pane backgrounds so the grid is actually visible
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True

        ax.xaxis.pane.set_alpha(0.08)
        ax.yaxis.pane.set_alpha(0.08)
        ax.zaxis.pane.set_alpha(0.08)

        # force visible grid
        ax.grid(True)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"]["linewidth"] = 0.6
            axis._axinfo["grid"]["linestyle"] = "--"
            axis._axinfo["grid"]["alpha"] = 0.35

    fig.suptitle(f"{method.upper()}-{dim}D Across Layers", fontsize=18, y=0.985)

    fig.text(
        0.5,
        0.935,
        "More gap means strong contextual influence",
        ha="center",
        fontsize=9,
    )

    fig.legend(
        handles=build_legend(get_color),
        loc="upper center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        fontsize=10,
        handletextpad=0.6,
        columnspacing=1.6,
    )

    plt.subplots_adjust(
        top=0.84,
        hspace=0.28,
        wspace=0.18,
    )
    plt.savefig(out_dir / f"{method}_{dim}d_layers_combined.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with open(args.points) as f:
        payload = json.load(f)

    out_dir = Path(args.out) if args.out else Path(args.points).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    get_color = build_colormap(categories)

    plot_distance(
        payload["distance_per_layer"],
        payload["method"],
        payload["dim"],
        out_dir,
    )

    plot_multi_embedding(
        payload["layers"],
        out_dir,
        get_color,
    )


if __name__ == "__main__":
    main()
