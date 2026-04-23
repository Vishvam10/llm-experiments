import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import catppuccin
from matplotlib.lines import Line2D

from catppuccin.extras.matplotlib import get_colormap_from_list

mpl.style.use(catppuccin.PALETTE.macchiato.identifier)


def build_colormap(categories):
    cmap = get_colormap_from_list(
        catppuccin.PALETTE.macchiato.identifier,
        ["blue", "mauve", "green", "peach", "red"],
    )

    cat_to_idx = {c: i for i, c in enumerate(categories)}

    def get_color(cat):
        return cmap(cat_to_idx[cat] / (len(categories) - 1))

    return get_color


def build_legend(categories, get_color):
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


def plot_distance(distances, results_dir):
    plt.figure()
    plt.plot(range(len(distances)), distances, marker="o")
    plt.title("Context Influence Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Distance")
    plt.tight_layout()
    plt.savefig(results_dir / "distance.png", dpi=200)


def plot_embedding_2d(
    data, filename, results_dir, get_category, categories, get_color
):
    plt.figure(figsize=(10, 8))

    for p in data["points"]:
        marker = "o" if p["context"] == "no_ctx" else "x"
        color = get_color(get_category(p["task"]))

        plt.scatter(p["x"], p["y"], marker=marker, color=color)

        jitter = np.random.randint(2, 6)
        plt.annotate(
            str(p["index"]),
            (p["x"], p["y"]),
            textcoords="offset points",
            xytext=(jitter, jitter),
            fontsize=8,
        )

    title = f"{data['method'].upper()}-2D Layer {data['layer']}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=200)
    plt.close()


def plot_embedding_3d(
    data, filename, results_dir, get_category, categories, get_color
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Dark theme
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    for p in data["points"]:
        marker = "o" if p["context"] == "no_ctx" else "x"
        color = get_color(get_category(p["task"]))

        x, y, z = p["x"], p["y"], p["z"]

        ax.scatter(x, y, z, marker=marker, color=color)
        jitter = np.random.uniform(0.005, 0.02)
        ax.text(
            x + jitter,
            y + jitter,
            z + jitter,
            str(p["index"]),
            fontsize=7,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True

    ax.xaxis.pane.set_facecolor("#1e1e2e")
    ax.yaxis.pane.set_facecolor("#1e1e2e")
    ax.zaxis.pane.set_facecolor("#1e1e2e")

    ax.grid(False)

    ax.set_title(f"{data['method'].upper()}-3D Layer {data['layer']}")

    fig.legend(
        handles=build_legend(categories, get_color),
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(results_dir / filename, dpi=200)
    plt.close()


def plot_multi_embedding(
    layers, results_dir, get_category, categories, get_color
):
    num_layers = len(layers)
    dim = layers[0]["dim"]
    method = layers[0]["method"]

    cols = 4
    rows = int(np.ceil(num_layers / cols))

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    axes = []
    for i in range(num_layers):
        if dim == 2:
            ax = fig.add_subplot(rows, cols, i + 1)
        else:
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        axes.append(ax)

    for _, (layer_data, ax) in enumerate(zip(layers, axes)):
        for p in layer_data["points"]:
            marker = "o" if p["context"] == "no_ctx" else "x"
            color = get_color(get_category(p["task"]))

            if dim == 2:
                ax.scatter(p["x"], p["y"], marker=marker, color=color)
            else:
                ax.scatter(p["x"], p["y"], p["z"], marker=marker, color=color)

        ax.set_title(f"L{layer_data['layer']}", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        if dim == 3:
            ax.set_zticks([])

            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True

            ax.xaxis.pane.set_facecolor("#1e1e2e")
            ax.yaxis.pane.set_facecolor("#1e1e2e")
            ax.zaxis.pane.set_facecolor("#1e1e2e")

            ax.grid(False)

    for j in range(len(axes), rows * cols):
        fig.delaxes(fig.add_subplot(rows, cols, j + 1))


    fig.suptitle(
        f"{method.upper()}-{dim}D Across Layers",
        fontsize=16,
        y=0.98,
    )

    subtitle = (
        "Points closer together have more similar representations. "
        "Separation indicates stronger contextual or task-specific divergence."
    )

    fig.text(
        0.5,
        0.94,
        subtitle,
        ha="center",
        fontsize=8,
    )

    legend_elements = build_legend(categories, get_color)

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=min(len(legend_elements), 6),
        bbox_to_anchor=(0.5, 0.90),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(
        results_dir / f"{method}_{dim}d_layers_combined.png",
        dpi=200,
    )
    plt.close()