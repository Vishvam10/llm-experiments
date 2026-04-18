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


def plot_pca(pca_data, filename, results_dir, get_category, categories):
    get_color = build_colormap(categories)

    plt.figure(figsize=(10, 8))

    for p in pca_data["points"]:
        marker = "o" if p["context"] == "no_ctx" else "x"
        color = get_color(get_category(p["task"]))

        x, y = p["x"], p["y"]
        plt.scatter(x, y, marker=marker, color=color)

        jitter = np.random.randint(2, 6)
        plt.annotate(
            str(p["index"]),
            (x, y),
            textcoords="offset points",
            xytext=(jitter, jitter),
            fontsize=8,
        )

    plt.legend(handles=build_legend(categories, get_color))
    plt.title(f"PCA Layer {pca_data['layer']}")
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=200)


def plot_multi(pca_layers, results_dir, get_category, categories):
    get_color = build_colormap(categories)

    cols = 4
    rows = int(np.ceil(len(pca_layers) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    for idx, layer_data in enumerate(pca_layers):
        ax = axes[idx]

        for p in layer_data["points"]:
            marker = "o" if p["context"] == "no_ctx" else "x"
            color = get_color(get_category(p["task"]))

            ax.scatter(p["x"], p["y"], marker=marker, color=color)

            jitter = np.random.randint(2, 5)
            ax.annotate(
                str(p["index"]),
                (p["x"], p["y"]),
                textcoords="offset points",
                xytext=(jitter, jitter),
                fontsize=6,
            )

        ax.set_title(f"Layer {layer_data['layer']}")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(
        handles=build_legend(categories, get_color),
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.95),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(results_dir / "pca_layers.png", dpi=200)