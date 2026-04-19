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

    for p in data["points"]:
        marker = "o" if p["context"] == "no_ctx" else "x"
        color = get_color(get_category(p["task"]))

        ax.scatter(p["x"], p["y"], p["z"], marker=marker, color=color)

    ax.set_title(f"{data['method'].upper()}-3D Layer {data['layer']}")

    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=200)
    plt.close()


def plot_embedding(
    data, filename, results_dir, get_category, categories, get_color
):
    if data["dim"] == 2:
        plot_embedding_2d(
            data, filename, results_dir, get_category, categories, get_color
        )
    else:
        plot_embedding_3d(
            data, filename, results_dir, get_category, categories, get_color
        )


def plot_multi_embedding(
    layers, results_dir, get_category, categories, get_color
):
    for layer in layers:
        plot_embedding(
            layer,
            filename=f"{layer['method']}_{layer['dim']}d_layer_{layer['layer']}.png",
            results_dir=results_dir,
            get_category=get_category,
            categories=categories,
            get_color=get_color,
        )
