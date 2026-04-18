import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import catppuccin
from catppuccin.extras.matplotlib import get_colormap_from_list

# ------------------------------------------------------------------------------
# THEME
# ------------------------------------------------------------------------------
mpl.style.use(catppuccin.PALETTE.macchiato.identifier)

cmap = get_colormap_from_list(
    catppuccin.PALETTE.macchiato.identifier,
    ["blue", "mauve", "green"]
)

# ------------------------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------------------------
with open("results.json", "r") as f:
    data = json.load(f)

modes = list(data.keys())
scores = [data[m] for m in modes]

# ------------------------------------------------------------------------------
# PLOT
# ------------------------------------------------------------------------------
plt.figure(figsize=(7, 4))

bars = plt.bar(
    modes,
    scores,
    color=[cmap(i / len(modes)) for i in range(len(modes))]
)

plt.ylabel("Accuracy")
plt.title("LLM Routing vs Scaling (MLX Experiment)")

plt.ylim(0, 1.0)

# value labels
for bar, val in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.01,
        f"{val:.3f}",
        ha="center",
        fontsize=10
    )

plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()