import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

import catppuccin
from catppuccin.extras.matplotlib import get_colormap_from_list

# ------------------------------------------------------------------------------
# ARGS
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--use-4bit", action="store_true")
parser.add_argument("--num-pca-layers", type=int, default=8)
args = parser.parse_args()

# ------------------------------------------------------------------------------
# THEME
# ------------------------------------------------------------------------------
mpl.style.use(catppuccin.PALETTE.macchiato.identifier)

cmap = get_colormap_from_list(
    catppuccin.PALETTE.macchiato.identifier,
    ["blue", "mauve", "green"]
)

# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------
PROMPTS = [
    ("math", "What is 23 * 47?"),
    ("math", "What is 144 divided by 12?"),
    ("code", "Write a Python function to reverse a list."),
    ("code", "Write a function to check if a number is prime."),
    ("logic", "If all bloops are razzies and some razzies are lazzies, are some bloops lazzies?"),
    ("logic", "If John is taller than Mary and Mary is taller than Sam, who is tallest?")
]

CONTEXT = "You must be extremely accurate and double-check your answer."

# ------------------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------------------
print("Loading model...")

model_kwargs = {
    "device_map": "auto",
    "output_hidden_states": True
}

if args.use_4bit:
    model_kwargs["load_in_4bit"] = True

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    **model_kwargs
)

tokenizer = AutoTokenizer.from_pretrained(args.model)

device = next(model.parameters()).device

# ------------------------------------------------------------------------------
# GET HIDDEN STATES
# ------------------------------------------------------------------------------
def get_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    hs = out.hidden_states

    reps = []
    for layer in hs:
        rep = layer.mean(dim=1).squeeze().detach().cpu().numpy()
        reps.append(rep)

    return np.stack(reps)

# ------------------------------------------------------------------------------
# COLLECT
# ------------------------------------------------------------------------------
no_ctx, ctx, task_labels = [], [], []

for task, prompt in PROMPTS:
    no_ctx.append(get_hidden_states(prompt))
    ctx.append(get_hidden_states(CONTEXT + "\n" + prompt))
    task_labels.append(task)

no_ctx = np.stack(no_ctx)
ctx = np.stack(ctx)

num_layers = no_ctx.shape[1]

# ------------------------------------------------------------------------------
# DISTANCE VS LAYER
# ------------------------------------------------------------------------------
print("Computing distance curve...")

distances = []

for l in range(num_layers):
    d = cosine_distances(no_ctx[:, l, :], ctx[:, l, :])
    distances.append(np.mean(np.diag(d)))

plt.figure()
plt.plot(range(num_layers), distances, marker="o")
plt.title("Context Influence Across Layers")
plt.xlabel("Layer")
plt.ylabel("Cosine Distance")
plt.tight_layout()
plt.savefig("distance.png", dpi=200)

# ------------------------------------------------------------------------------
# PCA (LAST LAYER)
# ------------------------------------------------------------------------------
print("Generating last-layer PCA...")

layer_idx = num_layers - 1

X = np.vstack([
    no_ctx[:, layer_idx, :],
    ctx[:, layer_idx, :]
])

labels_ctx = ["no_ctx"] * len(no_ctx) + ["ctx"] * len(ctx)
labels_task = task_labels + task_labels

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

plt.figure()

for i in range(len(X2)):
    marker = "o" if labels_ctx[i] == "no_ctx" else "x"

    plt.scatter(
        X2[i, 0],
        X2[i, 1],
        marker=marker,
        alpha=0.8
    )

plt.title("Representation Shift (Last Layer)")
plt.tight_layout()
plt.savefig("pca_last_layer.png", dpi=200)

# ------------------------------------------------------------------------------
# MULTI-LAYER PCA GRID
# ------------------------------------------------------------------------------
print("Generating multi-layer PCA grid...")

num_plots = args.num_pca_layers
layer_indices = np.linspace(0, num_layers - 1, num_plots, dtype=int)

cols = 4
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
axes = axes.flatten()

for idx, layer_idx in enumerate(layer_indices):
    ax = axes[idx]

    X = np.vstack([
        no_ctx[:, layer_idx, :],
        ctx[:, layer_idx, :]
    ])

    labels_ctx = ["no_ctx"] * len(no_ctx) + ["ctx"] * len(ctx)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    for i in range(len(X2)):
        marker = "o" if labels_ctx[i] == "no_ctx" else "x"

        ax.scatter(
            X2[i, 0],
            X2[i, 1],
            marker=marker,
            alpha=0.8
        )

    ax.set_title(f"Layer {layer_idx}")

# remove empty axes
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Representation Shift Across Layers", fontsize=14)
plt.tight_layout()
plt.savefig("pca_layers.png", dpi=200)

print("\nSaved:")
print(" - distance.png")
print(" - pca_last_layer.png")
print(" - pca_layers.png")