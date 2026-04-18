import json
import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

from model_utils import load_model, get_hidden_states
from data import PROMPTS, get_context
from analyze import (
    compute_distances,
    compute_pca,
    compute_multi_pca,
    drop_points_from_pca,
    drop_points_from_multi,
)
from plot import plot_distance, plot_pca, plot_multi
from utils import categories, get_category


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
RESULTS_DIR = Path("results") / timestamp
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--num-pca-layers", type=int, default=8)
args = parser.parse_args()


model, tokenizer, device = load_model(args.model)

no_ctx, ctx, task_labels = [], [], []

for task, prompt in PROMPTS:
    no_ctx.append(get_hidden_states(prompt, model, tokenizer, device))

    ctx_prompt = get_context(task) + "\n" + prompt
    ctx.append(get_hidden_states(ctx_prompt, model, tokenizer, device))

    task_labels.append(task)

no_ctx = np.stack(no_ctx)
ctx = np.stack(ctx)

num_layers = no_ctx.shape[1]

###############################################################################
# ANALYSIS
###############################################################################

distances = compute_distances(no_ctx, ctx)

layer_indices = np.linspace(0, num_layers - 1, args.num_pca_layers, dtype=int)

pca_last = compute_pca(
    no_ctx,
    ctx,
    task_labels,
    layer_idx=num_layers - 1,
)

pca_layers = compute_multi_pca(
    no_ctx,
    ctx,
    task_labels,
    layer_indices=layer_indices,
)

###############################################################################
# PLOTTING
###############################################################################

plot_distance(distances, RESULTS_DIR)

plot_pca(
    pca_data=pca_last,
    filename="pca_last_layer.png",
    results_dir=RESULTS_DIR,
    get_category=get_category,
    categories=categories,
)

plot_multi(
    pca_layers=pca_layers,
    results_dir=RESULTS_DIR,
    get_category=get_category,
    categories=categories,
)


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


results = {
    "model": args.model,
    "num_layers": int(num_layers),
    "distance_per_layer": [float(x) for x in distances],
    "avg_distance": float(np.mean(distances)),
    "layer_indices": layer_indices.tolist(),
    "pca_last_layer": drop_points_from_pca(pca_last),
    "pca_layers": drop_points_from_multi(pca_layers),
    "prompts": [{"task": t, "prompt": p} for t, p in PROMPTS],
}

with open(RESULTS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=to_serializable)

print(f"Saved all outputs to : {RESULTS_DIR}")
