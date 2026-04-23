import json
import argparse
import numpy as np

from datetime import datetime
from pathlib import Path

from model_utils import load_model, get_hidden_states
from data import PROMPTS, get_context
from analyze import (
    compute_distances,
    compute_multi_embedding,
    drop_points_from_multi_result,
)
from plot import (
    build_colormap,
    plot_distance,
    plot_multi_embedding
)
from utils import categories, get_category

def get_model_name(model_path: str) -> str:
    name = Path(model_path).name
    return name if name else model_path.split("/")[-1]


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--num-pca-layers", type=int, default=8)
parser.add_argument(
    "--method", type=str, default="umap", choices=["pca", "umap"]
)
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
model_name = get_model_name(args.model).lower()

RESULTS_DIR = Path("results") / model_name / timestamp
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

layer_embeddings = compute_multi_embedding(
    no_ctx,
    ctx,
    task_labels,
    layer_indices=layer_indices,
    method=args.method,
    dim=args.dim,
)

###############################################################################
# PLOTTING
###############################################################################

get_color = build_colormap(categories)

plot_distance(distances, RESULTS_DIR)

plot_multi_embedding(
    layers=layer_embeddings,
    results_dir=RESULTS_DIR,
    get_category=get_category,
    categories=categories,
    get_color=get_color,
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
    "method": args.method,
    "dim": args.dim,
    "num_layers": int(num_layers),
    "distance_per_layer": [float(x) for x in distances],
    "avg_distance": float(np.mean(distances)),
    "layer_indices": layer_indices.tolist(),
    "embedding_layers": drop_points_from_multi_result(layer_embeddings),
    "prompts": [{"task": t, "prompt": p} for t, p in PROMPTS],
}

with open(RESULTS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=to_serializable)

print(f"Saved all outputs to : {RESULTS_DIR}")
