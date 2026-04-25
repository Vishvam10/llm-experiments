import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from analyze import (
    compute_distances,
    compute_multi_embedding,
    drop_points_from_multi_result,
)
from data import PROMPTS, get_context
from model_utils import get_hidden_states, load_model
from plot import build_colormap, plot_distance, plot_multi_embedding
from utils import categories, get_category


def get_model_name(model_path: str) -> str:
    name = Path(model_path).name
    return name if name else model_path.split("/")[-1]


###############################################################################
# GENERATION
###############################################################################


def generate_text(prompt, model, tokenizer, device, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip()


###############################################################################
# SIMPLE EVALUATION
###############################################################################


def extract_number(text):
    last_line = text.strip().split("\n")[-1]
    matches = re.findall(r"-?\d+\.?\d*", last_line)
    if matches:
        return float(matches[-1])

    matches = re.findall(r"-?\d+\.?\d*", text)
    return float(matches[-1]) if matches else None


def grade_output(output, answer, grader):
    output = output.strip().lower()

    if grader == "exact_number":
        val = extract_number(output)
        return val is not None and abs(val - float(answer)) < 1e-6

    if grader == "approx_number":
        val = extract_number(output)
        return val is not None and abs(val - float(answer)) < 1e-2

    if grader == "contains":
        return answer.lower() in output

    if grader == "number_in_text":
        val = extract_number(output)
        return val is not None and abs(val - float(answer)) < 1e-2

    return False


###############################################################################
# SERIALIZATION
###############################################################################


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


###############################################################################
# ARGPARSE
###############################################################################

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
outputs = []

###############################################################################
# DATA COLLECTION
###############################################################################

for item in PROMPTS:
    task = item["task"]
    prompt = item["prompt"]
    answer = item["answer"]
    grader = item["grader"]

    ctx_prompt = get_context(task) + "\n" + prompt

    no_ctx.append(get_hidden_states(prompt, model, tokenizer, device))
    ctx.append(get_hidden_states(ctx_prompt, model, tokenizer, device))

    task_labels.append(task)

    out_no_ctx = generate_text(prompt, model, tokenizer, device)
    out_ctx = generate_text(ctx_prompt, model, tokenizer, device)

    score_no_ctx = grade_output(out_no_ctx, answer, grader)
    score_ctx = grade_output(out_ctx, answer, grader)

    outputs.append(
        {
            "task": task,
            "prompt": prompt,
            "no_ctx_output": out_no_ctx,
            "ctx_output": out_ctx,
            "no_ctx_correct": bool(score_no_ctx),
            "ctx_correct": bool(score_ctx),
        }
    )

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
    outputs=outputs,
)

###############################################################################
# PLOTTING
###############################################################################

get_color = build_colormap(categories)

plot_distance(distances, args.method, args.dim, RESULTS_DIR)

plot_multi_embedding(
    layers=layer_embeddings,
    results_dir=RESULTS_DIR,
    get_category=get_category,
    categories=categories,
    get_color=get_color,
)

###############################################################################
# SAVE RESULTS
###############################################################################

accuracy_no_ctx = np.mean([o["no_ctx_correct"] for o in outputs])
accuracy_ctx = np.mean([o["ctx_correct"] for o in outputs])

summary = {
    "model": model_name,
    "method": args.method,
    "dim": args.dim,
    "num_layers": int(num_layers),
    "layer_indices": layer_indices.tolist(),
    "embedding_layers": drop_points_from_multi_result(layer_embeddings),
    "accuracy": {
        "no_context": float(accuracy_no_ctx),
        "with_context": float(accuracy_ctx),
    },
    "samples": outputs,
}

points_dump = {
    "model": model_name,
    "method": args.method,
    "dim": args.dim,
    "layers": layer_embeddings,
    "distance_per_layer": [float(x) for x in distances],
    "avg_distance": float(np.mean(distances)),
}

with open(RESULTS_DIR / "results.json", "w") as f:
    json.dump(summary, f, indent=2, default=to_serializable)

with open(RESULTS_DIR / "points.json", "w") as f:
    json.dump(points_dump, f, indent=2, default=to_serializable)

print(f"Saved all outputs to : {RESULTS_DIR}")
