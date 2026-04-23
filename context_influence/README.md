## Context Influence on LLMs

This project studies how adding context to a prompt changes an LLM’s internal representations, not just its final output.

For a set of tasks (math, code, logic, language, adversarial), each prompt is run twice : once as-is and once with structured context prepended. The model’s hidden states are extracted at every layer for both versions.

The core idea is to **measure how much the representations diverge across layers**. This is done using cosine distance, producing a layer-wise view of where context has the most influence. 

Additionally, **PCA and UMAP are used to project these representations into 2D/3D to visualize how prompts cluster and how context shifts** them.

### Setup

#### 1. Install dependencies

```bash
uv add torch transformers accelerate bitsandbytes datasets matplotlib catppuccin umap-learn scikit-learn
```

#### 2. Download model

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ~/Desktop/Models/Qwen2.5-1.5B-Instruct
```

You can replace this with any local path or Hugging Face model ID.

#### Run experiment

```bash
# PCA (deterministic, preserves global structure)
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method pca --dim 2
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method pca --dim 3

# UMAP (non-linear, better local clustering)
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method umap --dim 2
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method umap --dim 3
```

#### Output


Each run creates a timestamped directory :

```bash
results/<model_name>:<timestamp>/
```

Containing :

```md
- distance.png — cosine distance across layers (context vs no context)
- <method>_<dim>d_layers_combined.png — embeddings across selected layers
- results.json — metadata and aggregated metrics
```

> [!NOTE]
> - `PCA` is more reliable for comparing geometry across runs. 
> - `UMAP` is useful for visual cluster separation but may distort distances. 
> - Results depend on prompt set and context design, not just the model.