## Context Influence On LLMs

### Setup

#### 1. Install dependencies

```bash
uv add torch transformers accelerate bitsandbytes datasets matplotlib catppuccin
```

#### 2. Download models

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ~/Desktop/Models/Qwen2.5-1.5B-Instruct
```

### Run experiment

```bash
python script.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct
```

### Outputs

distance.png — context influence across layers
pca_last_layer.png — representation shift (final layer)
pca_layers.png — PCA across layers
results.json — raw data (distances + PCA points)