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
# Using PCA
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method pca --dim 2
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method pca --dim 3

# Using UMAP
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method umap --dim 2
python main.py --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct --method umap --dim 3
```