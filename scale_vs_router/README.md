# LLM Routing Experiment (MLX setup)

This is a simple setup to compare:

- 1 big model (7B)
- 3 small specialist models (code / math / logic)
- routing vs voting vs single model

Everything runs locally using MLX servers.

## Setup

### 1. Install dependencies

```bash
uv add mlx-lm hf-transfer datasets requests matplotlib catppuccin
```

### 2. Download models

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download mlx-community/Qwen2.5-7B-Instruct-4bit \
  --local-dir ~/Desktop/Models/Qwen2.5-7B-Instruct-4bit

hf download mlx-community/Qwen2.5-Math-1.5B-4bit \
  --local-dir ~/Desktop/Models/Qwen2.5-Math-1.5B-4bit

hf download mlx-community/deepseek-coder-1.3b-instruct-4bit \
  --local-dir ~/Desktop/Models/deepseek-coder-1.3b-instruct-4bit

hf download mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --local-dir ~/Desktop/Models/Qwen2.5-1.5B-Instruct-4bit
```

### Start MLX servers

#### Big model (7B)

```bash
mlx_lm.server --model ~/Desktop/Models/Qwen2.5-7B-Instruct-4bit --port 8000
```

#### Small models
`
```bash
mlx_lm.server --model ~/Desktop/Models/deepseek-coder-1.3b-instruct-4bit --port 8001
mlx_lm.server --model ~/Desktop/Models/Qwen2.5-Math-1.5B-4bit --port 8002
mlx_lm.server --model ~/Desktop/Models/Qwen2.5-1.5B-Instruct-4bit --port 8003
```

Keep these running during evaluation.

## Run experiments

```bash
python eval.py --mode big
python eval.py --mode routed
python eval.py --mode vote
```

## Plot results

```bash
python plot.py
```