#!/usr/bin/env bash

set -e

BASE_DIR=~/Desktop/hf_models

echo "Starting MLX servers ..."

# ----------------------------
# 7B BIG MODEL
# ----------------------------
echo "Starting BIG model (port 8000) ..."
nohup mlx_lm.server \
  --model $BASE_DIR/qwen_7b \
  --port 8000 \
  > big.log 2>&1 &

echo $! > big.pid

# ----------------------------
# CODE MODEL
# ----------------------------
echo "Starting CODE model (port 8001) ..."
nohup mlx_lm.server \
  --model $BASE_DIR/code_1b \
  --port 8001 \
  > code.log 2>&1 &

echo $! > code.pid

# ----------------------------
# MATH MODEL
# ----------------------------
echo "Starting MATH model (port 8002) ..."
nohup mlx_lm.server \
  --model $BASE_DIR/math_1b \
  --port 8002 \
  > math.log 2>&1 &

echo $! > math.pid

# ----------------------------
# LOGIC / GENERAL MODEL
# ----------------------------
echo "Starting LOGIC model (port 8003) ..."
nohup mlx_lm.server \
  --model $BASE_DIR/general_1b \
  --port 8003 \
  > logic.log 2>&1 &

echo $! > logic.pid

echo "All MLX servers started."
echo "Logs : big.log, code.log, math.log, logic.log"