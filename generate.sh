#!/bin/bash
#SBATCH --job-name=0239_ai_agent_dialogue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=18
#SBATCH --output=eval-qwen3-32b-%j.out
#SBATCH --error=eval-qwen3-32b-%j.err

cd /data/experiments/0239_ai_agent_dialogue/tool_use_ja
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 

MODEL=""
PORT=8000
parser=""

# vLLMサーバーをバックグラウンド起動
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL --port $PORT \
    --enable-auto-tool-choice --tool-call-parser $parser \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    > server.log 2>&1 &
SERVER_PID=$!

# 起動待機（簡易）
sleep 240

# 問い合わせ
python generate.py

# サーバー停止
kill $SERVER_PID