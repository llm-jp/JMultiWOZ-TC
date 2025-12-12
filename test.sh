#!/bin/bash
#SBATCH --job-name=0239_ai_agent_dialogue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd /data/experiments/0239_ai_agent_dialogue/tool_use_ja_test
source .venv/bin/activate

export HF_HOME='data/experiments/0239_ai_agent_dialogue'

MODEL="Qwen/Qwen3-32B"
PORT=8000
parser="hermes"

# vLLMサーバーをバックグラウンド起動
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL --port $PORT --enable-auto-tool-choice --tool-call-parser $parser > server.log 2>&1 &
SERVER_PID=$!

# 起動待機（簡易）
sleep 120

# 問い合わせ
python test.py

# サーバー停止
kill $SERVER_PID