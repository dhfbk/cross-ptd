#!/bin/bash
#SBATCH --job-name=cross-domain
#SBATCH --output=cross-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

# Usage: sbatch slurm/run_vllm.sh <prompt_file> <model> [split]
#   e.g. sbatch slurm/run_vllm.sh Prompts/CoT_Prompt.txt meta-llama/Llama-3.3-70B-Instruct test
# Models used: meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen2.5-72B-Instruct
# Requires HF_TOKEN exported in your environment (gated models).

PROMPT_FILE=${1:?prompt file required}
MODEL=${2:?model required}
SPLIT=${3:-test}   # train | development | test

source "$PROMPT_FILE"   # defines SYSTEM_PROMPT and USER_TEMPLATE

# CoT: longer generation + {..} answer parsing; zero/few-shot: short answers
if [[ "$PROMPT_FILE" == *CoT* ]]; then
  SCRIPT=scripts/run_llm_CoT.py; MAX_TOKENS=1024
else
  SCRIPT=scripts/run_llm_eval.py; MAX_TOKENS=32
fi

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_COMPILE_MODE=none
export VLLM_DISABLE_COMPILE=1
export VLLM_USE_CUDAGRAPH=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

${SLURM_JOB_ID:+srun} python "$SCRIPT" \
  --model "$MODEL" \
  --quantization bitsandbytes \
  --system-prompt "$SYSTEM_PROMPT" \
  --user-template "$USER_TEMPLATE" \
  --enforce-eager \
  --compilation-mode 0 \
  --batch-size 1 \
  --max-model-len 8192 \
  --max-tokens "$MAX_TOKENS" \
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size 2 \
  --seed 69 \
  --split "$SPLIT" \
  --domains all \
  --use-chat-template
