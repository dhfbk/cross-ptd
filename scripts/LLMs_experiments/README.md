# LLM experiments – persuasion technique detection

Scripts to reproduce the LLM experiments (zero-shot, few-shot, CoT) on 7 persuasion techniques across three domains (`news`, `social-fb`, `transcripts`).

## Setup

```bash
pip install -r requirements.txt
```

Put the dataset in `cross-data/<domain>/{train,development,test}.tsv` (columns: `id`, `text`, `labels`, labels separated by `|`).

## Prompts

`Prompts/*.txt` define `SYSTEM_PROMPT` and `USER_TEMPLATE` (`{INPUT_TEXT}` is replaced with the input text).

## Open-weight models (vLLM, SLURM)

```bash
export HF_TOKEN=...   # for gated models
sbatch slurm/run_vllm.sh Prompts/CoT_Prompt.txt meta-llama/Llama-3.3-70B-Instruct test
sbatch slurm/run_vllm.sh Prompts/FewShot_Prompt.txt Qwen/Qwen2.5-72B-Instruct test
```

CoT prompts use `scripts/run_llm_CoT.py`, all others use `scripts/run_llm_eval.py`.
Predictions and metrics are written to `outputs/<model>/<domain>/<split>/`.

## OpenAI models (Batch API)

1. `scripts/OpenAI_batches.ipynb`: build the batch `.jsonl` files (set model, prompt and generation settings in the first cells) and submit them.
2. Download the batch output, then:

```bash
python scripts/convert_batch_output_to_predictions.py --batch-output batch_output/<file>.jsonl --split test
python scripts/compute_metrics_from_predictions.py --model <model> --split test
```

## Error analysis

`scripts/fp_error_analysis.ipynb`: false-positive analysis across models (set `OUTPUTS_DIR`, `PROMPT`, `SPLIT`).
