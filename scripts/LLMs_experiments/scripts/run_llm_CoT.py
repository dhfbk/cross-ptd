import argparse
import csv
import json
import os
import re
from collections import Counter

CANONICAL_LABELS = [
    "Loaded Language",
    "Name Calling",
    "Appeal to Fear",
    "Flag Waving",
    "Bandwagon",
    "Appeal to Authority",
    "Slogan",
]

DATASET_TO_CANONICAL = {
    "Loaded-language": "Loaded Language",
    "Name-calling-or-labeling": "Name Calling",
    "Appeal-to-fear": "Appeal to Fear",
    "Appeal-to-authority": "Appeal to Authority",
    "Flag-waving": "Flag Waving",
    "Bandwagon": "Bandwagon",
    "Slogan": "Slogan",
}

LABEL_ALIASES = {
    "Loaded Language": [
        "loaded language",
        "loaded-language",
        "loadedlanguage",
    ],
    "Name Calling": [
        "name calling",
        "name-calling",
        "name calling or labeling",
        "name calling or labelling",
        "name-calling-or-labeling",
        "name-calling or labeling",
        "name-calling or labelling",
    ],
    "Appeal to Fear": [
        "appeal to fear",
        "appeal-to-fear",
        "fear appeal",
    ],
    "Flag Waving": [
        "flag waving",
        "flag-waving",
        "flagwaving",
    ],
    "Bandwagon": [
        "bandwagon",
    ],
    "Appeal to Authority": [
        "appeal to authority",
        "appeal-to-authority",
    ],
    "Slogan": [
        "slogan",
        "slogans",
    ],
}


def sanitize_model_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    validate_prompt_template(template, "Prompt template missing {INPUT_TEXT} placeholder: %s" % path)
    return template


def validate_prompt_template(template, error_message):
    if "{INPUT_TEXT}" not in template:
        raise ValueError(error_message)
    return template


def render_prompt(template, input_text, use_chat_template, system_prompt, tokenizer):
    user_text = template.replace("{INPUT_TEXT}", input_text)
    if not use_chat_template:
        if system_prompt:
            return system_prompt.strip() + "\n\n" + user_text
        return user_text
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_gold(label_str):
    label_str = (label_str or "").strip()
    if not label_str:
        return [], []
    raw = [part.strip() for part in label_str.split("|") if part.strip()]
    labels = []
    ignored = []
    for item in raw:
        canonical = DATASET_TO_CANONICAL.get(item)
        if canonical:
            if canonical not in labels:
                labels.append(canonical)
        else:
            ignored.append(item)
    return labels, ignored


def parse_prediction(text):
    text = (text or "").strip()
    if not text:
        return []

    # Primary path: extract the last {...} block — the delimited final answer.
    # The prompt instructs the model to wrap its conclusion in {Technique1, Technique2}
    # or {None}. We take the last occurrence to skip any {}-like content in spans.
    brace_matches = re.findall(r'\{([^}]*)\}', text)
    if brace_matches:
        content = brace_matches[-1].strip()
        if not content or content.lower() == "none":
            return []
        found = set()
        for part in re.split(r'[,\n]', content):
            normalized = normalize_label(part.strip())
            if normalized:
                found.add(normalized)
        if found:
            return [label for label in CANONICAL_LABELS if label in found]
        return []

    # Fallback 1: entire output is a JSON list (original zero-shot format).
    try:
        data = json.loads(text)
        if isinstance(data, list):
            found = set()
            for item in data:
                if not isinstance(item, str):
                    continue
                normalized = normalize_label(item)
                if normalized:
                    found.add(normalized)
            if found:
                return [label for label in CANONICAL_LABELS if label in found]
            if any(isinstance(item, str) and item.strip().lower() == "none" for item in data):
                return []
    except Exception:
        pass

    # Fallback 2: scan for alias matches anywhere in the text.
    # NOTE: this is unreliable when the model writes all technique names in its
    # reasoning (as in CoT outputs). It is kept only as a last resort.
    found = set()
    lowered = text.lower()
    for label in CANONICAL_LABELS:
        for alias in LABEL_ALIASES[label]:
            if re.search(r"\b" + re.escape(alias) + r"\b", lowered):
                found.add(label)
                break
    if found:
        return [label for label in CANONICAL_LABELS if label in found]
    if re.search(r"\bnone\b", lowered):
        return []
    return []


def normalize_label(text):
    text = text.strip().lower()
    if text == "none":
        return None
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if text == alias:
                return label
    return None


class MetricsAggregator:
    def __init__(self, labels):
        self.labels = labels
        self.tp = {label: 0 for label in labels}
        self.fp = {label: 0 for label in labels}
        self.fn = {label: 0 for label in labels}
        self.support = {label: 0 for label in labels}
        self.examples = 0
        self.exact_match = 0
        self.ignored_labels = Counter()

    def update(self, gold_labels, pred_labels, ignored_labels):
        gold = set(gold_labels)
        pred = set(pred_labels)
        self.examples += 1
        if gold == pred:
            self.exact_match += 1
        for label in self.labels:
            if label in gold:
                self.support[label] += 1
            if label in pred and label in gold:
                self.tp[label] += 1
            elif label in pred and label not in gold:
                self.fp[label] += 1
            elif label not in pred and label in gold:
                self.fn[label] += 1
        for item in ignored_labels:
            self.ignored_labels[item] += 1

    def _safe_div(self, num, denom):
        return num / denom if denom else 0.0

    def summary(self):
        per_label = {}
        for label in self.labels:
            tp = self.tp[label]
            fp = self.fp[label]
            fn = self.fn[label]
            precision = self._safe_div(tp, tp + fp)
            recall = self._safe_div(tp, tp + fn)
            f1 = self._safe_div(2 * precision * recall, precision + recall)
            per_label[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": self.support[label],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        macro_precision = sum(v["precision"] for v in per_label.values()) / len(self.labels)
        macro_recall = sum(v["recall"] for v in per_label.values()) / len(self.labels)
        macro_f1 = sum(v["f1"] for v in per_label.values()) / len(self.labels)
        micro_tp = sum(self.tp.values())
        micro_fp = sum(self.fp.values())
        micro_fn = sum(self.fn.values())
        micro_precision = self._safe_div(micro_tp, micro_tp + micro_fp)
        micro_recall = self._safe_div(micro_tp, micro_tp + micro_fn)
        micro_f1 = self._safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        return {
            "examples": self.examples,
            "exact_match": self._safe_div(self.exact_match, self.examples),
            "exact_match_count": self.exact_match,
            "per_label": per_label,
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
            "ignored_labels": self.ignored_labels.most_common(),
        }


def iter_dataset_rows(path, max_samples):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if max_samples is not None and idx >= max_samples:
                break
            yield row


def chunked(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run_inference(args, dataset_path, output_dir, llm, sampling_params, prompt_template, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.jsonl")

    metrics = MetricsAggregator(CANONICAL_LABELS)
    with open(predictions_path, "w", encoding="utf-8") as pred_f:
        batch = []
        for row in iter_dataset_rows(dataset_path, args.max_samples):
            prompt = render_prompt(
                prompt_template,
                row["text"],
                args.use_chat_template,
                args.system_prompt,
                tokenizer,
            )
            gold_labels, ignored = parse_gold(row.get("labels", ""))
            batch.append(
                {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "gold_labels": gold_labels,
                    "ignored": ignored,
                    "prompt": prompt,
                    "gold_raw": row.get("labels", ""),
                }
            )
            if len(batch) >= args.batch_size:
                process_batch(batch, llm, sampling_params, metrics, pred_f, args)
                batch = []

        if batch:
            process_batch(batch, llm, sampling_params, metrics, pred_f, args)

    return metrics


def process_batch(batch, llm, sampling_params, metrics, pred_f, args):
    prompts = [item["prompt"] for item in batch]
    outputs = llm.generate(prompts, sampling_params)
    if len(outputs) != len(batch):
        raise RuntimeError("Unexpected output size: %d != %d" % (len(outputs), len(batch)))

    for item, output in zip(batch, outputs):
        raw_text = ""
        if output.outputs:
            raw_text = output.outputs[0].text
        pred_labels = parse_prediction(raw_text)
        metrics.update(item["gold_labels"], pred_labels, item["ignored"])
        record = {
            "id": item["id"],
            "text": item["text"],
            "gold_raw": item["gold_raw"],
            "gold": item["gold_labels"],
            "pred": pred_labels,
            "pred_raw": raw_text,
        }
        pred_f.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_domains(value):
    if value.strip().lower() == "all":
        return ["news", "social-fb", "transcripts"]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_stop(value):
    if value is None:
        return None
    stops = [item.strip() for item in value.split(",") if item.strip()]
    return stops if stops else None


def resolve_prompt_template(args):
    if args.prompt is not None:
        return validate_prompt_template(
            args.prompt,
            "Prompt missing {INPUT_TEXT} placeholder from --prompt.",
        )
    if args.user_template is not None:
        return validate_prompt_template(
            args.user_template,
            "User template missing {INPUT_TEXT} placeholder from --user-template.",
        )
    if args.prompt_path is not None:
        return load_prompt(args.prompt_path)
    raise SystemExit("Provide a prompt via --prompt, --user-template, or --prompt-path.")


def main():
    parser = argparse.ArgumentParser(
        description="Run CoT LLM inference on cross-data and compute metrics."
    )
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument("--prompt-path", default=None, help="Path to a plain-text prompt template file containing {INPUT_TEXT}.")
    parser.add_argument("--prompt", default=None, help="Inline prompt template.")
    parser.add_argument("--system-prompt", default=None, help="System prompt for chat models.")
    parser.add_argument("--user-template", default=None, help="User template for chat models.")
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Apply the tokenizer chat template using system/user roles.",
    )
    parser.add_argument("--data-dir", default="cross-data", help="Dataset root directory.")
    parser.add_argument("--domains", default="all", help="Comma list or 'all'.")
    parser.add_argument("--split", default="test", help="Split: train, development, test.")
    parser.add_argument("--output-dir", default="outputs", help="Root output directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per split.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate.")
    parser.add_argument("--stop", default=None, help="Comma-separated stop strings.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for vLLM.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument(
        "--tokenizer-mode",
        default="auto",
        help='Tokenizer mode for vLLM (e.g., "auto", "mistral").',
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype for vLLM.")
    parser.add_argument(
        "--quantization",
        default=None,
        help="vLLM quantization mode (e.g., awq, gptq, bitsandbytes).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution (disables CUDA graphs).",
    )
    parser.add_argument(
        "--compilation-mode",
        type=int,
        default=None,
        help="vLLM compilation mode (0=NONE, 1=STOCK_TORCH_COMPILE, 2=DYNAMO_TRACE_ONCE, 3=VLLM_COMPILE).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model length.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use.",
    )
    args = parser.parse_args()

    args.stop = parse_stop(args.stop)
    domains = parse_domains(args.domains)

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("vLLM is required. Install with pip install vllm") from exc

    prompt_template = resolve_prompt_template(args)

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "tokenizer_mode": args.tokenizer_mode,
    }
    if args.seed is not None:
        llm_kwargs["seed"] = args.seed
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.compilation_mode is not None:
        llm_kwargs["compilation_config"] = args.compilation_mode
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop or None,
        seed=args.seed,
    )

    tokenizer = llm.get_tokenizer()
    if args.use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise SystemExit("Tokenizer does not support chat templates.")

    model_slug = sanitize_model_name(args.model)
    overall_metrics = {}
    all_metrics = []

    for domain in domains:
        dataset_path = os.path.join(args.data_dir, domain, "%s.tsv" % args.split)
        if not os.path.exists(dataset_path):
            raise SystemExit("Missing dataset file: %s" % dataset_path)
        output_dir = os.path.join(args.output_dir, model_slug, domain, args.split)
        metrics = run_inference(
            args,
            dataset_path,
            output_dir,
            llm,
            sampling_params,
            prompt_template,
            tokenizer,
        )
        all_metrics.append(metrics)
        summary = metrics.summary()
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        overall_metrics[domain] = summary

    if len(domains) > 1:
        combined = MetricsAggregator(CANONICAL_LABELS)
        for metrics in all_metrics:
            combined.examples += metrics.examples
            combined.exact_match += metrics.exact_match
            for label in CANONICAL_LABELS:
                combined.tp[label] += metrics.tp[label]
                combined.fp[label] += metrics.fp[label]
                combined.fn[label] += metrics.fn[label]
                combined.support[label] += metrics.support[label]
            combined.ignored_labels.update(metrics.ignored_labels)
        combined_summary = combined.summary()
        combined_path = os.path.join(args.output_dir, model_slug, "combined_%s.json" % args.split)
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(
                {"per_domain": overall_metrics, "combined": combined_summary},
                f,
                indent=2,
                ensure_ascii=True,
            )


if __name__ == "__main__":
    main()
