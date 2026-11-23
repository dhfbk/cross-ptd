#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

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


def parse_domains(value):
    if value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


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


def normalize_label(text):
    text = text.strip().lower()
    if text == "none":
        return None
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if text == alias:
                return label
    return None


def parse_prediction(text):
    text = (text or "").strip()
    if not text:
        return []

    # Primary path: extract the last {...} block — the delimited final answer.
    # The CoT prompt instructs the model to wrap its conclusion in {Technique1, Technique2}
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
    # NOTE: unreliable for CoT outputs where all technique names appear in reasoning.
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


def load_dataset(data_dir, domain, split, id_field):
    path = Path(data_dir) / domain / ("%s.tsv" % split)
    if not path.exists():
        raise SystemExit("Missing dataset file: %s" % path)
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit("Empty TSV file: %s" % path)
        if id_field not in reader.fieldnames:
            raise SystemExit("Missing '%s' column in %s" % (id_field, path))
        for row in reader:
            row_id = str(row.get(id_field, "")).strip()
            if row_id:
                rows[row_id] = row
    return rows


def extract_response_content(payload):
    response = payload.get("response") or {}
    if not isinstance(response, dict):
        return "", None, None, None
    body = response.get("body") or {}
    if not isinstance(body, dict):
        body = {}
    status = response.get("status_code")
    response_id = body.get("id")
    request_id = response.get("request_id")
    model = body.get("model")
    content = ""
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            content = message.get("content") or ""
    return content, status, response_id, model


def infer_model_from_files(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                _content, _status, _resp_id, model = extract_response_content(payload)
                if model:
                    return model
    return None


def parse_custom_id(value):
    parts = value.split("|", 2)
    if len(parts) != 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenAI batch output JSONL into predictions.jsonl files."
    )
    parser.add_argument(
        "--batch-output",
        nargs="+",
        default=[str(PROJECT_ROOT / "batch_output" / "gpt-5-mini-2025-08-07.jsonl")],
        help="One or more batch output JSONL paths.",
    )
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "cross-data"), help="Dataset root directory.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"), help="Root outputs directory.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--domains", default="all", help="Comma list or 'all'.")
    parser.add_argument("--id-field", default="id", help="ID column name in TSV.")
    parser.add_argument("--text-field", default="text", help="Text column name in TSV.")
    parser.add_argument("--labels-field", default="labels", help="Gold labels column.")
    parser.add_argument(
        "--predictions-name",
        default="predictions.jsonl",
        help="Predictions filename to write.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip records missing from the dataset instead of exiting.",
    )
    args = parser.parse_args()

    args.data_dir = str(resolve_path(args.data_dir, PROJECT_ROOT))
    args.output_dir = str(resolve_path(args.output_dir, PROJECT_ROOT))

    batch_paths = [resolve_path(path, PROJECT_ROOT) for path in args.batch_output]
    for path in batch_paths:
        if not path.exists():
            raise SystemExit("Missing batch output file: %s" % path)

    model = args.model or infer_model_from_files(batch_paths) or "openai-batch"
    model_slug = sanitize_model_name(model)

    domain_filter = parse_domains(args.domains)
    dataset_cache = {}
    output_handles = {}

    def get_dataset(domain):
        if domain not in dataset_cache:
            dataset_cache[domain] = load_dataset(
                args.data_dir, domain, args.split, args.id_field
            )
        return dataset_cache[domain]

    def get_output_handle(domain):
        handle = output_handles.get(domain)
        if handle:
            return handle
        output_dir = Path(args.output_dir) / model_slug / domain / args.split
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / args.predictions_name
        handle = path.open("w", encoding="utf-8")
        output_handles[domain] = handle
        return handle

    total = 0
    written = 0
    skipped = 0
    missing = 0
    truncated_empty = 0

    try:
        for path in batch_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        skipped += 1
                        continue
                    total += 1

                    custom_id = str(payload.get("custom_id", "")).strip()
                    domain, split, row_id = parse_custom_id(custom_id)
                    if not domain or not split or not row_id:
                        skipped += 1
                        continue
                    if split != args.split:
                        continue
                    if domain_filter and domain not in domain_filter:
                        continue

                    dataset = get_dataset(domain)
                    row = dataset.get(row_id)
                    if row is None:
                        missing += 1
                        if args.allow_missing:
                            continue
                        raise SystemExit("Missing id %s in %s/%s.tsv" % (row_id, domain, args.split))

                    text = str(row.get(args.text_field, "")).strip()
                    gold_raw = str(row.get(args.labels_field, "")).strip()
                    gold_labels, _ignored = parse_gold(gold_raw)

                    pred_raw, status, response_id, model_from_response = extract_response_content(payload)
                    if model_from_response and not args.model:
                        model = model_from_response
                        model_slug = sanitize_model_name(model)
                    if status == 200 and not pred_raw and payload.get("response", {}).get("body", {}).get("choices"):
                        finish_reason = payload["response"]["body"]["choices"][0].get("finish_reason")
                        if finish_reason == "length":
                            truncated_empty += 1

                    pred_labels = parse_prediction(pred_raw)
                    record = {
                        "id": row_id,
                        "text": text,
                        "gold_raw": gold_raw,
                        "gold": gold_labels,
                        "pred": pred_labels,
                        "pred_raw": pred_raw,
                    }
                    if response_id:
                        record["response_id"] = response_id
                    if status is not None:
                        record["status_code"] = status
                    record["custom_id"] = custom_id

                    handle = get_output_handle(domain)
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    written += 1
    finally:
        for handle in output_handles.values():
            handle.close()

    print("Done.")
    print("Model:", model)
    print("Total batch lines:", total)
    print("Written predictions:", written)
    print("Skipped:", skipped)
    print("Missing ids:", missing)
    print("Truncated empty responses:", truncated_empty)
    if truncated_empty:
        print(
            "Warning: Many responses ended with finish_reason=length and empty content. "
            "For reasoning models, increase max_completion_tokens in batch generation."
        )


if __name__ == "__main__":
    main()
