#!/usr/bin/env python3
import argparse
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


def parse_domains(value):
    if value.strip().lower() == "all":
        return ["news", "social-fb", "transcripts"]
    return [item.strip() for item in value.split(",") if item.strip()]


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


def coerce_label_list(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
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


def load_metrics_from_predictions(predictions_path):
    metrics = MetricsAggregator(CANONICAL_LABELS)
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            gold_raw = record.get("gold_raw") or record.get("labels") or ""
            gold_labels = coerce_label_list(record.get("gold"))
            ignored = []
            if gold_raw:
                _gold_from_raw, ignored = parse_gold(gold_raw)
                if gold_labels is None:
                    gold_labels = _gold_from_raw
            if gold_labels is None:
                gold_labels = []
            pred_labels = coerce_label_list(record.get("pred"))
            if pred_labels is None:
                pred_labels = parse_prediction(record.get("pred_raw", ""))
            metrics.update(gold_labels, pred_labels, ignored)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from existing predictions.jsonl files."
    )
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07", help="Model name or slug.")
    parser.add_argument("--output-dir", default="outputs", help="Root outputs directory.")
    parser.add_argument("--domains", default="all", help="Comma list or 'all'.")
    parser.add_argument("--split", default="test", help="Split: train, development, test.")
    parser.add_argument(
        "--predictions-name",
        default="predictions.jsonl",
        help="Predictions filename to read.",
    )
    args = parser.parse_args()

    model_slug = sanitize_model_name(args.model)
    domains = parse_domains(args.domains)

    overall_metrics = {}
    all_metrics = []

    for domain in domains:
        output_dir = os.path.join(args.output_dir, model_slug, domain, args.split)
        predictions_path = os.path.join(output_dir, args.predictions_name)
        if not os.path.exists(predictions_path):
            raise SystemExit("Missing predictions file: %s" % predictions_path)
        metrics = load_metrics_from_predictions(predictions_path)
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
        combined_path = os.path.join(
            args.output_dir, model_slug, "combined_%s.json" % args.split
        )
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(
                {"per_domain": overall_metrics, "combined": combined_summary},
                f,
                indent=2,
                ensure_ascii=True,
            )


if __name__ == "__main__":
    main()
