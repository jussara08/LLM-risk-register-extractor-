"""
Evaluates model outputs on a held-out test set against two things:
1. Structural validity: does the output parse as JSON matching our schema?
2. Field-level accuracy: for classification fields (category, likelihood,
   impact, rating, mitigation_status), does the predicted value match gold?

Run this against BOTH the base model (zero/few-shot prompted) and the
fine-tuned model to produce the baseline-vs-fine-tuned comparison table
that's the key result for the project writeup.

Usage:
    python scripts/evaluate.py --predictions outputs/finetuned_preds.jsonl --gold data/test.jsonl
"""

import argparse
import json

from schema_config import REQUIRED_OUTPUT_FIELDS, validate_output

CLASSIFICATION_FIELDS = ["risk_category", "likelihood", "impact", "risk_rating", "mitigation_status"]


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def try_parse(raw: str) -> dict | None:
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def evaluate(predictions: list[dict], gold: list[dict]) -> dict:
    assert len(predictions) == len(gold), "predictions and gold must be same length and aligned"

    n = len(gold)
    valid_json_count = 0
    schema_valid_count = 0
    field_correct = {field: 0 for field in CLASSIFICATION_FIELDS}
    field_total = {field: 0 for field in CLASSIFICATION_FIELDS}

    for pred_row, gold_row in zip(predictions, gold):
        parsed = pred_row.get("output") if isinstance(pred_row.get("output"), dict) else try_parse(pred_row.get("raw_output", ""))
        gold_output = gold_row["output"]

        if parsed is None:
            continue
        valid_json_count += 1

        errors = validate_output(parsed)
        if not errors:
            schema_valid_count += 1

        for field in CLASSIFICATION_FIELDS:
            field_total[field] += 1
            if parsed.get(field) == gold_output.get(field):
                field_correct[field] += 1

    results = {
        "n_examples": n,
        "valid_json_rate": round(valid_json_count / n, 3),
        "schema_valid_rate": round(schema_valid_count / n, 3),
        "field_accuracy": {
            field: round(field_correct[field] / field_total[field], 3) if field_total[field] else None
            for field in CLASSIFICATION_FIELDS
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="jsonl with {'output': {...}} or {'raw_output': '...'} per line")
    parser.add_argument("--gold", required=True, help="jsonl test set with {'narrative', 'output'} per line")
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    gold = load_jsonl(args.gold)

    results = evaluate(predictions, gold)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
