"""
Expands the hand-written seed examples into a larger synthetic training set
using few-shot prompting against the Anthropic API.

Uses Claude's tool-use (function calling) with a schema generated directly
from the GeneratedTrainingExample Pydantic model, so Claude is forced to
return output that structurally matches the schema -- categories, ratings,
and mitigation_status can only be one of the allowed enum values, since
Claude is filling in a tool's input schema rather than free-generating JSON
text. This removes almost all of the "invalid JSON" / "wrong enum value"
failure modes you'd get from just prompting a model to output raw JSON.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_dataset.py --n 400 --out data/train.jsonl

Design notes:
- Seeds are used as few-shot exemplars, sampled randomly per generation call,
  so the model sees varied style/category/severity combinations each time.
- Each generation call asks for ONE new narrative+output pair at a time rather
  than a batch, to keep quality high and make validation/retry simple.
- We still re-validate with Pydantic on receipt (belt-and-suspenders) and
  retry on failure, since tool-use enforces structure but not semantic
  correctness (e.g. Claude could still put "Critical" on a trivial finding
  if not steered by the target_rating constraint in the prompt).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from models import GeneratedTrainingExample, RiskCategory, RiskRating, MitigationStatus
from pydantic import ValidationError

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
FEW_SHOT_K = 4  # how many seed examples to show per generation call

RISK_CATEGORIES = [c.value for c in RiskCategory]
RISK_RATINGS = [r.value for r in RiskRating]
MITIGATION_STATUSES = [m.value for m in MitigationStatus]

SYSTEM_PROMPT = """You are a GRC analyst assistant generating realistic training \
data. Given constraints on category, rating, and mitigation status, invent an \
original, realistic ISO 27001 audit/risk narrative (2-5 sentences, specific \
details like numbers/systems/timeframes) and call the risk_training_example \
tool with the narrative plus its structured extraction. Set `confidence` to \
reflect how clearly the narrative supports the extracted fields (lower \
confidence for ambiguous or borderline narratives).
"""

# Tool definition built directly from the Pydantic model's JSON schema --
# this is the mechanism that forces structurally valid output.
TRAINING_EXAMPLE_TOOL = {
    "name": "risk_training_example",
    "description": "Record a synthetic risk narrative paired with its structured risk register extraction.",
    "input_schema": GeneratedTrainingExample.model_json_schema(),
}


def load_seeds(path: str) -> list[dict]:
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def build_prompt(seeds: list[dict], target_category: str, target_rating: str, target_status: str) -> str:
    examples_text = "\n\n".join(
        f"Narrative: {s['narrative']}\nJSON: {json.dumps(s['output'])}"
        for s in random.sample(seeds, min(FEW_SHOT_K, len(seeds)))
    )
    return f"""Here are examples of risk narratives paired with structured risk register JSON:

{examples_text}

Now invent ONE new, original ISO 27001 audit/risk narrative and call the \
risk_training_example tool with it and its extraction.
Requirements for this specific example:
- risk_category must be: {target_category}
- risk_rating must be: {target_rating}
- mitigation_status must be: {target_status}
- Do NOT reuse wording, topic, or scenario from the examples above -- invent a distinct, realistic scenario.
"""


def generate_one(client, seeds, target_category, target_rating, target_status) -> dict | None:
    prompt = build_prompt(seeds, target_category, target_rating, target_status)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                tools=[TRAINING_EXAMPLE_TOOL],
                tool_choice={"type": "tool", "name": "risk_training_example"},
                messages=[{"role": "user", "content": prompt}],
            )

            tool_use_block = next(b for b in resp.content if b.type == "tool_use")

            # Pydantic re-validation as a second layer of defense on top of
            # the tool-use schema enforcement
            validated = GeneratedTrainingExample.model_validate(tool_use_block.input)

            return {
                "narrative": validated.narrative,
                "output": validated.entry.model_dump(),
            }

        except (ValidationError, StopIteration) as e:
            print(f"  retry {attempt + 1}/{MAX_RETRIES} failed: {e}", file=sys.stderr)
            time.sleep(1)
        except Exception as e:
            print(f"  retry {attempt + 1}/{MAX_RETRIES} failed (API error): {e}", file=sys.stderr)
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="data/seed_examples.jsonl")
    parser.add_argument("--out", default="data/train.jsonl")
    parser.add_argument("--n", type=int, default=400, help="number of synthetic examples to generate")
    args = parser.parse_args()

    seeds = load_seeds(args.seeds)
    print(f"Loaded {len(seeds)} seed examples", flush=True)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated, failed = 0, 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5  # avoid getting stuck forever on one bad combo

    with open(out_path, "w") as f:
        while generated < args.n:
            # cycle target category/rating/status so the dataset stays balanced
            # rather than letting the model drift toward its own defaults
            target_category = RISK_CATEGORIES[generated % len(RISK_CATEGORIES)]
            target_rating = RISK_RATINGS[generated % len(RISK_RATINGS)]
            target_status = MITIGATION_STATUSES[generated % len(MITIGATION_STATUSES)]

            example = generate_one(client, seeds, target_category, target_rating, target_status)
            if example is None:
                failed += 1
                consecutive_failures += 1
                print(f"  [{generated}/{args.n}] failed on {target_category}/{target_rating}/{target_status} "
                      f"({consecutive_failures} consecutive failures)")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"  Too many consecutive failures -- skipping this slot and moving on.")
                    generated += 1  # give up on this one, move to the next target combo
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            f.write(json.dumps(example) + "\n")
            generated += 1
            if generated % 25 == 0:
                print(f"  {generated}/{args.n} generated ({failed} failed/skipped)", flush=True)

    print(f"Done. Wrote {generated} examples to {out_path} ({failed} failed and were skipped).")


if __name__ == "__main__":
    main()
