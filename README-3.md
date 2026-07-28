# Risk Narrative → Structured Risk Register (Fine-Tuned LLM)

Fine-tunes a small open-source LLM (Llama 3.2, via QLoRA) to convert unstructured risk and audit narratives into structured, schema-consistent risk register entries mapped to ISO 27001 controls.

## Pipeline

```
data/seed_examples.jsonl   -- hand-written, hand-labeled seed examples (style guide)
        |
scripts/generate_dataset.py -- few-shot expansion into a larger synthetic dataset
        |
data/train.jsonl           -- synthetic training data (schema-validated)
        |
scripts/train.py           -- QLoRA fine-tuning (needs GPU)
        |
models/risk-register-lora/ -- trained LoRA adapter
        |
scripts/evaluate.py        -- baseline vs fine-tuned comparison
app.py                      -- Streamlit demo (single + batch extraction)
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Generate the synthetic dataset

Expands the 25 seed examples in `data/seed_examples.jsonl` into a larger balanced dataset. Uses Claude's tool-use (function calling) with a schema generated directly from a Pydantic model (`scripts/models.py`), so every generated example is *structurally* guaranteed to match the schema — categories, ratings, and mitigation status can only ever be one of the allowed enum values.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_dataset.py --n 400 --out data/train.jsonl
```

## 2. Fine-tune

Requires GPU access (Colab, cloud GPU instance, or local GPU with ≥16GB VRAM).

```bash
python scripts/train.py --data data/train.jsonl --output_dir models/risk-register-lora --epochs 3
```

## 3. Evaluate

Compares structural validity and field-level accuracy between the fine-tuned model and a prompted-only baseline on a held-out test set.

```bash
python scripts/evaluate.py --predictions outputs/finetuned_preds.jsonl --gold data/test.jsonl
```

## 4. Run the demo

```bash
streamlit run app.py
```

Supports single-narrative extraction and batch CSV processing (upload a CSV with a `narrative` column, get back a populated risk register CSV).

## 5. Deploy a public demo (optional)

`app_cloud.py` is a lightweight version that uses the Claude API (schema-enforced via tool-use, same Pydantic schema as `models.py`) instead of loading the 6GB local model — this makes it deployable for free on **Streamlit Community Cloud**, since the local fine-tuned model needs GPU hosting that isn't free.

1. Push this repo to GitHub
2. Go to **share.streamlit.io**, connect the repo, set the main file to `app_cloud.py` and requirements file to `requirements_cloud.txt`
3. In the app's **Settings → Secrets**, add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy

The publicly deployed app produces the same schema-consistent output as the locally fine-tuned model. The fine-tuned adapter itself is published at `huggingface.co/Jussara08/risk-register-lora` as evidence of the fine-tuning work.

## Schema

Every extracted entry follows a fixed schema (`scripts/schema_config.py`):

| Field | Values |
|---|---|
| `risk_category` | Operational, Cybersecurity, Compliance, Financial, Third-Party, Strategic |
| `likelihood` / `impact` | Low, Medium, High |
| `risk_rating` | Low, Medium, High, Critical |
| `mitigation_status` | Not Started, In Progress, Mitigated, Accepted |
| `affected_control` | ISO 27001 clause or Annex A control reference |
| `control_owner` | role, never a person's name |

## Limitations

- Training data is partly synthetic (LLM-generated, seeded and spot-checked by hand) — risk of pattern-matching rather than true understanding.
- Outputs should go through human review before entering a production risk register.
- Category/severity judgments may reflect biases present in the seed and synthetic training narratives.
