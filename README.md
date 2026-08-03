# Risk Register Extractor

**Fine-tuned LLM for structured GRC risk extraction, mapped to ISO 27001 controls.**

[![Live Demo](https://img.shields.io/badge/demo-live-A8552E?style=flat-square)](https://cebvsxshntykvtwfkhkeue.streamlit.app)
[![Model on HF](https://img.shields.io/badge/model-huggingface-yellow?style=flat-square)](https://huggingface.co/Jussara08/risk-register-lora)
[![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square)](https://www.python.org/)

**[Live demo →](https://cebvsxshntykvtwfkhkeue.streamlit.app)** · **[Fine-tuned model →](https://huggingface.co/Jussara08/risk-register-lora)**

---

## The problem

GRC analysts spend a lot of time manually converting unstructured text — audit findings, incident reports, risk assessments — into structured risk register entries: assigning a category, scoring likelihood and impact, mapping to the right control, naming an owner, drafting a remediation step. This is slow, and it's inconsistent between analysts, since two people reading the same finding can rate or categorize it differently.

This project automates that translation, from raw narrative to a schema-consistent, register-ready entry:

```
"During the Q2 access review, it was found that 14 terminated employees
still had active VPN credentials, with the oldest dating back 90 days
post-termination. No automated deprovisioning process exists between
HR and IT."
```
↓
```json
{
  "risk_title": "Delayed deprovisioning of terminated employee access",
  "risk_category": "Cybersecurity",
  "likelihood": "High",
  "impact": "High",
  "risk_rating": "Critical",
  "affected_control": "ISO 27001 Annex A 5.18 (Access rights)",
  "control_owner": "IT Security",
  "mitigation_status": "Not Started",
  "recommended_action": "Implement automated deprovisioning triggered by HR offboarding workflow"
}
```

## Why schema enforcement, not just prompting

A generic LLM prompt gives you an inconsistent paragraph every time. This project forces every extraction through a fixed schema — a **Pydantic model** (`scripts/models.py`) with strict enums for category, likelihood, impact, rating, and mitigation status — so the output is structurally guaranteed to be usable downstream (a register, a dashboard, a report), not just fast to produce.

That schema is enforced in two different ways depending on the deployment:
- **Local / fine-tuned model** (`app.py`): a Llama 3.2 3B model fine-tuned with LoRA to reproduce the schema directly in its output.
- **Public demo** (`app_cloud.py`): Claude's tool-use / function-calling, with the same Pydantic schema passed as a tool definition, so the API is structurally constrained to return valid enum values.

## Pipeline

```
25 hand-written seed examples (data/seed_examples.jsonl)
        │  used as few-shot exemplars + style guide
        ▼
Synthetic dataset generation (scripts/generate_dataset.py)
  Claude tool-use + Pydantic schema -> schema-guaranteed synthetic examples
        ▼
149 synthetic training examples (data/train.jsonl)
        ▼
QLoRA fine-tuning (scripts/train.py)
  Llama 3.2 3B Instruct + LoRA, trained on Kaggle (T4 GPU)
        ▼
Fine-tuned adapter -> huggingface.co/Jussara08/risk-register-lora
        ▼
   +--------------------+-------------------------+
   v                    v
Local demo (app.py)   Public demo (app_cloud.py)
loads fine-tuned      Claude API + tool-use,
model directly         same schema, no GPU needed
                       -> deployed on Streamlit Cloud
```

## What's in this repo

| File | Purpose |
|---|---|
| `scripts/models.py` | Pydantic schema — the single source of truth for the risk register structure |
| `scripts/schema_config.py` | Shared constants and system prompt |
| `data/seed_examples.jsonl` | 25 hand-written, hand-labeled seed examples (ISO 27001-grounded) |
| `scripts/generate_dataset.py` | Expands seeds into synthetic training data via Claude tool-use |
| `scripts/train.py` | QLoRA fine-tuning script (Llama 3.2 3B + LoRA) |
| `scripts/evaluate.py` | Structural validity + field-level accuracy scoring |
| `app.py` | Streamlit demo using the locally fine-tuned model |
| `app_cloud.py` | Lightweight Streamlit demo using the Claude API (deployed publicly) |

## Schema

| Field | Values |
|---|---|
| `risk_category` | Operational, Cybersecurity, Compliance, Financial, Third-Party, Strategic |
| `likelihood` / `impact` | Low, Medium, High |
| `risk_rating` | Low, Medium, High, Critical |
| `mitigation_status` | Not Started, In Progress, Mitigated, Accepted |
| `affected_control` | ISO 27001 clause or Annex A control reference |
| `control_owner` | Role, never a person's name |
| `confidence` | Model's self-assessed confidence (0-1) |

## Fine-tuning notes

Training ran on a Kaggle T4 GPU. A few real constraints shaped the final setup:
- 4-bit quantization (`bitsandbytes`) was dropped after persistent CUDA compatibility issues on the available GPUs — the model (3B params) was fine-tuned in bf16 instead, with gradient checkpointing and a batch size of 1 (gradient accumulation x4) to fit in 16GB of GPU memory.
- LoRA applied to attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), r=16, alpha=32.

## Setup

```bash
pip install -r requirements.txt
```

**1. Generate synthetic training data:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_dataset.py --n 400 --out data/train.jsonl
```

**2. Fine-tune** (needs GPU — Colab/Kaggle recommended):
```bash
python scripts/train.py --data data/train.jsonl --output_dir models/risk-register-lora --epochs 3
```

**3. Run the local demo:**
```bash
streamlit run app.py
```

**4. Or run the lightweight API-based demo:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app_cloud.py
```

## Limitations

- Training data is partly synthetic (Claude-generated, seeded and spot-checked by hand) — some risk of pattern-matching rather than true generalization.
- Outputs should go through human review before entering a production risk register.
- Category and severity judgments may reflect biases present in the seed and synthetic training narratives — automation replaces analyst inconsistency with model consistency, which is not automatically the same as correctness, and should be periodically audited.
