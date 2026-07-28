"""
Streamlit demo: paste a risk narrative (or upload a CSV of narratives) and
get back structured risk register entries.

Loads the base model + fine-tuned LoRA adapter for inference.

Usage:
    streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.schema_config import SYSTEM_PROMPT

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "models/risk-register-lora"

st.set_page_config(page_title="Risk Register Extractor", layout="wide", page_icon="📋")

# ---------------------------------------------------------------------------
# Design system: an "audit ledger" aesthetic -- ink-navy chrome, warm paper
# document cards, serif headers, monospace for control codes and ratings.
# Severity is shown as a rotated ink-stamp rather than a generic colored pill.
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #161C24;
    --ink-soft: #232B36;
    --paper: #F7F4EC;
    --paper-line: #E4DDCB;
    --rust: #A8552E;
    --rust-dark: #8A4425;
    --text-on-ink: #EDE8DA;
    --text-muted: #8B93A0;
    --critical: #8B3A3A;
    --high: #A8752F;
    --medium: #48628A;
    --low: #5B7A63;
    --mitigated: #3F6E68;
    --accepted: #6B5B95;
}

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: var(--ink);
}

/* Header / masthead */
.ledger-masthead {
    border-bottom: 2px solid var(--rust);
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
}
.ledger-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--rust);
    margin-bottom: 0.3rem;
}
.ledger-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: var(--text-on-ink);
    line-height: 1.15;
    margin: 0;
}
.ledger-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--ink-soft);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    color: var(--rust) !important;
    border-bottom: 2px solid var(--rust) !important;
}

/* Text area (intake form) */
.stTextArea textarea {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid var(--paper-line) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}

/* Primary button styled like a rubber stamp action */
.stButton button {
    background: var(--rust) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.15s ease;
}
.stButton button:hover {
    background: var(--rust-dark) !important;
}

/* Result card: the "register entry" on paper */
.entry-card {
    background: var(--paper);
    border: 1px solid var(--paper-line);
    border-left: 5px solid var(--sev-color, var(--rust));
    border-radius: 2px;
    padding: 1.4rem 1.6rem 1.6rem 1.6rem;
    margin-top: 1.2rem;
    position: relative;
}
.entry-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.25rem;
    color: var(--ink);
    margin: 0 0 0.9rem 0;
    padding-right: 6.5rem;
}
.entry-desc {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem;
    color: #3A3F33;
    line-height: 1.55;
    margin-bottom: 1.1rem;
}
.entry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.9rem 1.4rem;
    border-top: 1px dashed var(--paper-line);
    padding-top: 1rem;
}
.entry-field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8A8368;
    margin-bottom: 0.2rem;
}
.entry-field-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.86rem;
    color: var(--ink);
}
.entry-action {
    grid-column: 1 / -1;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
    color: #3A3F33;
    background: #EFE9D8;
    border-radius: 2px;
    padding: 0.7rem 0.9rem;
    margin-top: 0.2rem;
}

/* The ink stamp -- signature element, top-right of each card */
.ink-stamp {
    position: absolute;
    top: 1.3rem;
    right: 1.5rem;
    border: 2.5px solid var(--sev-color, var(--rust));
    color: var(--sev-color, var(--rust));
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    transform: rotate(4deg);
    opacity: 0.92;
}

/* Section labels used before file_uploader / dataframe */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
</style>
"""

SEVERITY_COLOR = {
    "Critical": "#8B3A3A",
    "High": "#A8752F",
    "Medium": "#48628A",
    "Low": "#5B7A63",
}

st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def extract(narrative: str, model, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract a structured risk register entry from this narrative:\n\n{narrative}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, temperature=0.2, do_sample=True)

    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Model output was not valid JSON", "raw_output": raw}


def render_entry_card(entry: dict):
    rating = entry.get("risk_rating", "—")
    color = SEVERITY_COLOR.get(rating, "#A8552E")

    fields = [
        ("Category", entry.get("risk_category", "—")),
        ("Likelihood", entry.get("likelihood", "—")),
        ("Impact", entry.get("impact", "—")),
        ("Affected control", entry.get("affected_control", "—")),
        ("Control owner", entry.get("control_owner", "—")),
        ("Mitigation status", entry.get("mitigation_status", "—")),
    ]
    fields_html = "".join(
        f'<div><div class="entry-field-label">{label}</div>'
        f'<div class="entry-field-value">{value}</div></div>'
        for label, value in fields
    )

    card_html = f"""
    <div class="entry-card" style="--sev-color:{color};">
        <div class="ink-stamp">{rating} risk</div>
        <div class="entry-title">{entry.get('risk_title', 'Untitled risk')}</div>
        <div class="entry-desc">{entry.get('description', '')}</div>
        <div class="entry-grid">
            {fields_html}
            <div class="entry-action">→ {entry.get('recommended_action', '')}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="ledger-masthead">
        <div class="ledger-eyebrow">Fine-tuned LLM · ISO 27001 extraction</div>
        <div class="ledger-title">Risk Register Extractor</div>
        <div class="ledger-subtitle">Turn unstructured audit and risk narratives into structured, schema-consistent register entries.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Single Narrative", "Batch (CSV)"])

with tab1:
    narrative_input = st.text_area(
        "Narrative",
        height=150,
        placeholder="During the Q2 access review, it was found that...",
        label_visibility="visible",
    )
    if st.button("Extract entry", type="primary"):
        if not narrative_input.strip():
            st.warning("Please paste a narrative first.")
        else:
            with st.spinner("Extracting..."):
                model, tokenizer = load_model()
                result = extract(narrative_input, model, tokenizer)
            if "error" in result:
                st.error(result["error"])
                st.code(result["raw_output"])
            else:
                render_entry_card(result)

with tab2:
    st.markdown('<div class="section-label">Upload CSV with a `narrative` column</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type="csv", label_visibility="collapsed")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "narrative" not in df.columns:
            st.error("CSV must contain a 'narrative' column.")
        else:
            if st.button("Process batch", type="primary"):
                model, tokenizer = load_model()
                results = []
                progress = st.progress(0)
                for i, row in df.iterrows():
                    result = extract(row["narrative"], model, tokenizer)
                    results.append(result)
                    progress.progress((i + 1) / len(df))

                for result in results:
                    if "error" not in result:
                        render_entry_card(result)

                results_df = pd.DataFrame(results)
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download populated risk register (CSV)",
                    data=csv_out,
                    file_name="risk_register_output.csv",
                    mime="text/csv",
                )
