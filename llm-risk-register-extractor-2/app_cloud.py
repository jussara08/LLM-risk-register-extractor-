"""
Lightweight, API-based version of the Risk Register Extractor, built for
free public deployment (Streamlit Community Cloud) where a 6GB local model
with GPU inference isn't practical.

Uses Claude's tool-use with the same Pydantic schema (scripts/models.py)
that was used to generate the training data and that the locally fine-tuned
model (see app.py) was trained to replicate. This keeps the public demo
schema-consistent with the fine-tuned model without requiring GPU hosting.

Usage locally:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app_cloud.py

On Streamlit Community Cloud:
    Set ANTHROPIC_API_KEY in the app's Secrets (Settings -> Secrets), as:
        ANTHROPIC_API_KEY = "sk-ant-..."
"""

import json
import os

import anthropic
import pandas as pd
import streamlit as st

from scripts.models import GeneratedTrainingExample, RiskRegisterEntry
from scripts.schema_config import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="Risk Register Extractor", layout="wide", page_icon="📋")

# ---------------------------------------------------------------------------
# Same "audit ledger" design system as the local app.py
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
}

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--ink); }

.ledger-masthead { border-bottom: 2px solid var(--rust); padding-bottom: 1.1rem; margin-bottom: 1.6rem; }
.ledger-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--rust); margin-bottom: 0.3rem;
}
.ledger-title {
    font-family: 'Source Serif 4', serif; font-weight: 700; font-size: 2.1rem;
    color: var(--text-on-ink); line-height: 1.15; margin: 0;
}
.ledger-subtitle { font-size: 0.95rem; color: var(--text-muted); margin-top: 0.4rem; }

.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--ink-soft); }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--text-muted); padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] { color: var(--rust) !important; border-bottom: 2px solid var(--rust) !important; }

.stTextArea textarea {
    background: var(--paper) !important; color: var(--ink) !important;
    border: 1px solid var(--paper-line) !important; border-radius: 2px !important;
    font-size: 0.95rem !important;
}
.stTextArea label {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted) !important;
}

.stButton button {
    background: var(--rust) !important; color: var(--paper) !important; border: none !important;
    border-radius: 2px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.55rem 1.4rem !important; transition: background 0.15s ease;
}
.stButton button:hover { background: var(--rust-dark) !important; }

.entry-card {
    background: var(--paper); border: 1px solid var(--paper-line);
    border-left: 5px solid var(--sev-color, var(--rust)); border-radius: 2px;
    padding: 1.4rem 1.6rem 1.6rem 1.6rem; margin-top: 1.2rem; position: relative;
}
.entry-title {
    font-family: 'Source Serif 4', serif; font-weight: 600; font-size: 1.25rem;
    color: var(--ink); margin: 0 0 0.9rem 0; padding-right: 6.5rem;
}
.entry-desc { font-size: 0.92rem; color: #3A3F33; line-height: 1.55; margin-bottom: 1.1rem; }
.entry-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.9rem 1.4rem; border-top: 1px dashed var(--paper-line); padding-top: 1rem;
}
.entry-field-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8A8368; margin-bottom: 0.2rem;
}
.entry-field-value { font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem; color: var(--ink); }
.entry-action {
    grid-column: 1 / -1; font-size: 0.88rem; color: #3A3F33; background: #EFE9D8;
    border-radius: 2px; padding: 0.7rem 0.9rem; margin-top: 0.2rem;
}
.ink-stamp {
    position: absolute; top: 1.3rem; right: 1.5rem;
    border: 2.5px solid var(--sev-color, var(--rust)); color: var(--sev-color, var(--rust));
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.72rem;
    letter-spacing: 0.12em; text-transform: uppercase; padding: 0.3rem 0.6rem;
    border-radius: 4px; transform: rotate(4deg); opacity: 0.92;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;
}
.footnote {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--text-muted);
    margin-top: 2rem; line-height: 1.6;
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


def get_api_key() -> str | None:
    # Prefer Streamlit secrets (used on Streamlit Community Cloud);
    # fall back to environment variable for local runs.
    if "ANTHROPIC_API_KEY" in st.secrets:
        return st.secrets["ANTHROPIC_API_KEY"]
    return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource
def get_client():
    api_key = get_api_key()
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


EXTRACTION_TOOL = {
    "name": "risk_register_entry",
    "description": "Record a structured risk register entry extracted from a narrative.",
    "input_schema": RiskRegisterEntry.model_json_schema(),
}


def extract(narrative: str, client) -> dict:
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=800,
            system=BASE_SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "risk_register_entry"},
            messages=[{
                "role": "user",
                "content": f"Extract a structured risk register entry from this narrative:\n\n{narrative}",
            }],
        )
        tool_use_block = next(b for b in resp.content if b.type == "tool_use")
        validated = RiskRegisterEntry.model_validate(tool_use_block.input)
        return validated.model_dump()
    except Exception as e:
        return {"error": str(e)}


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
        ("Confidence", f"{entry.get('confidence', 0):.2f}" if "confidence" in entry else "—"),
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
        <div class="ledger-eyebrow">Schema-enforced extraction · ISO 27001</div>
        <div class="ledger-title">Risk Register Extractor</div>
        <div class="ledger-subtitle">Turn unstructured audit and risk narratives into structured, schema-consistent register entries.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

client = get_client()
if client is None:
    st.error(
        "No ANTHROPIC_API_KEY found. Set it in Streamlit Secrets "
        "(Settings → Secrets) or as an environment variable locally."
    )
    st.stop()

tab1, tab2 = st.tabs(["Single Narrative", "Batch (CSV)"])

with tab1:
    narrative_input = st.text_area(
        "Narrative",
        height=150,
        placeholder="During the Q2 access review, it was found that...",
    )
    if st.button("Extract entry", type="primary"):
        if not narrative_input.strip():
            st.warning("Please paste a narrative first.")
        else:
            with st.spinner("Extracting..."):
                result = extract(narrative_input, client)
            if "error" in result:
                st.error(f"Extraction failed: {result['error']}")
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
                results = []
                progress = st.progress(0)
                for i, row in df.iterrows():
                    result = extract(row["narrative"], client)
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

st.markdown(
    """
    <div class="footnote">
        This public demo uses schema-enforced tool-use extraction via the Claude API.<br>
        A locally fine-tuned Llama 3.2 model trained to replicate this exact schema is available at
        huggingface.co/Jussara08/risk-register-lora.
    </div>
    """,
    unsafe_allow_html=True,
)
