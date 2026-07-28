"""
Central schema definition for the risk register extraction task.
Keeping this in one file means the dataset generator, training script,
and evaluator all validate against the exact same schema.
"""

RISK_CATEGORIES = [
    "Operational",
    "Cybersecurity",
    "Compliance",
    "Financial",
    "Third-Party",
    "Strategic",
]

LIKELIHOOD_LEVELS = ["Low", "Medium", "High"]
IMPACT_LEVELS = ["Low", "Medium", "High"]
RISK_RATINGS = ["Low", "Medium", "High", "Critical"]
MITIGATION_STATUSES = ["Not Started", "In Progress", "Mitigated", "Accepted"]

REQUIRED_OUTPUT_FIELDS = [
    "risk_title",
    "risk_category",
    "description",
    "likelihood",
    "impact",
    "risk_rating",
    "affected_control",
    "control_owner",
    "mitigation_status",
    "recommended_action",
]

FIELD_VALUE_CONSTRAINTS = {
    "risk_category": RISK_CATEGORIES,
    "likelihood": LIKELIHOOD_LEVELS,
    "impact": IMPACT_LEVELS,
    "risk_rating": RISK_RATINGS,
    "mitigation_status": MITIGATION_STATUSES,
}

SYSTEM_PROMPT = """You are a GRC analyst assistant. Given an unstructured risk or \
audit finding narrative, extract a structured risk register entry as a single \
JSON object with exactly these fields:

- risk_title: short descriptive title (string)
- risk_category: one of Operational, Cybersecurity, Compliance, Financial, Third-Party, Strategic
- description: 1-2 sentence summary of the risk in your own words
- likelihood: Low, Medium, or High
- impact: Low, Medium, or High
- risk_rating: Low, Medium, High, or Critical (overall rating, not just likelihood x impact multiplied mechanically -- use judgment)
- affected_control: the relevant ISO 27001 clause or Annex A control (e.g., "ISO 27001 Annex A 5.18 (Access rights)")
- control_owner: a role/function, never a person's name (e.g., "IT Security", "Legal / Compliance Officer")
- mitigation_status: Not Started, In Progress, Mitigated, or Accepted
- recommended_action: one concrete, actionable next step

Return ONLY the JSON object. No preamble, no markdown code fences, no explanation.
"""


def validate_output(output: dict) -> list[str]:
    """Returns a list of validation errors (empty list = valid)."""
    errors = []
    for field in REQUIRED_OUTPUT_FIELDS:
        if field not in output:
            errors.append(f"missing field: {field}")
    for field, allowed in FIELD_VALUE_CONSTRAINTS.items():
        if field in output and output[field] not in allowed:
            errors.append(f"invalid value for {field}: {output[field]!r} (allowed: {allowed})")
    return errors
