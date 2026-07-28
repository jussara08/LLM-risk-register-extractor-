from enum import Enum

from pydantic import BaseModel, Field


class RiskCategory(str, Enum):
    cybersecurity = "Cybersecurity"
    operational = "Operational"
    financial = "Financial"
    third_party = "Third-Party"
    strategic = "Strategic"
    compliance = "Compliance"


class Level(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class RiskRating(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    critical = "Critical"


class MitigationStatus(str, Enum):
    not_started = "Not Started"
    in_progress = "In Progress"
    mitigated = "Mitigated"
    accepted = "Accepted"


class RiskRegisterEntry(BaseModel):
    risk_title: str = Field(description="Short descriptive title of the risk")

    risk_category: RiskCategory = Field(
        description="Cybersecurity, Operational, Financial, Third-Party, Strategic, Compliance"
    )

    description: str = Field(description="Detailed explanation of the risk")

    likelihood: Level = Field(description="Low, Medium, High")

    impact: Level = Field(description="Low, Medium, High")

    risk_rating: RiskRating = Field(description="Low, Medium, High, Critical")

    affected_control: str = Field(description="ISO 27001 control or clause affected")

    control_owner: str = Field(description="Responsible team or owner (role, not a person's name)")

    mitigation_status: MitigationStatus = Field(
        description="Not Started, In Progress, Mitigated, Accepted"
    )

    recommended_action: str = Field(description="Suggested remediation action")

    confidence: float = Field(
        ge=0,
        le=1,
        description="Model's confidence in this extraction, from 0 to 1",
    )


class GeneratedTrainingExample(BaseModel):
    """Wraps a synthetic narrative with its structured extraction.

    Used as the tool schema during dataset generation, so Claude returns
    both the invented narrative AND its structured extraction in a single,
    schema-enforced tool call.
    """

    narrative: str = Field(
        description="The invented, realistic ISO 27001 audit/risk narrative (2-5 sentences)"
    )
    entry: RiskRegisterEntry = Field(
        description="The structured risk register entry extracted from the narrative"
    )
