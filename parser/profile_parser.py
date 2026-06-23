from typing import List, Optional
import re

from pydantic import BaseModel, Field


class PatientProfile(BaseModel):

    patient_name: Optional[str] = Field(default=None)

    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120
    )

    primary_wellness_goals: List[str] = Field(
        default_factory=list
    )

    sleeping_habits: Optional[str] = Field(
        default=None
    )

    current_struggles: List[str] = Field(
        default_factory=list
    )


def extract_patient_profile(
    onboarding_text: str,
    llm=None
) -> PatientProfile:

    profile = PatientProfile()

    text = onboarding_text.lower()

    name_match = re.search(
        r"my name is\s+([a-zA-Z]+)",
        onboarding_text,
        re.IGNORECASE
    )

    if name_match:
        profile.patient_name = name_match.group(1)

    age_match = re.search(
        r"(\d+)\s*years?\s*old",
        onboarding_text,
        re.IGNORECASE
    )

    if age_match:
        profile.age = int(
            age_match.group(1)
        )

    goals = []

    if "lose weight" in text:
        goals.append("lose weight")

    if "sleep" in text:
        goals.append("improve sleep")

    if "fitness" in text:
        goals.append("improve fitness")

    profile.primary_wellness_goals = goals

    if "sleep" in text:
        profile.sleeping_habits = onboarding_text

    struggles = []

    if "snacking" in text:
        struggles.append(
            "late-night snacking"
        )

    if "exercise" in text:
        struggles.append(
            "exercise consistency"
        )

    profile.current_struggles = struggles

    return profile