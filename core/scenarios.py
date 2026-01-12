"""
Scenario definitions shared across Streamlit, GIF export, and sliders.
Keeping them in one place prevents subtle drift between UIs.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.roadmap import TRAIN_PARAMS, TARGET_PARAMS


# Base scenario presets (kept small + readable for workshop explanation).
NO_SYSTEM_PARAMS = {"N": 0.25, "F": 0.15, "D": 0.20, "V": 0.10, "R": 0.15}


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario metadata to keep the UI labels + colors consistent."""

    name: str
    params: dict[str, float]
    description: str
    color: str


def build_scenarios(optimized_params: dict[str, float] | None = None) -> list[ScenarioConfig]:
    """
    Return the three fixed scenarios used across the demo.
    Optimized parameters can be overridden (e.g., by sliders in Streamlit).
    """
    optimized = TARGET_PARAMS if optimized_params is None else optimized_params
    return [
        ScenarioConfig(
            name="No system",
            params=NO_SYSTEM_PARAMS.copy(),
            description="No training system in place",
            color="#7f7f7f",
        ),
        ScenarioConfig(
            name="Training system",
            params=TRAIN_PARAMS.copy(),
            description="Baseline training introduced",
            color="#1f77b4",
        ),
        ScenarioConfig(
            name="Optimized",
            params=optimized.copy(),
            description="DoJo-optimized (slider driven)",
            color="#2ca02c",
        ),
    ]
