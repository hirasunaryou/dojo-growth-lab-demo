"""Research progress helpers (yearly maturity + parameter interpolation)."""

from __future__ import annotations

from typing import Iterable
import math

from core.constants import PARAM_KEYS

# Default research parameters (train -> target), easy to tweak later.
TRAIN_PARAMS = {"N": 0.40, "F": 0.40, "D": 0.35, "V": 0.30, "R": 0.35}
TARGET_PARAMS = {"N": 0.80, "F": 0.70, "D": 0.60, "V": 0.75, "R": 0.70}


def maturity(year: float, k: float) -> float:
    """Saturating maturity curve m(y) = 1 - exp(-k y)."""
    return 1.0 - math.exp(-k * year)


def interpolate_params(
    year: float,
    train_params: dict[str, float],
    target_params: dict[str, float],
    k: float,
) -> dict[str, float]:
    """Interpolate N/F/D/V/R between train and target at a given year."""
    m = maturity(year, k)
    return {
        key: train_params[key] + (target_params[key] - train_params[key]) * m
        for key in PARAM_KEYS
    }


def build_year_table(
    years: Iterable[float],
    train_params: dict[str, float],
    target_params: dict[str, float],
    k: float,
) -> list[dict[str, float]]:
    """Return per-year parameter rows for display tables."""
    rows: list[dict[str, float]] = []
    for year in years:
        params = interpolate_params(year, train_params, target_params, k)
        rows.append({"year": float(year), **params})
    return rows


def year_from_week(week: float, weeks_per_year: int = 52) -> float:
    """Convert a week index to a floating year value."""
    return week / weeks_per_year
