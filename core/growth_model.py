"""
Shared growth model logic for Streamlit, Matplotlib sliders, and GIF export.
Keep formulas consistent across all entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


from core.constants import PARAM_KEYS
from core.roadmap import interpolate_params, year_from_week


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulation outputs."""

    week: np.ndarray
    S: np.ndarray
    Smax: np.ndarray
    U: np.ndarray
    v: float
    frontier_strength: float
    Smax_cap: float


@dataclass(frozen=True)
class ResearchSimulationResult:
    """Container for research-progress simulation outputs (time-varying params)."""

    week: np.ndarray
    S: np.ndarray
    Smax: np.ndarray
    U: np.ndarray
    v: np.ndarray
    Smax_cap: np.ndarray


def v_from_params(params: dict[str, float], v0: float = 0.25) -> float:
    """
    Growth velocity derived from the geometric mean of N/F/D/V/R.
    """
    prod = 1.0
    for key in PARAM_KEYS:
        prod *= params[key]
    return v0 * (prod ** (1 / 5))


def smax_cap_from_params(
    params: dict[str, float],
    Smax_base: float,
    delta_Smax: float,
    gamma: float,
) -> float:
    """
    Theoretical ceiling derived from V×D (frontier strength).
    """
    frontier_strength = (params["V"] * params["D"]) ** gamma
    return Smax_base + delta_Smax * frontier_strength


def simulate(
    params: dict[str, float],
    T: int,
    S0: float,
    v0: float,
    Smax_base: float,
    delta_Smax: float,
    Umax: float,
    u0: float,
    enable_frontier: bool,
    gamma: float = 0.5,
) -> SimulationResult:
    """
    Simulate the growth curve and frontier unlock over discrete weeks.
    gamma controls how strongly V×D expands the theoretical ceiling (Smax_cap).
    """
    weeks = np.arange(T + 1)
    v = v_from_params(params, v0=v0)
    # Frontier strength is the "ceiling amplifier" driven by V×D.
    frontier_strength = (params["V"] * params["D"]) ** gamma
    Smax_cap = smax_cap_from_params(params, Smax_base, delta_Smax, gamma)

    S = np.zeros(T + 1)
    U = np.zeros(T + 1)
    Smax = np.zeros(T + 1)
    S[0] = S0

    for t in range(T + 1):
        # Frontier unlock (saturating growth)
        if enable_frontier and t > 0:
            u_rate = u0 * (params["V"] * params["D"])
            U[t] = U[t - 1] + u_rate * (1 - U[t - 1] / Umax)
        else:
            # When frontier is off, unlock stays at 0 (ceiling stays fixed).
            U[t] = 0.0

        if enable_frontier:
            # V×D affects the theoretical ceiling; U(t) determines how much is unlocked.
            Smax[t] = Smax_base + delta_Smax * frontier_strength * (U[t] / Umax)
        else:
            # Frontier OFF means ceiling is fixed at the base value.
            Smax[t] = Smax_base

        # Skill update (discrete-time form)
        if t < T:
            S[t + 1] = S[t] + v * (Smax[t] - S[t])

    return SimulationResult(
        week=weeks,
        S=S,
        Smax=Smax,
        U=U,
        v=v,
        frontier_strength=frontier_strength,
        Smax_cap=Smax_cap,
    )


def simulate_research_progress(
    train_params: dict[str, float],
    target_params: dict[str, float],
    T: int,
    S0: float,
    v0: float,
    Smax_base: float,
    delta_Smax: float,
    Umax: float,
    u0: float,
    enable_frontier: bool,
    gamma: float,
    k: float,
    weeks_per_year: int = 52,
) -> ResearchSimulationResult:
    """
    Simulate a single learner while N/F/D/V/R improve year by year.
    Each week, parameters are re-computed from the maturity curve.
    """
    weeks = np.arange(T + 1)
    S = np.zeros(T + 1)
    U = np.zeros(T + 1)
    Smax = np.zeros(T + 1)
    v_series = np.zeros(T + 1)
    smax_cap_series = np.zeros(T + 1)
    S[0] = S0

    for t in range(T + 1):
        year = year_from_week(t, weeks_per_year=weeks_per_year)
        params = interpolate_params(year, train_params, target_params, k)
        v = v_from_params(params, v0=v0)
        v_series[t] = v
        smax_cap_series[t] = smax_cap_from_params(params, Smax_base, delta_Smax, gamma)

        # Frontier unlock (saturating growth), driven by current V×D.
        if enable_frontier and t > 0:
            u_rate = u0 * (params["V"] * params["D"])
            U[t] = U[t - 1] + u_rate * (1 - U[t - 1] / Umax)
        else:
            # When frontier is off, unlock stays at 0 (ceiling stays fixed).
            U[t] = 0.0

        if enable_frontier:
            frontier_strength = (params["V"] * params["D"]) ** gamma
            Smax[t] = Smax_base + delta_Smax * frontier_strength * (U[t] / Umax)
        else:
            Smax[t] = Smax_base

        # Skill update (discrete-time form) using the current v and Smax.
        if t < T:
            S[t + 1] = S[t] + v * (Smax[t] - S[t])

    return ResearchSimulationResult(
        week=weeks,
        S=S,
        Smax=Smax,
        U=U,
        v=v_series,
        Smax_cap=smax_cap_series,
    )


def weeks_to_reach(S_series: np.ndarray, threshold: float) -> int | None:
    """
    Return the first week index where S reaches the threshold.
    """
    idx = np.argmax(S_series >= threshold)
    if S_series[idx] < threshold:
        return None
    return int(idx)
