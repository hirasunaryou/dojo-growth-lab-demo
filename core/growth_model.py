"""
Shared growth model logic for Streamlit, Matplotlib sliders, and GIF export.
Keep formulas consistent across all entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PARAM_KEYS = ("N", "F", "D", "V", "R")


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulation outputs."""

    week: np.ndarray
    S: np.ndarray
    Smax: np.ndarray
    U: np.ndarray
    v: float


def v_from_params(params: dict[str, float], v0: float = 0.25) -> float:
    """
    Growth velocity derived from the geometric mean of N/F/D/V/R.
    """
    prod = 1.0
    for key in PARAM_KEYS:
        prod *= params[key]
    return v0 * (prod ** (1 / 5))


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
) -> SimulationResult:
    """
    Simulate the growth curve and frontier unlock over discrete weeks.
    """
    weeks = np.arange(T + 1)
    v = v_from_params(params, v0=v0)

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
            U[t] = 0.0

        if enable_frontier:
            Smax[t] = Smax_base + delta_Smax * (U[t] / Umax)
        else:
            Smax[t] = 0.0

        # Skill update (discrete-time form)
        if t < T:
            S[t + 1] = S[t] + v * (Smax[t] - S[t])

    return SimulationResult(week=weeks, S=S, Smax=Smax, U=U, v=v)


def weeks_to_reach(S_series: np.ndarray, threshold: float) -> int | None:
    """
    Return the first week index where S reaches the threshold.
    """
    idx = np.argmax(S_series >= threshold)
    if S_series[idx] < threshold:
        return None
    return int(idx)
