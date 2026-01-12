"""
Generate GIFs for PPT assets.

Modes:
  Sweep (existing):
    python scripts/export_gifs.py --preset optimized --param N --min 0.2 --max 0.9 --frames 30 --fps 10 --out out/gifs
    python scripts/export_gifs.py --all --preset optimized
    python scripts/export_gifs.py --compare3 --preset optimized --param V --frames 30 --fps 10 --out out/gifs

  Research Progress (new):
    python scripts/export_gifs.py --mode research_progress --years 0 3 --frames 30 --fps 10 --out out/gifs
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `python scripts/export_gifs.py` works anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as exc:
    raise SystemExit(
        "imageio が見つかりません。次を実行してください: python -m pip install imageio Pillow"
    ) from exc

import matplotlib
import numpy as np

from core.constants import PARAM_KEYS
from core.growth_model import (
    simulate,
    simulate_research_progress,
    smax_cap_from_params,
    v_from_params,
    weeks_to_reach,
)
from core.roadmap import TRAIN_PARAMS, TARGET_PARAMS, interpolate_params
from core.scenarios import NO_SYSTEM_PARAMS, build_scenarios

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NO_SYSTEM = NO_SYSTEM_PARAMS.copy()
TRAINING = TRAIN_PARAMS.copy()
OPTIMIZED = TARGET_PARAMS.copy()
PRESETS = {
    "no_system": NO_SYSTEM,
    "training": TRAINING,
    "optimized": OPTIMIZED,
    "baseline": NO_SYSTEM,
    "dojo": OPTIMIZED,
    "custom": OPTIMIZED,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GIFs for DoJo Growth Lab.")
    parser.add_argument(
        "--mode",
        choices=["sweep", "research_progress"],
        default="sweep",
        help="Export mode: sweep or research progress animation.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="optimized")
    parser.add_argument("--param", choices=PARAM_KEYS, help="Parameter to sweep.")
    parser.add_argument("--all", action="store_true", help="Generate GIFs for all parameters.")
    parser.add_argument("--compare3", action="store_true", help="Render No/Training/Optimized together.")
    parser.add_argument("--min", dest="min_value", type=float, default=0.2)
    parser.add_argument("--max", dest="max_value", type=float, default=0.9)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=str, default="out/gifs")
    parser.add_argument("--T", type=int, default=520)
    parser.add_argument("--S0", type=float, default=20.0)
    parser.add_argument("--v0", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument(
        "--disable-frontier",
        action="store_false",
        dest="enable_frontier",
        help="Disable Frontier Unlock (Smax uplift).",
    )
    parser.add_argument("--Smax-base", type=float, default=80.0)
    parser.add_argument("--delta-Smax", type=float, default=15.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--Umax", type=float, default=10.0)
    parser.add_argument("--u0", type=float, default=2.0)
    # Research progress args
    parser.add_argument("--years", nargs=2, type=float, default=[0.0, 3.0])
    parser.add_argument(
        "--research-view",
        choices=["continuous", "cohort"],
        default="continuous",
        help="Research progress view: continuous or cohort.",
    )
    parser.add_argument("--k", type=float, default=1.0, help="Maturity speed for research progress.")
    return parser


def preset_params(preset: str) -> dict[str, float]:
    return PRESETS[preset].copy()


def render_frame(
    params: dict[str, float],
    sweep_param: str,
    T: int,
    S0: float,
    v0: float,
    threshold: float,
    Smax_base: float,
    delta_Smax: float,
    gamma: float,
    Umax: float,
    u0: float,
    enable_frontier: bool,
    compare3: bool,
) -> np.ndarray:
    """Render a single sweep frame to an RGB array."""
    # Use years on the X-axis so the 10-year horizon reads naturally.
    years = T / 52
    x_axis = np.arange(T + 1) / 52

    scenarios = build_scenarios(optimized_params=params)
    results = {
        scenario.name: simulate(
            scenario.params, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
        )
        for scenario in scenarios
    }

    fig, (ax_skill, ax_smax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        constrained_layout=True,
    )

    # Skill curve
    colors = {scenario.name: scenario.color for scenario in scenarios}
    if compare3:
        for label, result in results.items():
            ax_skill.plot(x_axis, result.S, color=colors[label], label=label)
    else:
        ax_skill.plot(
            x_axis,
            results["Optimized"].S,
            color=colors["Optimized"],
            label="Optimized",
        )
    ax_skill.axhline(threshold, linestyle="--", linewidth=1, color="#666666")
    ax_skill.set_xlim(0, years)
    ax_skill.set_ylim(0, 100)
    ax_skill.set_xlabel("Year")
    ax_skill.set_ylabel("Skill S")
    ax_skill.set_title("Skill curve S(t)")
    ax_skill.legend(loc="lower right")

    # Ceiling curve
    if compare3:
        for label, result in results.items():
            ax_smax.plot(x_axis, result.Smax, color=colors[label], label=f"{label} Smax")
    else:
        ax_smax.plot(
            x_axis,
            results["Optimized"].Smax,
            color=colors["Optimized"],
            label="Optimized Smax",
        )
    ax_smax.set_xlim(0, years)
    ax_smax.set_ylim(0, 100)
    ax_smax.set_xlabel("Year")
    ax_smax.set_ylabel("Smax")

    # Overlay metrics (PPT-friendly)
    info_lines = [f"{sweep_param}={params[sweep_param]:.2f}"]
    for scenario in scenarios:
        result = results[scenario.name]
        w_thr = weeks_to_reach(result.S, threshold)
        w_thr_label = f"{w_thr / 52:.1f}y" if w_thr is not None else "not reached"
        info_lines.append(
            f"{scenario.name}: N={scenario.params['N']:.2f} F={scenario.params['F']:.2f} "
            f"D={scenario.params['D']:.2f} V={scenario.params['V']:.2f} "
            f"R={scenario.params['R']:.2f} | v={result.v:.3f}/week | "
            f"Smax_cap={result.Smax_cap:.1f} | reach {threshold}: {w_thr_label}"
        )
    info = "\n".join(info_lines)
    ax_skill.text(0.01, 0.98, info, transform=ax_skill.transAxes, va="top", fontsize=10)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    image = imageio.imread(buffer)
    return image


def render_research_continuous_frame(
    year_value: float,
    T: int,
    S0: float,
    v0: float,
    threshold: float,
    Smax_base: float,
    delta_Smax: float,
    gamma: float,
    Umax: float,
    u0: float,
    enable_frontier: bool,
    k: float,
) -> np.ndarray:
    """Render a research-progress continuous frame with moving year marker."""
    years = T / 52
    x_axis = np.arange(T + 1) / 52

    no_result = simulate(
        NO_SYSTEM, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
    )
    training_result = simulate(
        TRAINING, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
    )
    research_result = simulate_research_progress(
        TRAINING,
        OPTIMIZED,
        T,
        S0,
        v0,
        Smax_base,
        delta_Smax,
        Umax,
        u0,
        enable_frontier,
        gamma,
        k,
    )

    fig, (ax_skill, ax_smax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        constrained_layout=True,
    )

    colors = {"No system": "#7f7f7f", "Training": "#1f77b4", "Research": "#2ca02c"}
    ax_skill.plot(x_axis, no_result.S, color=colors["No system"], label="No system")
    ax_skill.plot(x_axis, training_result.S, color=colors["Training"], label="Training")
    ax_skill.plot(x_axis, research_result.S, color=colors["Research"], label="Research-progress")
    ax_skill.axhline(threshold, linestyle="--", linewidth=1, color="#666666")
    ax_skill.axvline(year_value, linestyle=":", linewidth=1, color="#999999")
    ax_skill.set_xlim(0, years)
    ax_skill.set_ylim(0, 100)
    ax_skill.set_xlabel("Year")
    ax_skill.set_ylabel("Skill S")
    ax_skill.set_title("Research Progress (Continuous)")
    ax_skill.legend(loc="lower right")

    ax_smax.plot(x_axis, no_result.Smax, color=colors["No system"], label="No system Smax")
    ax_smax.plot(x_axis, training_result.Smax, color=colors["Training"], label="Training Smax")
    ax_smax.plot(x_axis, research_result.Smax, color=colors["Research"], label="Research Smax")
    ax_smax.set_xlim(0, years)
    ax_smax.set_ylim(0, 100)
    ax_smax.set_xlabel("Year")
    ax_smax.set_ylabel("Smax")

    # Overlay current-year params + metrics for all three scenarios.
    params = interpolate_params(year_value, TRAINING, OPTIMIZED, k)
    info_lines = [f"Year {year_value:.2f} (Research params)"]
    for label, scenario_params in [
        ("No system", NO_SYSTEM),
        ("Training system", TRAINING),
        ("Research (current)", params),
    ]:
        v = v_from_params(scenario_params, v0=v0)
        smax_cap = smax_cap_from_params(scenario_params, Smax_base, delta_Smax, gamma)
        info_lines.append(
            f"{label}: N={scenario_params['N']:.2f} F={scenario_params['F']:.2f} "
            f"D={scenario_params['D']:.2f} V={scenario_params['V']:.2f} "
            f"R={scenario_params['R']:.2f} | v={v:.3f}/week | Smax_cap={smax_cap:.1f}"
        )
    info = "\n".join(info_lines)
    ax_skill.text(0.01, 0.98, info, transform=ax_skill.transAxes, va="top", fontsize=10)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


def render_research_cohort_frame(
    year_value: float,
    cohort_years: list[int],
    cohort_reach: list[float | None],
    T: int,
    v0: float,
    Smax_base: float,
    delta_Smax: float,
    gamma: float,
    k: float,
) -> np.ndarray:
    """Render a cohort bar chart frame with a highlighted year."""
    years_horizon = T / 52
    fig, ax = plt.subplots(figsize=(8, 4))

    bar_values = [val if val is not None else years_horizon for val in cohort_reach]
    colors = ["#1f77b4" if yr <= year_value else "#c7c7c7" for yr in cohort_years]
    bars = ax.bar(cohort_years, bar_values, color=colors)

    for bar, value in zip(bars, cohort_reach):
        label = "NR" if value is None else f"{value:.1f}y"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Entry year (research maturity)")
    ax.set_ylabel("Years to reach threshold")
    ax.set_title("Research Progress (Cohort)")
    ax.set_ylim(0, years_horizon)

    params = interpolate_params(year_value, TRAINING, OPTIMIZED, k)
    info_lines = [f"Year {year_value:.2f} (Research params)"]
    for label, scenario_params in [
        ("No system", NO_SYSTEM),
        ("Training system", TRAINING),
        ("Research (current)", params),
    ]:
        v = v_from_params(scenario_params, v0=v0)
        smax_cap = smax_cap_from_params(scenario_params, Smax_base, delta_Smax, gamma)
        info_lines.append(
            f"{label}: N={scenario_params['N']:.2f} F={scenario_params['F']:.2f} "
            f"D={scenario_params['D']:.2f} V={scenario_params['V']:.2f} "
            f"R={scenario_params['R']:.2f} | v={v:.3f}/week | Smax_cap={smax_cap:.1f}"
        )
    info = "\n".join(info_lines)
    ax.text(0.01, 0.98, info, transform=ax.transAxes, va="top", fontsize=10)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


def export_gif(
    sweep_param: str,
    params: dict[str, float],
    min_value: float,
    max_value: float,
    frames: int,
    fps: int,
    out_dir: Path,
    T: int,
    S0: float,
    v0: float,
    threshold: float,
    Smax_base: float,
    delta_Smax: float,
    gamma: float,
    Umax: float,
    u0: float,
    enable_frontier: bool,
    compare3: bool,
) -> Path:
    """Generate a GIF for a single parameter sweep."""
    values = np.linspace(min_value, max_value, frames)
    images = []
    for value in values:
        params[sweep_param] = float(value)
        images.append(
            render_frame(
                params,
                sweep_param,
                T,
                S0,
                v0,
                threshold,
                Smax_base,
                delta_Smax,
                gamma,
                Umax,
                u0,
                enable_frontier,
                compare3,
            )
        )

    out_path = out_dir / f"sweep_{sweep_param}.gif"
    imageio.mimsave(out_path, images, fps=fps)
    return out_path


def export_research_progress_gif(
    start_year: float,
    end_year: float,
    frames: int,
    fps: int,
    out_dir: Path,
    view: str,
    T: int,
    S0: float,
    v0: float,
    threshold: float,
    Smax_base: float,
    delta_Smax: float,
    gamma: float,
    Umax: float,
    u0: float,
    enable_frontier: bool,
    k: float,
) -> Path:
    """Generate a research-progress GIF for either continuous or cohort view."""
    years = np.linspace(start_year, end_year, frames)
    images = []

    if view == "continuous":
        for year_value in years:
            images.append(
                render_research_continuous_frame(
                    year_value,
                    T,
                    S0,
                    v0,
                    threshold,
                    Smax_base,
                    delta_Smax,
                    gamma,
                    Umax,
                    u0,
                    enable_frontier,
                    k,
                )
            )
    else:
        cohort_start = int(math.floor(start_year))
        cohort_end = int(math.ceil(end_year))
        cohort_years = list(range(cohort_start, cohort_end + 1))
        cohort_reach: list[float | None] = []
        for entry_year in cohort_years:
            params = interpolate_params(entry_year, TRAINING, OPTIMIZED, k)
            result = simulate(
                params,
                T,
                S0,
                v0,
                Smax_base,
                delta_Smax,
                Umax,
                u0,
                enable_frontier,
                gamma,
            )
            w_thr = weeks_to_reach(result.S, threshold)
            cohort_reach.append(None if w_thr is None else w_thr / 52)

        for year_value in years:
            images.append(
                render_research_cohort_frame(
                    year_value,
                    cohort_years,
                    cohort_reach,
                    T,
                    v0,
                    Smax_base,
                    delta_Smax,
                    gamma,
                    k,
                )
            )

    out_path = out_dir / f"research_progress_{view}.gif"
    imageio.mimsave(out_path, images, fps=fps)
    return out_path


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "research_progress":
        start_year, end_year = args.years
        export_research_progress_gif(
            start_year,
            end_year,
            args.frames,
            args.fps,
            out_dir,
            args.research_view,
            args.T,
            args.S0,
            args.v0,
            args.threshold,
            args.Smax_base,
            args.delta_Smax,
            args.gamma,
            args.Umax,
            args.u0,
            args.enable_frontier,
            args.k,
        )
        return

    params = preset_params(args.preset)

    targets = list(PARAM_KEYS) if args.all else [args.param]
    if not all(targets):
        raise SystemExit("Specify --param or use --all.")

    for param in targets:
        export_gif(
            param,
            params.copy(),
            args.min_value,
            args.max_value,
            args.frames,
            args.fps,
            out_dir,
            args.T,
            args.S0,
            args.v0,
            args.threshold,
            args.Smax_base,
            args.delta_Smax,
            args.gamma,
            args.Umax,
            args.u0,
            args.enable_frontier,
            args.compare3,
        )


if __name__ == "__main__":
    main()
