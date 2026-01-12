"""
Generate GIFs that sweep one parameter (N/F/D/V/R) at a time for PPT assets.
Usage:
  python scripts/export_gifs.py --preset optimized --param N --min 0.2 --max 0.9 --frames 30 --fps 10 --out out/gifs
  python scripts/export_gifs.py --all --preset optimized
  python scripts/export_gifs.py --param V --compare3 --out out/gifs
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

from core.growth_model import PARAM_KEYS, simulate, weeks_to_reach

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NO_SYSTEM = {"N": 0.25, "F": 0.15, "D": 0.20, "V": 0.10, "R": 0.15}
TRAINING = {"N": 0.40, "F": 0.40, "D": 0.35, "V": 0.30, "R": 0.35}
OPTIMIZED = {"N": 0.80, "F": 0.70, "D": 0.60, "V": 0.75, "R": 0.70}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export N/F/D/V/R sweep GIFs.")
    parser.add_argument(
        "--preset",
        choices=["no_system", "training", "optimized", "custom"],
        default="optimized",
    )
    parser.add_argument("--param", choices=PARAM_KEYS, help="Parameter to sweep.")
    parser.add_argument("--all", action="store_true", help="Generate GIFs for all parameters.")
    parser.add_argument(
        "--compare3",
        action="store_true",
        help="Render No system / Training / Optimized comparison in each frame.",
    )
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
    parser.add_argument("--Umax", type=float, default=10.0)
    parser.add_argument("--u0", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    return parser


def preset_params(preset: str) -> dict[str, float]:
    if preset == "no_system":
        return NO_SYSTEM.copy()
    if preset == "training":
        return TRAINING.copy()
    if preset == "optimized":
        return OPTIMIZED.copy()
    return OPTIMIZED.copy()


def render_frame(
    params: dict[str, float],
    sweep_param: str,
    T: int,
    S0: float,
    v0: float,
    threshold: float,
    Smax_base: float,
    delta_Smax: float,
    Umax: float,
    u0: float,
    enable_frontier: bool,
    gamma: float,
    compare3: bool,
) -> np.ndarray:
    """Render a single frame to an RGB array."""
    years = T / 52
    # The optimized curve is the sweep target; other scenarios stay fixed in compare3 mode.
    result = simulate(
        params, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma=gamma
    )
    no_result = simulate(
        NO_SYSTEM, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma=gamma
    )
    train_result = simulate(
        TRAINING, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma=gamma
    )

    fig, (ax_skill, ax_smax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        constrained_layout=True,
    )

    # Skill curve
    if compare3:
        ax_skill.plot(no_result.week / 52, no_result.S, label="No system", color="#7f7f7f")
        ax_skill.plot(train_result.week / 52, train_result.S, label="Training", color="#2ca02c")
        ax_skill.plot(result.week / 52, result.S, label="Optimized", color="#1f77b4")
        ax_skill.legend(loc="lower right")
    else:
        ax_skill.plot(result.week / 52, result.S, color="#1f77b4", label="Skill S(t)")
    ax_skill.axhline(threshold, linestyle="--", linewidth=1, color="#666666")
    ax_skill.set_xlim(0, years)
    ax_skill.set_ylim(0, 100)
    ax_skill.set_xlabel("Year")
    ax_skill.set_ylabel("Skill S")
    ax_skill.set_title("Skill curve S(t)")

    # Ceiling curve
    if compare3:
        ax_smax.plot(no_result.week / 52, no_result.Smax, label="No system", color="#7f7f7f")
        ax_smax.plot(train_result.week / 52, train_result.Smax, label="Training", color="#2ca02c")
        ax_smax.plot(result.week / 52, result.Smax, label="Optimized", color="#ff7f0e")
    else:
        ax_smax.plot(result.week / 52, result.Smax, color="#ff7f0e", label="Smax(t)")
    ax_smax.set_xlim(0, years)
    ax_smax.set_ylim(0, 100)
    ax_smax.set_xlabel("Year")
    ax_smax.set_ylabel("Smax")

    # Overlay metrics (PPT-friendly)
    w_thr = weeks_to_reach(result.S, threshold)
    w_thr_label = f"{w_thr / 52:.2f}y" if w_thr is not None else "not reached"
    info = (
        f"{sweep_param}={params[sweep_param]:.2f} | v={result.v:.3f}/week | "
        f"S@end={result.S[-1]:.1f} | Smax@end={result.Smax[-1]:.1f} | "
        f"Smax_cap={result.Smax_cap:.1f} | reach {threshold}: {w_thr_label}"
    )
    if compare3:
        # Show quick comparison for each scenario on separate lines.
        no_thr = weeks_to_reach(no_result.S, threshold)
        train_thr = weeks_to_reach(train_result.S, threshold)
        no_label = f"{no_thr / 52:.2f}y" if no_thr is not None else "not reached"
        train_label = f"{train_thr / 52:.2f}y" if train_thr is not None else "not reached"
        info = (
            f"Optimized: {info}\n"
            f"No system: v={no_result.v:.3f}/week | S@end={no_result.S[-1]:.1f} | "
            f"Smax@end={no_result.Smax[-1]:.1f} | Smax_cap={no_result.Smax_cap:.1f} | "
            f"reach {threshold}: {no_label}\n"
            f"Training: v={train_result.v:.3f}/week | S@end={train_result.S[-1]:.1f} | "
            f"Smax@end={train_result.Smax[-1]:.1f} | Smax_cap={train_result.Smax_cap:.1f} | "
            f"reach {threshold}: {train_label}"
        )
    ax_skill.text(0.01, 0.98, info, transform=ax_skill.transAxes, va="top", fontsize=10)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    image = imageio.imread(buffer)
    return image


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
    Umax: float,
    u0: float,
    enable_frontier: bool,
    gamma: float,
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
                Umax,
                u0,
                enable_frontier,
                gamma,
                compare3,
            )
        )

    suffix = "compare3" if compare3 else "single"
    out_path = out_dir / f"sweep_{sweep_param}_{suffix}.gif"
    imageio.mimsave(out_path, images, fps=fps)
    return out_path


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            args.Umax,
            args.u0,
            args.enable_frontier,
            args.gamma,
            args.compare3,
        )


if __name__ == "__main__":
    main()
