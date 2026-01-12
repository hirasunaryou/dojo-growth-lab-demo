"""
Generate GIFs that sweep one parameter (N/F/D/V/R) at a time for PPT assets.
Usage:
  python scripts/export_gifs.py --preset dojo --param N --min 0.2 --max 0.9 --frames 30 --fps 10 --out out/gifs
  python scripts/export_gifs.py --all --preset dojo
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


BASELINE = {"N": 0.35, "F": 0.35, "D": 0.40, "V": 0.30, "R": 0.35}
DOJO = {"N": 0.80, "F": 0.70, "D": 0.60, "V": 0.75, "R": 0.70}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export N/F/D/V/R sweep GIFs.")
    parser.add_argument("--preset", choices=["baseline", "dojo", "custom"], default="dojo")
    parser.add_argument("--param", choices=PARAM_KEYS, help="Parameter to sweep.")
    parser.add_argument("--all", action="store_true", help="Generate GIFs for all parameters.")
    parser.add_argument("--min", dest="min_value", type=float, default=0.2)
    parser.add_argument("--max", dest="max_value", type=float, default=0.9)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=str, default="out/gifs")
    parser.add_argument("--T", type=int, default=52)
    parser.add_argument("--S0", type=float, default=50.0)
    parser.add_argument("--v0", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument(
        "--disable-frontier",
        action="store_false",
        dest="enable_frontier",
        help="Disable Frontier Unlock (Smax uplift).",
    )
    parser.add_argument("--Smax-base", type=float, default=80.0)
    parser.add_argument("--delta-Smax", type=float, default=12.0)
    parser.add_argument("--Umax", type=float, default=10.0)
    parser.add_argument("--u0", type=float, default=2.0)
    return parser


def preset_params(preset: str) -> dict[str, float]:
    if preset == "baseline":
        return BASELINE.copy()
    if preset == "dojo":
        return DOJO.copy()
    return DOJO.copy()


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
) -> np.ndarray:
    """Render a single frame to an RGB array."""
    result = simulate(params, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)

    fig, (ax_skill, ax_smax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        constrained_layout=True,
    )

    # Skill curve
    ax_skill.plot(result.week, result.S, color="#1f77b4", label="Skill S(t)")
    ax_skill.axhline(threshold, linestyle="--", linewidth=1, color="#666666")
    ax_skill.set_xlim(0, T)
    ax_skill.set_ylim(0, 100)
    ax_skill.set_xlabel("Week")
    ax_skill.set_ylabel("Skill S")
    ax_skill.set_title("Skill curve S(t)")

    # Ceiling curve
    ax_smax.plot(result.week, result.Smax, color="#ff7f0e", label="Smax(t)")
    ax_smax.set_xlim(0, T)
    ax_smax.set_ylim(Smax_base, Smax_base + delta_Smax)
    ax_smax.set_xlabel("Week")
    ax_smax.set_ylabel("Smax")

    # Overlay metrics (PPT-friendly)
    w_thr = weeks_to_reach(result.S, threshold)
    w_thr_label = f"{w_thr}w" if w_thr is not None else "not reached"
    info = (
        f"{sweep_param}={params[sweep_param]:.2f} | v={result.v:.3f}/week | "
        f"S@end={result.S[-1]:.1f} | Smax@end={result.Smax[-1]:.1f} | "
        f"reach {threshold}: {w_thr_label}"
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
            )
        )

    out_path = out_dir / f"sweep_{sweep_param}.gif"
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
        )


if __name__ == "__main__":
    main()
