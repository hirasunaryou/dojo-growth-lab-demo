# growth_lab_streamlit.py
# Interactive N-F-D-V-R simulator (Streamlit)
# Run: streamlit run growth_lab_streamlit.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.constants import PARAM_KEYS
from core.growth_model import (
    simulate,
    simulate_research_progress,
    smax_cap_from_params,
    v_from_params,
    weeks_to_reach,
)
from core.roadmap import TARGET_PARAMS, build_year_table, interpolate_params
from core.scenarios import build_scenarios

st.set_page_config(page_title="DoJo Growth Lab", layout="wide")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")
# Fixed reference lines (adjustable later).
TARGET = TARGET_PARAMS.copy()

research_mode = st.sidebar.toggle("Research Progress Mode", value=False)

# Optimized sliders are the only editable line (per spec).
# When research mode is on, we still show the sliders but disable them to avoid confusion.
sliders_disabled = research_mode
p = {
    "N": st.sidebar.slider("N (Trials)", 0.0, 1.0, TARGET["N"], 0.01, disabled=sliders_disabled),
    "F": st.sidebar.slider("F (Feedback)", 0.0, 1.0, TARGET["F"], 0.01, disabled=sliders_disabled),
    "D": st.sidebar.slider("D (Depth)", 0.0, 1.0, TARGET["D"], 0.01, disabled=sliders_disabled),
    "V": st.sidebar.slider("V (Variety)", 0.0, 1.0, TARGET["V"], 0.01, disabled=sliders_disabled),
    "R": st.sidebar.slider("R (Repetition)", 0.0, 1.0, TARGET["R"], 0.01, disabled=sliders_disabled),
}

# Scenario configs (shared labels/colors across UI + exports).
scenarios = build_scenarios(optimized_params=p)
scenario_lookup = {scenario.name: scenario for scenario in scenarios}
no_params = scenario_lookup["No system"].params
training_params = scenario_lookup["Training system"].params
optimized_params = scenario_lookup["Optimized"].params

st.sidebar.markdown("---")
st.sidebar.subheader("Model knobs (advanced)")
years = st.sidebar.slider("Horizon (years)", 1, 15, 10, 1)
T = years * 52
S0 = st.sidebar.slider("Start skill S0", 0.0, 100.0, 20.0, 1.0)
v0 = st.sidebar.slider("v0 (velocity base)", 0.01, 0.20, 0.03, 0.01)
threshold = st.sidebar.slider("Threshold", 0.0, 100.0, 75.0, 1.0)

enable_frontier = st.sidebar.checkbox("Enable Frontier (Smax uplift via U)", value=True)
Smax_base = st.sidebar.slider("Smax base", 50.0, 100.0, 80.0, 1.0)
delta_Smax = st.sidebar.slider("Smax uplift (max)", 0.0, 30.0, 15.0, 1.0)
gamma = st.sidebar.slider("gamma (frontier strength)", 0.1, 1.0, 0.5, 0.05)
Umax = st.sidebar.slider("Umax", 1.0, 30.0, 10.0, 1.0)
u0 = st.sidebar.slider("u0 (unlock speed)", 0.1, 5.0, 2.0, 0.1)
fix_axes = st.sidebar.toggle("軸固定 (Fix axes)", value=True)
show_params_on_plot = st.sidebar.checkbox("Show scenario params on plot", value=False)

if research_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Research Progress Mode")
    research_years = st.sidebar.slider("Research years (0-5)", 0, 5, 3, 1)
    k = st.sidebar.slider("k (maturity speed)", 0.1, 2.0, 1.0, 0.05)
    research_view = st.sidebar.radio("Display mode", ["Continuous", "Cohort"], horizontal=True)
else:
    research_years = 0
    k = 1.0
    research_view = "Continuous"

# -----------------------------
# Helper functions
# -----------------------------

def summarize(df, result):
    """Summarize output metrics for a static-parameter simulation."""
    S_end = float(df["S"].iloc[-1])
    S_12 = float(df["S"].iloc[min(12, len(df) - 1)])
    S_26 = float(df["S"].iloc[min(26, len(df) - 1)])
    Smax_end = float(df["Smax"].iloc[-1])
    w_thr = weeks_to_reach(df["S"].values, threshold)
    years_thr = None if w_thr is None else w_thr / 52
    return {
        "v (per week)": result.v,
        "S @ 12w": S_12,
        "S @ 26w": S_26,
        "S @ end": S_end,
        "Smax @ end": Smax_end,
        "Smax cap": result.Smax_cap,
        f"Weeks to reach {threshold}": w_thr,
        f"Years to reach {threshold}": years_thr,
    }


def build_scenario_table(
    scenarios,
    results,
    threshold_value: float,
) -> pd.DataFrame:
    """Build a scenario parameter table for always-visible comparison."""
    rows = []
    for scenario in scenarios:
        result = results[scenario.name]
        w_thr = weeks_to_reach(result.S, threshold_value)
        years_thr = None if w_thr is None else w_thr / 52
        # Keep columns aligned with the teaching goals: inputs -> derived metrics -> outcomes.
        row = {
            "Scenario": scenario.name,
            "N": scenario.params["N"],
            "F": scenario.params["F"],
            "D": scenario.params["D"],
            "V": scenario.params["V"],
            "R": scenario.params["R"],
            "V*D": scenario.params["V"] * scenario.params["D"],
            "v": result.v,
            "Smax_cap": result.Smax_cap,
            "S_end": float(result.S[-1]),
            "Smax_end": float(result.Smax[-1]),
            "years_to_threshold": years_thr,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def highlight_bottleneck(row: pd.Series) -> list[str]:
    """Highlight the minimum N/F/D/V/R per scenario for bottleneck insight."""
    keys = ["N", "F", "D", "V", "R"]
    min_value = row[keys].min()
    styles = []
    for key in row.index:
        if key in keys and row[key] == min_value:
            styles.append("background-color: #ffe8e6")
        else:
            styles.append("")
    return styles


def cohort_years_to_reach(params: dict[str, float]) -> float | None:
    """Reference: time to threshold if the system is frozen at year y."""
    cohort_result = simulate(
        params, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
    )
    w_thr = weeks_to_reach(cohort_result.S, threshold)
    return None if w_thr is None else w_thr / 52


# -----------------------------
# Layout
# -----------------------------
st.title("DoJo Growth Lab — Move N-F-D-V-R and *see* growth")
# Always-visible formulas for quick explanation in workshops.
st.latex(r"\frac{dS}{dt} = v \times (S_{max} - S)")
st.latex(r"v = v_0 \times (N \times F \times D \times V \times R)^{1/5}")
st.latex(
    r"u\_rate = u_0 \times (V \times D)"
    r",\;\; U_t = U_{t-1} + u\_rate \times (1 - U_{t-1}/U_{max})"
    r",\;\; S_{max}(t) = S_{max\_base} + \Delta S_{max} \times (V D)^{\gamma} \times (U_t/U_{max})"
)

with st.expander("パラメータの説明（日本語ガイド / JP Guide）", expanded=True):
    st.markdown(
        """
        - **N（試行回数 / Trials）**：練習ループの回転数。AIロールプレイ等で増やせる。  
          **N↑ → 伸び始めが速くなる。**
        - **F（フィードバック密度 / Feedback）**：即時・定量・ブレないFB（ルーブリック採点等）で増やせる。  
          **F↑ → 伸びの角度が安定して上がる。**
        - **D（深さ / Depth）**：難易度/複雑性。背伸び可能な領域を狙う。  
          **D↑ → 学習の上限（Smax）が押し上がる。**
        - **V（多様性 / Variety）**：ケース分布の広さ。長尾・未知パターン・環境変化を含める。  
          **V↑ → Frontier Unlock が進み、Smaxが上がりやすい。**
        - **R（反復 / Repetition）**：弱点に戻って再挑戦できる設計（復習タイミング・再出題）。  
          **R↑ → 学習が収束しやすくなる。**
        - **V × D**：Frontier Unlock を押し上げ、**Smaxが伸びる**（上限の拡張）。
          gamma が大きいほど、V×D の影響が強くなる。
        """
    )

# -----------------------------
# Scenario parameter table (always visible)
# -----------------------------
# We precompute the three core scenarios so the table is consistent across modes.
scenario_results = {
    scenario.name: simulate(
        scenario.params, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
    )
    for scenario in scenarios
}
scenario_table = build_scenario_table(scenarios, scenario_results, threshold)
scenario_table = scenario_table[
    [
        "Scenario",
        "N",
        "F",
        "D",
        "V",
        "R",
        "V*D",
        "v",
        "Smax_cap",
        "S_end",
        "Smax_end",
        "years_to_threshold",
    ]
]
st.subheader("Scenario Parameter Table (N-F-D-V-R + derived metrics)")
st.dataframe(
    scenario_table.set_index("Scenario")
    .style.apply(highlight_bottleneck, axis=1)
    .format(precision=3),
    use_container_width=True,
)

# Bottleneck text guidance for workshop facilitation.
st.subheader("改善優先候補（ボトルネック）")
for _, row in scenario_table.iterrows():
    min_value = min(row["N"], row["F"], row["D"], row["V"], row["R"])
    min_params = [key for key in ["N", "F", "D", "V", "R"] if row[key] == min_value]
    st.write(f"{row['Scenario']}: 改善優先候補 = {', '.join(min_params)}")

# Contribution chart (optional but recommended): log-decomposed v uplift vs No system.
st.subheader("v寄与の分解（No system基準 / Training + Research uplift）")
contrib_keys = list(PARAM_KEYS)
ref_row = scenario_table.loc[scenario_table["Scenario"] == "No system"].iloc[0]
training_row = scenario_table.loc[scenario_table["Scenario"] == "Training system"].iloc[0]
optimized_row = scenario_table.loc[scenario_table["Scenario"] == "Optimized"].iloc[0]

# Use a small clip to avoid log(0) and keep the uplift math stable.
ref_vals = np.clip(ref_row[contrib_keys].to_numpy(dtype=float), 1e-6, None)
training_vals = np.clip(training_row[contrib_keys].to_numpy(dtype=float), 1e-6, None)
optimized_vals = np.clip(optimized_row[contrib_keys].to_numpy(dtype=float), 1e-6, None)

# v = v0 * (N*F*D*V*R)^(1/5) -> Δlog(v) is the sum of per-parameter log-diffs / 5.
c_train = (np.log(training_vals) - np.log(ref_vals)) / 5
c_extra = (np.log(optimized_vals) - np.log(training_vals)) / 5

fig_contrib, ax_contrib = plt.subplots()
ax_contrib.bar(
    contrib_keys,
    c_train,
    color=scenario_lookup["Training system"].color,
    label="Training uplift",
)
ax_contrib.bar(
    contrib_keys,
    c_extra,
    bottom=c_train,
    color=scenario_lookup["Optimized"].color,
    label="Extra uplift (Optimized-Training)",
)
ax_contrib.axhline(0, color="#999999", linewidth=1)
ax_contrib.set_ylabel("Δlog(v) contribution")
ax_contrib.set_title("v寄与の分解（No system基準 / Training + Research uplift）")
ax_contrib.legend()
st.pyplot(fig_contrib)

if research_mode:
    # -----------------------------
    # Research Progress Mode
    # -----------------------------
    no_result = scenario_results["No system"]
    training_result = scenario_results["Training system"]
    research_result = simulate_research_progress(
        training_params,
        optimized_params,
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

    df_no = pd.DataFrame(
        {"week": no_result.week, "S": no_result.S, "Smax": no_result.Smax, "U": no_result.U}
    )
    df_training = pd.DataFrame(
        {
            "week": training_result.week,
            "S": training_result.S,
            "Smax": training_result.Smax,
            "U": training_result.U,
        }
    )
    df_research = pd.DataFrame(
        {
            "week": research_result.week,
            "S": research_result.S,
            "Smax": research_result.Smax,
            "U": research_result.U,
            "v": research_result.v,
            "Smax_cap": research_result.Smax_cap,
        }
    )

    df_no["year"] = df_no["week"] / 52
    df_training["year"] = df_training["week"] / 52
    df_research["year"] = df_research["week"] / 52

    # Build the year-by-year table for display (N/F/D/V/R + v + Smax_cap).
    year_rows = build_year_table(range(research_years + 1), training_params, optimized_params, k)
    for row in year_rows:
        params = {key: row[key] for key in PARAM_KEYS}
        row["v(y)"] = v_from_params(params, v0=v0)
        row["Smax_cap(y)"] = smax_cap_from_params(params, Smax_base, delta_Smax, gamma)
        # Cohort reference: new hire at this year with parameters frozen.
        row["Cohort years to reach threshold"] = cohort_years_to_reach(params)
    df_years = pd.DataFrame(year_rows)

    # Summaries (continuous mode uses evolving params).
    w_thr_research = weeks_to_reach(df_research["S"].values, threshold)
    years_thr_research = None if w_thr_research is None else w_thr_research / 52

    # Additional reference points to explain threshold reach in a simple way.
    year0_params = interpolate_params(0, training_params, optimized_params, k)
    yearN_params = interpolate_params(research_years, training_params, optimized_params, k)
    year0_reach = cohort_years_to_reach(year0_params)
    yearN_reach = cohort_years_to_reach(yearN_params)

    colA, colB = st.columns([2, 1])

    with colA:
        if research_view == "Continuous":
            fig, ax = plt.subplots()
            ax.plot(df_no["year"], df_no["S"], label="No system (no training)")
            ax.plot(df_training["year"], df_training["S"], label="Training system")
            ax.plot(df_research["year"], df_research["S"], label="Research-progress (same person)")
            ax.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold {threshold}")
            # Year boundaries show that parameters improve annually.
            for year_mark in range(1, years + 1):
                ax.axvline(year_mark, linestyle=":", linewidth=0.8, color="#999999")
                ax.text(
                    year_mark,
                    98,
                    f"Year {year_mark}",
                    rotation=90,
                    va="top",
                    fontsize=7,
                    color="#666666",
                )
            ax.set_xlabel("Year")
            ax.set_ylabel("Skill S")
            ax.set_title("Skill curve S(t) with Research Progress")
            ax.legend()
            if fix_axes:
                ax.set_xlim(0, years)
                ax.set_ylim(0, 100)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.plot(df_no["year"], df_no["Smax"], label="No system: Smax")
            ax2.plot(df_training["year"], df_training["Smax"], label="Training: Smax")
            ax2.plot(df_research["year"], df_research["Smax"], label="Research-progress: Smax")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Smax (ceiling)")
            ax2.set_title("Ceiling Smax(t)")
            ax2.legend()
            if fix_axes:
                ax2.set_xlim(0, years)
                ax2.set_ylim(0, 100)
            st.pyplot(fig2)
        else:
            # Cohort view: how long each entry year takes to reach the threshold.
            cohort_years = df_years["year"].tolist()
            reach_values = df_years["Cohort years to reach threshold"].tolist()
            fig, ax = plt.subplots()
            bars = ax.bar(cohort_years, [v or 0 for v in reach_values], color="#1f77b4")
            # Annotate bars with year counts (or "NR" if not reached).
            for bar, value in zip(bars, reach_values):
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
            ax.set_ylabel(f"Years to reach {threshold}")
            ax.set_title("Cohort view: later years reach Veteran faster")
            ax.set_ylim(0, years)
            st.pyplot(fig)

    with colB:
        st.subheader("Research progress table (year-by-year)")
        st.dataframe(
            df_years.set_index("year").round(3),
            use_container_width=True,
        )

        st.subheader("Threshold reach (explainable reference)")
        st.write(
            "Continuous mode is a *single person* evolving with the system. "
            "To keep it explainable, we also show cohort-style reference times below."
        )
        st.write(
            {
                "Evolving system (actual reach)": years_thr_research,
                "Year 0 system (cohort reference)": year0_reach,
                f"Year {research_years} system (cohort reference)": yearN_reach,
            }
        )

    st.caption(
        "Note: Research Progress Mode visualizes how the *system* improves year by year. "
        "The maturity curve is saturating (m(y)=1-exp(-k y))."
    )

else:
    # -----------------------------
    # Baseline Mode (No system / Training / Optimized)
    # -----------------------------
    no_result = scenario_results["No system"]
    training_result = scenario_results["Training system"]
    opt_result = scenario_results["Optimized"]

    df_no = pd.DataFrame(
        {"week": no_result.week, "S": no_result.S, "Smax": no_result.Smax, "U": no_result.U}
    )
    df_training = pd.DataFrame(
        {
            "week": training_result.week,
            "S": training_result.S,
            "Smax": training_result.Smax,
            "U": training_result.U,
        }
    )
    df_opt = pd.DataFrame(
        {"week": opt_result.week, "S": opt_result.S, "Smax": opt_result.Smax, "U": opt_result.U}
    )
    # Convert weeks to years for a 10-year view.
    df_no["year"] = df_no["week"] / 52
    df_training["year"] = df_training["week"] / 52
    df_opt["year"] = df_opt["week"] / 52

    sum_no = summarize(df_no, no_result)
    sum_training = summarize(df_training, training_result)
    sum_opt = summarize(df_opt, opt_result)

    # Show the current slider-derived v and Smax ceiling span.
    st.write(
        f"Optimized の v（成長速度）= **{opt_result.v:.3f} / week** ｜ "
        f"Smax_cap（理論上限）= **{opt_result.Smax_cap:.1f}** ｜ "
        f"Smax_end（最終週のSmax）= **{opt_result.Smax[-1]:.1f}**"
    )

    colA, colB = st.columns([2, 1])

    with colA:
        # Plot skill curves + threshold
        fig = plt.figure()
        plt.plot(df_no["year"], df_no["S"], label="No system (no training)")
        plt.plot(df_training["year"], df_training["S"], label="Training system")
        plt.plot(df_opt["year"], df_opt["S"], label="Optimized (DoJo)")
        plt.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold {threshold}")
        plt.xlabel("Year")
        plt.ylabel("Skill S")
        plt.title("Skill curve S(t)")
        plt.legend()
        ax = plt.gca()
        if show_params_on_plot:
            # Small end-of-line labels to reinforce "this curve = this system".
            for scenario in scenarios:
                label = (
                    f"N={scenario.params['N']:.2f} "
                    f"F={scenario.params['F']:.2f} "
                    f"D={scenario.params['D']:.2f} "
                    f"V={scenario.params['V']:.2f} "
                    f"R={scenario.params['R']:.2f} | "
                    f"v={scenario_results[scenario.name].v:.3f} | "
                    f"Smax_cap={scenario_results[scenario.name].Smax_cap:.1f}"
                )
                ax.text(
                    years * 0.98,
                    scenario_results[scenario.name].S[-1],
                    label,
                    fontsize=7,
                    ha="right",
                    va="center",
                    color=scenario.color,
                )
        if fix_axes:
            plt.xlim(0, years)
            plt.ylim(0, 100)
        st.pyplot(fig)

        fig2 = plt.figure()
        plt.plot(df_no["year"], df_no["Smax"], label="No system: Smax")
        plt.plot(df_training["year"], df_training["Smax"], label="Training: Smax")
        plt.plot(df_opt["year"], df_opt["Smax"], label="Optimized: Smax")
        plt.xlabel("Year")
        plt.ylabel("Smax (ceiling)")
        plt.title("Ceiling Smax(t)")
        plt.legend()
        if fix_axes:
            plt.xlim(0, years)
            plt.ylim(0, 100)
        st.pyplot(fig2)

    with colB:
        st.subheader("N-F-D-V-R (0–1)")
        st.write(pd.DataFrame({"param": list(p.keys()), "value": list(p.values())}).set_index("param"))

        st.subheader("Metrics (No system / Training / Optimized)")
        df_metrics = pd.DataFrame(
            [sum_no, sum_training, sum_opt], index=["No system", "Training", "Optimized"]
        )
        st.dataframe(df_metrics)

        st.subheader("One-at-a-time sensitivity (rough)")
        # Increase each param by +0.10 (capped at 1.0) and measure delta in S@end
        sens_rows = []
        base_S_end = sum_opt["S @ end"]
        for key in PARAM_KEYS:
            p2 = p.copy()
            p2[key] = min(1.0, p2[key] + 0.10)
            sim2 = simulate(
                p2, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
            )
            delta = float(sim2.S[-1] - base_S_end)
            sens_rows.append({"param": key, "ΔS@end (+0.10)": delta})
        sens = pd.DataFrame(sens_rows).sort_values("ΔS@end (+0.10)", ascending=False)
        st.dataframe(sens)

st.caption(
    "Note: This is a conceptual simulator to explain *why* improving N/F/D/V/R changes growth. "
    "Calibrate later with your real learning curves & KPIs."
)
