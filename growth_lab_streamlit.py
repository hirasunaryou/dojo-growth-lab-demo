\
# growth_lab_streamlit.py
# Interactive N-F-D-V-R simulator (Streamlit)
# Run: streamlit run growth_lab_streamlit.py

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core.growth_model import PARAM_KEYS, simulate, weeks_to_reach

st.set_page_config(page_title="DoJo Growth Lab", layout="wide")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

# Fixed reference scenarios (editable later from code).
# Reference baselines are fixed so the story stays consistent across workshops.
no_system = {"N": 0.25, "F": 0.15, "D": 0.20, "V": 0.10, "R": 0.15}
training = {"N": 0.40, "F": 0.40, "D": 0.35, "V": 0.30, "R": 0.35}
optimized_default = {"N": 0.80, "F": 0.70, "D": 0.60, "V": 0.75, "R": 0.70}

# Optimized sliders are the editable scenario for workshops.
p = {
    "N": st.sidebar.slider("N (Trials)", 0.0, 1.0, optimized_default["N"], 0.01),
    "F": st.sidebar.slider("F (Feedback)", 0.0, 1.0, optimized_default["F"], 0.01),
    "D": st.sidebar.slider("D (Depth)", 0.0, 1.0, optimized_default["D"], 0.01),
    "V": st.sidebar.slider("V (Variety)", 0.0, 1.0, optimized_default["V"], 0.01),
    "R": st.sidebar.slider("R (Repetition)", 0.0, 1.0, optimized_default["R"], 0.01),
}

st.sidebar.markdown("---")
st.sidebar.subheader("Model knobs (advanced)")
years = st.sidebar.slider("Horizon (years)", 1, 20, 10, 1)
# We still simulate weekly, but plot in years for the 10-year narrative.
T = years * 52
S0 = st.sidebar.slider("Start skill S0", 0.0, 100.0, 20.0, 1.0)
v0 = st.sidebar.slider("v0 (velocity base)", 0.005, 0.20, 0.03, 0.005)
threshold = st.sidebar.slider("Threshold", 0.0, 100.0, 75.0, 1.0)

enable_frontier = st.sidebar.checkbox("Enable Frontier (Smax uplift via U)", value=True)
Smax_base = st.sidebar.slider("Smax base", 50.0, 100.0, 80.0, 1.0)
delta_Smax = st.sidebar.slider("Smax uplift (max)", 0.0, 30.0, 15.0, 1.0)
Umax = st.sidebar.slider("Umax", 1.0, 30.0, 10.0, 1.0)
u0 = st.sidebar.slider("u0 (unlock speed)", 0.1, 5.0, 2.0, 0.1)
gamma = st.sidebar.slider("gamma (V×D effect)", 0.1, 1.0, 0.5, 0.05)
fix_axes = st.sidebar.toggle("軸固定 (Fix axes)", value=True)

# -----------------------------
# Run scenarios: no system / training / optimized
# -----------------------------
no_result = simulate(
    no_system, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma=gamma
)
train_result = simulate(
    training, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma=gamma
)
opt_result = simulate(
    p, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma=gamma
)

def as_df(result: "SimulationResult") -> pd.DataFrame:
    # Convert weeks into years for 10-year scale readability.
    return pd.DataFrame(
        {
            "year": result.week / 52,
            "S": result.S,
            "Smax": result.Smax,
            "U": result.U,
        }
    )

df_no = as_df(no_result)
df_train = as_df(train_result)
df_opt = as_df(opt_result)

# -----------------------------
# Summary metrics
# -----------------------------
def summarize(result):
    # Keep metrics aligned with the workshop narrative (10-year horizon).
    S_end = float(result.S[-1])
    Smax_end = float(result.Smax[-1])
    w_thr = weeks_to_reach(result.S, threshold)
    years_thr = None if w_thr is None else round(w_thr / 52, 2)
    return {
        "v (per week)": result.v,
        "S @ end": S_end,
        "Smax @ end": Smax_end,
        "Smax cap": result.Smax_cap,
        f"Weeks to reach {threshold}": w_thr,
        f"Years to reach {threshold}": years_thr,
    }

sum_no = summarize(no_result)
sum_train = summarize(train_result)
sum_opt = summarize(opt_result)

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
    r",\;\; S_{max}(t) = S_{max\_base} + \Delta S_{max} \times (V \times D)^{\gamma} \times (U_t/U_{max})"
)
# Show the current slider-derived v and Smax ceilings.
st.write(
    f"現在の v（成長速度）= **{opt_result.v:.3f} / week** ｜ "
    f"Smax_cap（理論上限）= **{opt_result.Smax_cap:.1f}** ｜ "
    f"Smax_end（最終週）= **{opt_result.Smax[-1]:.1f}**"
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
        - **V × D**：Frontier Unlock を押し上げ、**Smax_cap（理論上限）**が伸びる。
        """
    )

colA, colB = st.columns([2, 1])

with colA:
    # Plot skill curves + threshold
    fig = plt.figure()
    plt.plot(df_no["year"], df_no["S"], label="No system")
    plt.plot(df_train["year"], df_train["S"], label="Training system")
    plt.plot(df_opt["year"], df_opt["S"], label="Optimized (DoJo)")
    plt.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold {threshold}")
    plt.xlabel("Year")
    plt.ylabel("Skill S")
    plt.title("Skill curve S(t)")
    plt.legend()
    if fix_axes:
        plt.xlim(0, years)
        plt.ylim(0, 100)
    st.pyplot(fig)

    fig2 = plt.figure()
    plt.plot(df_no["year"], df_no["Smax"], label="No system: Smax")
    plt.plot(df_train["year"], df_train["Smax"], label="Training: Smax")
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
        [sum_no, sum_train, sum_opt],
        index=["No system", "Training", "Optimized"],
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
            p2, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma=gamma
        )
        delta = float(sim2.S[-1] - base_S_end)
        sens_rows.append({"param": key, "ΔS@end (+0.10)": delta})
    sens = pd.DataFrame(sens_rows).sort_values("ΔS@end (+0.10)", ascending=False)
    st.dataframe(sens)

st.caption("Note: This is a conceptual simulator to explain *why* improving N/F/D/V/R changes growth. Calibrate later with your real learning curves & KPIs.")
