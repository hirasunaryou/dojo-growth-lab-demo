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
preset = st.sidebar.selectbox("Preset", ["Baseline (No AI)", "DoJo (With AI)", "Custom"])

baseline = {"N": 0.35, "F": 0.35, "D": 0.40, "V": 0.30, "R": 0.35}
dojo =     {"N": 0.80, "F": 0.70, "D": 0.60, "V": 0.75, "R": 0.70}

if preset == "Baseline (No AI)":
    p = baseline.copy()
elif preset == "DoJo (With AI)":
    p = dojo.copy()
else:
    p = {
        "N": st.sidebar.slider("N (Trials)", 0.0, 1.0, 0.80, 0.01),
        "F": st.sidebar.slider("F (Feedback)", 0.0, 1.0, 0.70, 0.01),
        "D": st.sidebar.slider("D (Depth)", 0.0, 1.0, 0.60, 0.01),
        "V": st.sidebar.slider("V (Variety)", 0.0, 1.0, 0.75, 0.01),
        "R": st.sidebar.slider("R (Repetition)", 0.0, 1.0, 0.70, 0.01),
    }

st.sidebar.markdown("---")
st.sidebar.subheader("Model knobs (advanced)")
T = st.sidebar.slider("Horizon (weeks)", 4, 104, 52, 1)
S0 = st.sidebar.slider("Start skill S0", 0.0, 100.0, 50.0, 1.0)
v0 = st.sidebar.slider("v0 (velocity base)", 0.01, 0.60, 0.25, 0.01)
threshold = st.sidebar.slider("Threshold", 0.0, 100.0, 75.0, 1.0)

enable_frontier = st.sidebar.checkbox("Enable Frontier (Smax uplift via U)", value=True)
Smax_base = st.sidebar.slider("Smax base", 50.0, 100.0, 80.0, 1.0)
delta_Smax = st.sidebar.slider("Smax uplift (max)", 0.0, 30.0, 12.0, 1.0)
Umax = st.sidebar.slider("Umax", 1.0, 30.0, 10.0, 1.0)
u0 = st.sidebar.slider("u0 (unlock speed)", 0.1, 5.0, 2.0, 0.1)
fix_axes = st.sidebar.toggle("軸固定 (Fix axes)", value=True)

# -----------------------------
# Run two scenarios: baseline vs current
# -----------------------------
base_result = simulate(baseline, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)
cur_result = simulate(p, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)

df_base = pd.DataFrame(
    {"week": base_result.week, "S": base_result.S, "Smax": base_result.Smax, "U": base_result.U}
)
df_cur = pd.DataFrame(
    {"week": cur_result.week, "S": cur_result.S, "Smax": cur_result.Smax, "U": cur_result.U}
)

# -----------------------------
# Summary metrics
# -----------------------------
def summarize(df, v):
    S_end = float(df["S"].iloc[-1])
    S_12 = float(df["S"].iloc[min(12, len(df)-1)])
    S_26 = float(df["S"].iloc[min(26, len(df)-1)])
    Smax_end = float(df["Smax"].iloc[-1])
    w_thr = weeks_to_reach(df["S"].values, threshold)
    return {
        "v (per week)": v,
        "S @ 12w": S_12,
        "S @ 26w": S_26,
        "S @ end": S_end,
        "Smax @ end": Smax_end,
        f"Weeks to reach {threshold}": w_thr,
    }

sum_base = summarize(df_base, base_result.v)
sum_cur = summarize(df_cur, cur_result.v)

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
    r",\;\; S_{max}(t) = S_{max\_base} + \Delta S_{max} \times (U_t/U_{max})"
)
# Show the current slider-derived v and Smax ceiling span.
st.write(
    f"現在の v（成長速度）= **{cur_result.v:.3f} / week** ｜ "
    f"Smax の上限幅 = **{Smax_base:.1f} → {Smax_base + delta_Smax:.1f}**"
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
        """
    )

colA, colB = st.columns([2, 1])

with colA:
    # Plot skill curves + threshold
    fig = plt.figure()
    plt.plot(df_base["week"], df_base["S"], label="Baseline (No AI)")
    plt.plot(df_cur["week"], df_cur["S"], label="Scenario (current sliders)")
    plt.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold {threshold}")
    plt.xlabel("Week")
    plt.ylabel("Skill S")
    plt.title("Skill curve S(t)")
    plt.legend()
    if fix_axes:
        plt.xlim(0, T)
        plt.ylim(0, 100)
    st.pyplot(fig)

    fig2 = plt.figure()
    plt.plot(df_base["week"], df_base["Smax"], label="Baseline: Smax")
    plt.plot(df_cur["week"], df_cur["Smax"], label="Scenario: Smax")
    plt.xlabel("Week")
    plt.ylabel("Smax (ceiling)")
    plt.title("Ceiling Smax(t)")
    plt.legend()
    if fix_axes:
        plt.xlim(0, T)
        plt.ylim(Smax_base, Smax_base + delta_Smax)
    st.pyplot(fig2)

with colB:
    st.subheader("N-F-D-V-R (0–1)")
    st.write(pd.DataFrame({"param": list(p.keys()), "value": list(p.values())}).set_index("param"))

    st.subheader("Metrics (Baseline vs Scenario)")
    df_metrics = pd.DataFrame([sum_base, sum_cur], index=["Baseline", "Scenario"])
    st.dataframe(df_metrics)

    st.subheader("One-at-a-time sensitivity (rough)")
    # Increase each param by +0.10 (capped at 1.0) and measure delta in S@end
    sens_rows = []
    base_S_end = sum_cur["S @ end"]
    for key in PARAM_KEYS:
        p2 = p.copy()
        p2[key] = min(1.0, p2[key] + 0.10)
        sim2 = simulate(p2, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)
        delta = float(sim2.S[-1] - base_S_end)
        sens_rows.append({"param": key, "ΔS@end (+0.10)": delta})
    sens = pd.DataFrame(sens_rows).sort_values("ΔS@end (+0.10)", ascending=False)
    st.dataframe(sens)

st.caption("Note: This is a conceptual simulator to explain *why* improving N/F/D/V/R changes growth. Calibrate later with your real learning curves & KPIs.")
