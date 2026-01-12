\
# growth_lab_streamlit.py
# Interactive N-F-D-V-R simulator (Streamlit)
# Run: streamlit run growth_lab_streamlit.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="DoJo Growth Lab", layout="wide")

# -----------------------------
# Core model
# -----------------------------
def v_from_params(p: dict, v0: float = 0.25) -> float:
    # Geometric mean keeps scale stable and preserves multiplicative intuition
    prod = p["N"] * p["F"] * p["D"] * p["V"] * p["R"]
    return v0 * (prod ** (1/5))

def simulate(p: dict, T: int, S0: float, v0: float, Smax_base: float, delta_Smax: float,
             Umax: float, u0: float, enable_frontier: bool):
    weeks = np.arange(T + 1)
    v = v_from_params(p, v0=v0)

    S = np.zeros(T + 1)
    U = np.zeros(T + 1)
    Smax = np.zeros(T + 1)
    S[0] = S0

    for t in range(T + 1):
        # Frontier unlock
        if enable_frontier and t > 0:
            u_rate = u0 * (p["V"] * p["D"])
            U[t] = U[t-1] + u_rate * (1 - U[t-1] / Umax)  # saturating growth
        else:
            U[t] = 0.0

        Smax[t] = Smax_base + (delta_Smax * (U[t] / Umax) if enable_frontier else 0.0)

        # Skill update (discrete-time form)
        if t < T:
            S[t+1] = S[t] + v * (Smax[t] - S[t])

    df = pd.DataFrame({"week": weeks, "S": S, "Smax": Smax, "U": U})
    return df, v

def weeks_to_reach(S_series: np.ndarray, threshold: float):
    idx = np.argmax(S_series >= threshold)
    if S_series[idx] < threshold:
        return None
    return int(idx)

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

# -----------------------------
# Run two scenarios: baseline vs current
# -----------------------------
df_base, v_base = simulate(baseline, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)
df_cur, v_cur   = simulate(p,       T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)

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

sum_base = summarize(df_base, v_base)
sum_cur  = summarize(df_cur,  v_cur)

# -----------------------------
# Layout
# -----------------------------
st.title("DoJo Growth Lab — Move N-F-D-V-R and *see* growth")

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
    st.pyplot(fig)

    fig2 = plt.figure()
    plt.plot(df_base["week"], df_base["Smax"], label="Baseline: Smax")
    plt.plot(df_cur["week"], df_cur["Smax"], label="Scenario: Smax")
    plt.xlabel("Week")
    plt.ylabel("Smax (ceiling)")
    plt.title("Ceiling Smax(t)")
    plt.legend()
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
    for key in ["N", "F", "D", "V", "R"]:
        p2 = p.copy()
        p2[key] = min(1.0, p2[key] + 0.10)
        df2, v2 = simulate(p2, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)
        delta = float(df2["S"].iloc[-1] - base_S_end)
        sens_rows.append({"param": key, "ΔS@end (+0.10)": delta})
    sens = pd.DataFrame(sens_rows).sort_values("ΔS@end (+0.10)", ascending=False)
    st.dataframe(sens)

st.caption("Note: This is a conceptual simulator to explain *why* improving N/F/D/V/R changes growth. Calibrate later with your real learning curves & KPIs.")
