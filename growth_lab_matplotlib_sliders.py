\
# growth_lab_matplotlib_sliders.py
# Zero-dependency interactive sliders (Matplotlib widgets)
# Run: python growth_lab_matplotlib_sliders.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -----------------------------
# Core model
# -----------------------------
def v_from_params(p: dict, v0: float = 0.25) -> float:
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
        if enable_frontier and t > 0:
            u_rate = u0 * (p["V"] * p["D"])
            U[t] = U[t-1] + u_rate * (1 - U[t-1] / Umax)
        else:
            U[t] = 0.0

        Smax[t] = Smax_base + (delta_Smax * (U[t] / Umax) if enable_frontier else 0.0)

        if t < T:
            S[t+1] = S[t] + v * (Smax[t] - S[t])

    return weeks, S, Smax, U, v

def weeks_to_reach(S, thr):
    idx = np.argmax(S >= thr)
    if S[idx] < thr:
        return None
    return int(idx)

# -----------------------------
# Defaults
# -----------------------------
baseline = {"N": 0.35, "F": 0.35, "D": 0.40, "V": 0.30, "R": 0.35}
dojo =     {"N": 0.80, "F": 0.70, "D": 0.60, "V": 0.75, "R": 0.70}

p = dojo.copy()   # start from DoJo
T = 52
S0 = 50.0
v0 = 0.25
thr = 75.0

enable_frontier = True
Smax_base = 80.0
delta_Smax = 12.0
Umax = 10.0
u0 = 2.0

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(10, 6))
ax_skill = plt.axes([0.08, 0.36, 0.88, 0.58])
ax_smax  = plt.axes([0.08, 0.30, 0.88, 0.05])

# Sliders area
slider_axes = {}
slider_y0 = 0.20
dy = 0.035
for i, key in enumerate(["N", "F", "D", "V", "R"]):
    slider_axes[key] = plt.axes([0.12, slider_y0 - i*dy, 0.76, 0.02])

ax_btn_reset = plt.axes([0.12, 0.02, 0.18, 0.05])
ax_btn_base  = plt.axes([0.34, 0.02, 0.18, 0.05])
ax_btn_dojo  = plt.axes([0.56, 0.02, 0.18, 0.05])

# Initial sim
w_b, S_b, Smax_b, U_b, v_b = simulate(baseline, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)
w,   S,   Smax,   U,   v   = simulate(p,       T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)

# Plot lines
line_base, = ax_skill.plot(w_b, S_b, label="Baseline (No AI)")
line_cur,  = ax_skill.plot(w,   S,   label="Scenario (Sliders)")
thr_line = ax_skill.axhline(thr, linestyle="--", linewidth=1, label=f"Threshold {thr}")

ax_skill.set_xlabel("Week")
ax_skill.set_ylabel("Skill S")
ax_skill.set_title("DoJo Growth Lab — move sliders and see S(t)")
ax_skill.legend(loc="lower right")
ax_skill.set_ylim(0, 100)

line_smax_base, = ax_smax.plot(w_b, Smax_b, label="Baseline Smax")
line_smax_cur,  = ax_smax.plot(w,   Smax,   label="Scenario Smax")
ax_smax.set_ylabel("Smax")
ax_smax.set_yticks([])

# Text box for metrics
text = ax_skill.text(0.02, 0.98, "", transform=ax_skill.transAxes, va="top")

# Sliders
sliders = {}
for key in ["N", "F", "D", "V", "R"]:
    sliders[key] = Slider(slider_axes[key], key, 0.0, 1.0, valinit=p[key], valstep=0.01)

def update_metrics():
    w_thr = weeks_to_reach(S, thr)
    msg = (
        f"v={v:.3f}/week | S@12w={S[min(12, len(S)-1)]:.1f} | S@26w={S[min(26, len(S)-1)]:.1f} | "
        f"S@end={S[-1]:.1f} | Smax@end={Smax[-1]:.1f} | reach {thr}: {w_thr}w"
    )
    text.set_text(msg)

def on_change(val):
    global w, S, Smax, U, v
    for k in ["N", "F", "D", "V", "R"]:
        p[k] = sliders[k].val
    w, S, Smax, U, v = simulate(p, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)

    line_cur.set_xdata(w)
    line_cur.set_ydata(S)
    line_smax_cur.set_xdata(w)
    line_smax_cur.set_ydata(Smax)

    update_metrics()
    fig.canvas.draw_idle()

for s in sliders.values():
    s.on_changed(on_change)

# Buttons
btn_reset = Button(ax_btn_reset, "Reset Sliders")
btn_base  = Button(ax_btn_base, "Preset: Baseline")
btn_dojo  = Button(ax_btn_dojo, "Preset: DoJo")

def do_reset(event):
    for k in ["N", "F", "D", "V", "R"]:
        sliders[k].reset()

def set_preset(preset):
    for k in ["N", "F", "D", "V", "R"]:
        sliders[k].set_val(preset[k])

btn_reset.on_clicked(do_reset)
btn_base.on_clicked(lambda e: set_preset(baseline))
btn_dojo.on_clicked(lambda e: set_preset(dojo))

update_metrics()
plt.show()
