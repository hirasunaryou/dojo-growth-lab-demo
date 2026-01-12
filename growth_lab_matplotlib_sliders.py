\
# growth_lab_matplotlib_sliders.py
# Zero-dependency interactive sliders (Matplotlib widgets)
# Run: python growth_lab_matplotlib_sliders.py

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

from core.growth_model import PARAM_KEYS, simulate, weeks_to_reach

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
ax_btn_fix   = plt.axes([0.78, 0.02, 0.18, 0.05])

# Initial sim
base_result = simulate(baseline, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)
cur_result = simulate(p, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)
fix_axes = True

# Plot lines
line_base, = ax_skill.plot(base_result.week, base_result.S, label="Baseline (No AI)")
line_cur,  = ax_skill.plot(cur_result.week, cur_result.S, label="Scenario (Sliders)")
thr_line = ax_skill.axhline(thr, linestyle="--", linewidth=1, label=f"Threshold {thr}")

ax_skill.set_xlabel("Week")
ax_skill.set_ylabel("Skill S")
ax_skill.set_title("DoJo Growth Lab — move sliders and see S(t)")
ax_skill.legend(loc="lower right")
ax_skill.set_ylim(0, 100)
ax_skill.set_xlim(0, T)

line_smax_base, = ax_smax.plot(base_result.week, base_result.Smax, label="Baseline Smax")
line_smax_cur,  = ax_smax.plot(cur_result.week, cur_result.Smax, label="Scenario Smax")
ax_smax.set_ylabel("Smax")
ax_smax.set_yticks([])
ax_smax.set_xlim(0, T)
ax_smax.set_ylim(Smax_base, Smax_base + delta_Smax)

# Fixed formula text (top of figure)
formula_text = (
    "dS/dt = v × (Smax − S)\n"
    "v = v0 × (N×F×D×V×R)^(1/5)\n"
    "u_rate = u0 × (V×D),  U_t = U_{t-1} + u_rate × (1 − U_{t-1}/Umax),  "
    "Smax(t) = Smax_base + ΔSmax × (U_t/Umax)"
)
fig.text(0.08, 0.97, formula_text, ha="left", va="top", fontsize=9)

# Text box for dynamic metrics
text = fig.text(0.08, 0.88, "", ha="left", va="top", fontsize=9)

# Sliders
sliders = {}
for key in PARAM_KEYS:
    sliders[key] = Slider(slider_axes[key], key, 0.0, 1.0, valinit=p[key], valstep=0.01)

def update_metrics():
    w_thr = weeks_to_reach(cur_result.S, thr)
    w_thr_label = f"{w_thr}w" if w_thr is not None else "not reached"
    msg = (
        f"v={cur_result.v:.3f}/week | S@end={cur_result.S[-1]:.1f} | "
        f"Smax@end={cur_result.Smax[-1]:.1f} | reach {thr}: {w_thr_label}"
    )
    text.set_text(msg)

def apply_axis_mode():
    # Toggle between fixed axes (comparison-friendly) and auto scaling.
    if fix_axes:
        ax_skill.set_xlim(0, T)
        ax_skill.set_ylim(0, 100)
        ax_smax.set_xlim(0, T)
        ax_smax.set_ylim(Smax_base, Smax_base + delta_Smax)
    else:
        ax_skill.relim()
        ax_skill.autoscale()
        ax_smax.relim()
        ax_smax.autoscale()

def on_change(val):
    global cur_result
    for k in PARAM_KEYS:
        p[k] = sliders[k].val
    cur_result = simulate(p, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier)

    line_cur.set_xdata(cur_result.week)
    line_cur.set_ydata(cur_result.S)
    line_smax_cur.set_xdata(cur_result.week)
    line_smax_cur.set_ydata(cur_result.Smax)

    update_metrics()
    apply_axis_mode()
    fig.canvas.draw_idle()

for s in sliders.values():
    s.on_changed(on_change)

# Buttons
btn_reset = Button(ax_btn_reset, "Reset Sliders")
btn_base  = Button(ax_btn_base, "Preset: Baseline")
btn_dojo  = Button(ax_btn_dojo, "Preset: DoJo")
btn_fix = CheckButtons(ax_btn_fix, ["Fix axes"], [fix_axes])

def do_reset(event):
    for k in PARAM_KEYS:
        sliders[k].reset()

def set_preset(preset):
    for k in PARAM_KEYS:
        sliders[k].set_val(preset[k])

def toggle_fix(label):
    global fix_axes
    fix_axes = not fix_axes
    apply_axis_mode()
    fig.canvas.draw_idle()

btn_reset.on_clicked(do_reset)
btn_base.on_clicked(lambda e: set_preset(baseline))
btn_dojo.on_clicked(lambda e: set_preset(dojo))
btn_fix.on_clicked(toggle_fix)

update_metrics()
apply_axis_mode()
plt.show()
