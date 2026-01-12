\
# growth_lab_matplotlib_sliders.py
# Zero-dependency interactive sliders (Matplotlib widgets)
# Run: python growth_lab_matplotlib_sliders.py

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

from core.constants import PARAM_KEYS
from core.growth_model import simulate, weeks_to_reach

# -----------------------------
# Defaults
# -----------------------------
no_system = {"N": 0.25, "F": 0.15, "D": 0.20, "V": 0.10, "R": 0.15}
training = {"N": 0.40, "F": 0.40, "D": 0.35, "V": 0.30, "R": 0.35}
optimized = {"N": 0.80, "F": 0.70, "D": 0.60, "V": 0.75, "R": 0.70}

p = optimized.copy()   # start from Optimized defaults
T = 520
S0 = 20.0
v0 = 0.03
thr = 75.0

enable_frontier = True
Smax_base = 80.0
delta_Smax = 15.0
gamma = 0.5
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

# Initial sim (3 lines)
no_result = simulate(
    no_system, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
)
training_result = simulate(
    training, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
)
cur_result = simulate(
    p, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
)
fix_axes = True
# Convert week index to years so the 10-year horizon reads cleanly.
weeks_to_years = 52

# Plot lines
line_no, = ax_skill.plot(no_result.week / weeks_to_years, no_result.S, label="No system")
line_training, = ax_skill.plot(
    training_result.week / weeks_to_years, training_result.S, label="Training system"
)
line_cur, = ax_skill.plot(cur_result.week / weeks_to_years, cur_result.S, label="Optimized (Sliders)")
thr_line = ax_skill.axhline(thr, linestyle="--", linewidth=1, label=f"Threshold {thr}")

ax_skill.set_xlabel("Year")
ax_skill.set_ylabel("Skill S")
ax_skill.set_title("DoJo Growth Lab — move sliders and see S(t)")
ax_skill.legend(loc="lower right")
ax_skill.set_ylim(0, 100)
ax_skill.set_xlim(0, T / weeks_to_years)

line_smax_no, = ax_smax.plot(no_result.week / weeks_to_years, no_result.Smax, label="No system Smax")
line_smax_training, = ax_smax.plot(
    training_result.week / weeks_to_years, training_result.Smax, label="Training Smax"
)
line_smax_cur, = ax_smax.plot(
    cur_result.week / weeks_to_years, cur_result.Smax, label="Optimized Smax"
)
ax_smax.set_ylabel("Smax")
ax_smax.set_yticks([])
ax_smax.set_xlim(0, T / weeks_to_years)
ax_smax.set_ylim(0, 100)

# Fixed formula text (top of figure)
formula_text = (
    "dS/dt = v × (Smax − S)\n"
    "v = v0 × (N×F×D×V×R)^(1/5)\n"
    "u_rate = u0 × (V×D),  U_t = U_{t-1} + u_rate × (1 − U_{t-1}/Umax),  "
    "Smax(t) = Smax_base + ΔSmax × (V×D)^gamma × (U_t/Umax)"
)
fig.text(0.08, 0.97, formula_text, ha="left", va="top", fontsize=9)

# Text box for dynamic metrics
text = fig.text(0.08, 0.88, "", ha="left", va="top", fontsize=9)

# Sliders
sliders = {}
for key in PARAM_KEYS:
    sliders[key] = Slider(slider_axes[key], key, 0.0, 1.0, valinit=p[key], valstep=0.01)

def update_metrics():
    def threshold_label(result):
        w_thr = weeks_to_reach(result.S, thr)
        return f"{w_thr / weeks_to_years:.1f}y" if w_thr is not None else "not reached"

    w_thr_no = threshold_label(no_result)
    w_thr_training = threshold_label(training_result)
    w_thr_opt = threshold_label(cur_result)
    msg = (
        f"No system: v={no_result.v:.3f}/week | S@end={no_result.S[-1]:.1f} | "
        f"Smax@end={no_result.Smax[-1]:.1f} | Smax_cap={no_result.Smax_cap:.1f} | reach {thr}: {w_thr_no}\n"
        f"Training: v={training_result.v:.3f}/week | S@end={training_result.S[-1]:.1f} | "
        f"Smax@end={training_result.Smax[-1]:.1f} | Smax_cap={training_result.Smax_cap:.1f} | reach {thr}: {w_thr_training}\n"
        f"Optimized: v={cur_result.v:.3f}/week | S@end={cur_result.S[-1]:.1f} | "
        f"Smax@end={cur_result.Smax[-1]:.1f} | Smax_cap={cur_result.Smax_cap:.1f} | reach {thr}: {w_thr_opt}"
    )
    text.set_text(msg)

def apply_axis_mode():
    # Toggle between fixed axes (comparison-friendly) and auto scaling.
    if fix_axes:
        ax_skill.set_xlim(0, T / weeks_to_years)
        ax_skill.set_ylim(0, 100)
        ax_smax.set_xlim(0, T / weeks_to_years)
        ax_smax.set_ylim(0, 100)
    else:
        ax_skill.relim()
        ax_skill.autoscale()
        ax_smax.relim()
        ax_smax.autoscale()

def on_change(val):
    global cur_result, training_result, no_result
    for k in PARAM_KEYS:
        p[k] = sliders[k].val
    cur_result = simulate(
        p, T, S0, v0, Smax_base, delta_Smax, Umax, u0, enable_frontier, gamma
    )

    line_cur.set_xdata(cur_result.week / weeks_to_years)
    line_cur.set_ydata(cur_result.S)
    line_smax_cur.set_xdata(cur_result.week / weeks_to_years)
    line_smax_cur.set_ydata(cur_result.Smax)

    update_metrics()
    apply_axis_mode()
    fig.canvas.draw_idle()

for s in sliders.values():
    s.on_changed(on_change)

# Buttons
btn_reset = Button(ax_btn_reset, "Reset Sliders")
btn_base  = Button(ax_btn_base, "Preset: No system")
btn_dojo  = Button(ax_btn_dojo, "Preset: Optimized")
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
btn_base.on_clicked(lambda e: set_preset(no_system))
btn_dojo.on_clicked(lambda e: set_preset(optimized))
btn_fix.on_clicked(toggle_fix)

update_metrics()
apply_axis_mode()
plt.show()
