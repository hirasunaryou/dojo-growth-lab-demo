# DoJo Growth Lab (Interactive demo)

This mini demo lets you *move N-F-D-V-R sliders* and instantly see how they change:
- Learning velocity v (derived from N×F×D×V×R)
- Skill curve S(t) using dS/dt = v × (Smax − S)
- Ceiling (Smax) uplift via U (Frontier Unlock), driven by V×D

## Option A (recommended): Streamlit (web UI)
1) Install deps:
   pip install -r requirements_streamlit.txt
2) Run:
   streamlit run growth_lab_streamlit.py

## Option B (no extra deps): Matplotlib sliders (desktop window)
1) Install deps:
   pip install -r requirements_basic.txt
2) Run:
   python growth_lab_matplotlib_sliders.py

## Notes
- This is a *conceptual* simulator for stakeholder alignment.
- Calibrate parameters with real data later (e.g., onboarding learning curves, QA scores, KPI deltas).
