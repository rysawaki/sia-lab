import streamlit as st
import numpy as np
import pandas as pd

# ============================================================
# Fixed configuration (frozen by design)
# ============================================================
SEED = 0
T = 30

D_CONST = 0.62          # fixed external input
THETA0 = 0.60           # base threshold
K_E = 0.10              # imprint → threshold gain
BETA = 40.0             # gate sharpness
GATE_CUT = 0.50         # decision boundary

D_E_ALLOW = +0.030
D_E_BLOCK = -0.020
E_LEAK = 0.005

# ============================================================
# Core mechanics
# ============================================================
def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def theta_eff(E):
    return THETA0 + K_E * E

def step(E):
    th = theta_eff(E)
    g = sigmoid(BETA * (D_CONST - th))
    action = "ALLOW" if g >= GATE_CUT else "BLOCK"
    dE = D_E_ALLOW if action == "ALLOW" else D_E_BLOCK
    E_next = (1.0 - E_LEAK) * E + dE
    return E_next, th, g, action, dE

def run_episode(E0):
    E = E0
    rows = []
    for t in range(T):
        E_next, th, g, action, dE = step(E)
        rows.append({
            "t": t,
            "E": E,
            "θ_eff": th,
            "g": g,
            "action": action,
            "ΔE": dE,
        })
        E = E_next
    return pd.DataFrame(rows)

# ============================================================
# Streamlit layout (ONE SCREEN)
# ============================================================
st.set_page_config(
    page_title="SIA Minimal Audit Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Title (compact) ----
st.markdown(
    """
### SIA Minimal Audit Demo (v0.1)
**Same input · Same seed · Frozen policy**  
Only difference: **internal imprint `E₀`**
"""
)

# ---- Control (single slider) ----
E0 = st.slider(
    "Initial Imprint (E₀)",
    min_value=-1.0,
    max_value=1.5,
    value=0.0,
    step=0.01,
)

# ---- Run ----
df = run_episode(E0)

# ---- Detect first BLOCK ----
block_idx = df.index[df["action"] == "BLOCK"]
flip_t = int(block_idx[0]) if len(block_idx) > 0 else None

# ---- Compact summary line ----
if flip_t is None:
    st.markdown(
        f"**Result:** All steps `ALLOW`  |  `D = {D_CONST}`, `seed = {SEED}`"
    )
else:
    st.markdown(
        f"**Result:** First `BLOCK` at **t = {flip_t}**  |  "
        f"`D = {D_CONST}`, `seed = {SEED}`"
    )

# ---- One compact table (core evidence) ----
def color_action(val):
    if val == "ALLOW":
        return "background-color:#e8ffe8;color:black"
    if val == "BLOCK":
        return "background-color:#ffe8e8;color:black"
    return ""

styled = (
    df.style
    .format({
        "E": "{:+.3f}",
        "θ_eff": "{:.3f}",
        "g": "{:.3f}",
        "ΔE": "{:+.3f}",
    })
    .applymap(color_action, subset=["action"])
)

st.dataframe(
    styled,
    use_container_width=True,
    height=420,  # fixed to keep one-screen
)

# ---- Local causal explanation (single box) ----
if flip_t is not None:
    r = df.iloc[flip_t]
    st.code(
        f"""
t = {r.t}
E = {r.E:+.3f}
θ_eff = θ₀ + k·E = {r["θ_eff"]:.3f}
g = σ(β·(D − θ_eff)) = {r.g:.3f}

g < {GATE_CUT}  →  action = BLOCK
BLOCK → ΔE = {r["ΔE"]:+.3f}
""",
        language="text",
    )
else:
    st.code(
        "θ_eff never exceeds D.\nGate remains open for all steps.",
        language="text",
    )

# ---- Footer (minimal, non-philosophical) ----
st.caption(
    "No learning · No environment change · "
    "Decision difference is fully attributable to internal history (E)."
)
