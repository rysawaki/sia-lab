import streamlit as st
import numpy as np
import pandas as pd

# =========================
# Config (Fixed)
# =========================
SEED = 0
T = 40

D_CONST = 0.62
THETA0 = 0.60
K_E = 0.10
BETA = 40.0
GATE_CUT = 0.50

D_E_ALLOW = +0.030
D_E_BLOCK = -0.020
E_LEAK = 0.005

# =========================
# Core functions
# =========================
def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def theta_eff(E):
    return THETA0 + K_E * E

def gate(D, th):
    return sigmoid(BETA * (D - th))

def step(E):
    th = theta_eff(E)
    g = gate(D_CONST, th)
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
            "theta_eff": th,
            "g": g,
            "action": action,
            "dE": dE,
            "E_next": E_next,
        })
        E = E_next
    return pd.DataFrame(rows)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SIA Minimal Audit Demo", layout="wide")

st.title("SIA Minimal Audit Demo (Read-only)")
st.caption(
    "Same input, same seed. The only difference is the internal state E0. "
    "Mechanism: Imprint → θ_eff → Gate → Action → Next Imprint"
)

st.markdown("---")

# ---- Controls ----
st.subheader("Internal Imprint (E0)")
E0 = st.slider(
    "Initial Imprint E0",
    min_value=-1.0,
    max_value=1.5,
    value=0.0,
    step=0.01,
)

st.markdown(
    f"""
**Fixed external input** - D = `{D_CONST}`  
- seed = `{SEED}`  
- policy / parameters = frozen
"""
)

# ---- Run ----
df = run_episode(E0)

# ---- Divergence / Flip info ----
first_block = df.index[df["action"] == "BLOCK"]
flip_t = int(first_block[0]) if len(first_block) > 0 else None

st.markdown("---")
st.subheader("Result")

if flip_t is None:
    st.success("Result: All steps ALLOWED (No divergence for this E0)")
else:
    st.warning(f"First BLOCK occurred at: **t = {flip_t}**")

# ---- Table ----
def color_action(val):
    if val == "ALLOW":
        return "background-color: #e6ffe6; color: black"
    if val == "BLOCK":
        return "background-color: #ffe6e6; color: black"
    return ""

st.subheader("Causal Trace (time series)")

styled = (
    df.style
    .format({
        "E": "{:+.4f}",
        "theta_eff": "{:.4f}",
        "g": "{:.4f}",
        "dE": "{:+.3f}",
        "E_next": "{:+.4f}",
    })
    .applymap(color_action, subset=["action"])
)

st.dataframe(styled, use_container_width=True, height=520)

# ---- Minimal causal explanation ----
st.markdown("---")
st.subheader("Local Causal Explanation")

if flip_t is not None:
    r = df.iloc[flip_t]
    st.code(
        f"""
t = {r.t}
E = {r.E:+.4f}
θ_eff = θ0 + kE·E = {r.theta_eff:.4f}
g = σ(β·(D − θ_eff)) = {r.g:.4f}

Since g < {GATE_CUT}  →  action = BLOCK
BLOCK triggers dE = {r.dE:+.3f} → E_next = {r.E_next:+.4f}
""",
        language="text",
    )
else:
    st.info("With this E0, θ_eff does not exceed D, and the Gate remains open (ALLOW).")

st.markdown(
    """
**Notes** - No learning / No environmental change.  
- The output divergence is caused solely by E (history/imprint).
"""
)
