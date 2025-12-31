# SIA Minimal Audit Demo (v0.1)

## Purpose

This repository provides a **minimal, fully deterministic audit artifact** for **SIA (Self-Imprint Attribution)**.

It demonstrates a single, falsifiable claim:

> **With identical external input, identical random seed, and a frozen policy,
> action divergence can arise solely from differences in internal history (imprint).**

No learning, no randomness, no environment change.

Only **history**.

---

## What Is Demonstrated

Across multiple runs:

* External input `D(t)` is **identical**
* Random seed is **identical**
* Policy logic is **frozen**
* No weight updates, no adaptation

The **only** differing variable is the initial internal state `E₀` (imprint).

Despite this, the agent’s discrete decision (`ALLOW` / `BLOCK`) can diverge.

The divergence is **fully and deterministically explained** by the following causal chain:

```
Imprint (E)
   ↓
Effective threshold (θ_eff)
   ↓
Gate value (g)
   ↓
Action (ALLOW / BLOCK)
   ↓
Next imprint update (E → E′)
```

All intermediate variables are logged and inspectable at every timestep.

---

## What This Is *Not*

This demo intentionally excludes:

* ❌ Learning or optimization
* ❌ Reinforcement learning
* ❌ Neural network weight updates
* ❌ Policy search or sampling
* ❌ Environment dynamics
* ❌ World models or language models

This is **not an AI system** and **not a behavioral simulator**.

It is a **causal audit primitive**.

---

## Why This Matters

Most AI explanations rely on opaque parameters or post-hoc attribution.

SIA instead enforces:

* **Persistent internal state** as the decision driver
* **Deterministic state evolution**
* **Step-wise causal traceability**

This allows precise answers to questions such as:

* *Why did the system block now but not earlier?*
* *Which past state shifted the decision boundary?*
* *Is divergence caused by noise, environment, or history?*

In this demo, the answer is unambiguous:

> **Internal history only.**

---

## How to Run

```bash
pip install streamlit numpy pandas
streamlit run demo.py
```

Adjust only the initial imprint parameter (`E₀`) and observe how the decision trajectory diverges while all other factors remain fixed.

---

## Status

* **Version:** v0.1
* **Scope:** Minimal audit demonstration
* **Determinism:** Fully reproducible
* **Design:** Intentionally frozen

Future work may extend this framework, but **this artifact is not meant to evolve**.

---

## License

Apache License 2.0

---

**DOI:** [https://doi.org/10.5281/zenodo.18064418](https://doi.org/10.5281/zenodo.18064418)

---
## Paper

This repository includes a copy of the paper for convenience.
The canonical version is archived at Zenodo:

[https://doi.org/10.5281/zenodo.18104237](https://zenodo.org/records/18104237)
