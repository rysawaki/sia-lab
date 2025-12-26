# SIA Minimal Audit Demo (v0.1)

**DOI:** https://doi.org/10.5281/zenodo.18064418

## Overview

This repository contains a **minimal, fully deterministic demonstration** of **SIA (Self-Imprint Attribution)**.

The purpose of this demo is **not** to improve performance, learn policies, or simulate intelligence.
Its sole purpose is to demonstrate a single causal claim:

> **Even with identical external input, identical random seed, and a frozen policy,
> decisions can diverge solely due to differences in internal history (imprint).**

This repository provides a concrete, inspectable artifact showing how such divergence arises and how it can be **causally explained step by step**.

---

## What This Demo Shows

* External input `D(t)` is **fixed and identical** across runs
* Random seed is **fixed**
* No learning, no weight updates, no environment change
* The **only difference** between runs is the initial internal state `E₀` (imprint)

Despite this, the agent’s action (`ALLOW` / `BLOCK`) can diverge.

The divergence is fully explained by the following causal chain:

```
Imprint (E)
   ↓
Effective threshold (θ_eff)
   ↓
Gate value (g)
   ↓
Action (ALLOW / BLOCK)
   ↓
Next imprint update (E → E')
```

All variables are logged explicitly and can be inspected at every timestep.

---

## What This Demo Does NOT Do

To avoid misunderstanding, this demo explicitly does **not** include:

* ❌ Learning or optimization
* ❌ Reinforcement learning
* ❌ Neural network weight updates
* ❌ Policy search
* ❌ World models or environments
* ❌ Language models
* ❌ Probabilistic action sampling

This is **not** an AI system and **not** a behavioral model.

It is a **causal audit artifact**.

---

## Why This Matters

In many AI systems, post-hoc explanations rely on opaque model parameters or statistical attributions.

SIA takes a different approach:

* Decisions are attributed to **persistent internal state**
* Internal state evolves deterministically via explicit rules
* Every decision can be traced back to concrete prior experience

This makes it possible to answer questions such as:

* *Why did the system block this action now, but not earlier?*
* *Which past experiences shifted the decision boundary?*
* *Is the divergence due to environment, randomness, or internal history?*

In this demo, the answer is unambiguous: **internal history only**.

---

## How to Run

```bash
pip install streamlit numpy pandas
streamlit run demo.py
```

You can adjust a single parameter (`E₀`) and observe how the decision trajectory changes, while all other factors remain fixed.

---

## Status

* **Version:** v0.1
* **Scope:** Minimal audit demonstration
* **Stability:** Deterministic, reproducible
* **Intended use:** Inspection, discussion, and citation

Future versions may extend this framework, but **this version is intentionally frozen**.

---

## License

Apache License 2.0

---

