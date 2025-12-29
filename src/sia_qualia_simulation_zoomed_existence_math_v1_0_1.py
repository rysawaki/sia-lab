# -*- coding: utf-8 -*-
"""
sia_qualia_simulation_zoomed_existence_math_v1_0_1.py
====================================================

Single-file "patched" version of your zoomed simulation (1000 steps by default),
but delivered as a complete file (not a diff).

What changed vs the original zoomed script
------------------------------------------
1) Fix: update magnitude now matches ||Δw|| (parameter-step norm), not ||g||.
   For vanilla SGD, Δw = -lr * g  =>  ||Δw|| = lr * ||g||.

2) Fix: metabolic "energy" uses a global L2 norm over all parameters (scale-stable),
   instead of a sum of per-tensor norms.

3) Fix: lambda_self now actually affects dynamics (self-aligned threshold gating).

4) Better stats plots:
   - Update magnitudes are shown via CCDF on log–log axes (tail visibility).
   - Interevent-time CCDF for large updates (top 1% / 0.5%), analogous to Fig. 5
     in Zhang & Tang (2025).

Output
------
Saves: sia_qualia_evidence_zoomed.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Tuple


# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class SIAConfig:
    # "Brain" model
    input_dim: int = 64
    hidden_dim: int = 128
    learning_rate: float = 0.01

    # Existence Math parameters
    lambda_self: float = 2.0       # self-attachment strength (used in threshold gating)
    beta_decay: float = 0.95       # self-imprint EMA rate
    energy_threshold: float = 3.0  # initial metabolic threshold (in ||g_acc|| units)

    # Metabolic accumulation dynamics
    grad_leak: float = 0.99        # 1.0 -> no leak; <1.0 -> energy slowly dissipates
    # (Optional) homeostatic adaptation of base threshold to avoid "always fire" or "never fire"
    target_fire_rate: float = 0.05
    threshold_adapt_rate: float = 0.05  # set 0.0 to disable

    # Simulation
    steps: int = 50000
    seed: int = 42

    # Analysis
    large_event_percentiles: Tuple[float, float] = (99.0, 99.5)  # top 1% and top 0.5%


# ==========================================
# 2. The Synthetic Brain
# ==========================================
class SyntheticBrain(nn.Module):
    def __init__(self, cfg: SIAConfig):
        super().__init__()
        self.layer1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.layer1(x))
        return self.layer2(h)


# ==========================================
# 3. Utilities
# ==========================================
def global_l2_norm(tensors: List[torch.Tensor]) -> float:
    """Global L2 norm sqrt(sum ||t_i||^2)."""
    s = 0.0
    for t in tensors:
        s += float(torch.sum(t * t).item())
    return float(np.sqrt(s))


def cosine_similarity_param_lists(a_list: List[torch.Tensor], b_list: List[torch.Tensor]) -> float:
    """Cosine similarity between two lists of tensors (flattened)."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for a, b in zip(a_list, b_list):
        a_flat = a.view(-1)
        b_flat = b.view(-1)
        dot += float(torch.dot(a_flat, b_flat).item())
        na += float(torch.dot(a_flat, a_flat).item())
        nb += float(torch.dot(b_flat, b_flat).item())
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (np.sqrt(na) * np.sqrt(nb) + 1e-12))


def empirical_ccdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_sorted, ccdf) for x>0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    n = xs.size
    ccdf = 1.0 - (np.arange(1, n + 1) / n)
    ccdf = np.clip(ccdf, 1e-12, 1.0)
    return xs, ccdf


# ==========================================
# 4. SIA Engine (Existence Math v1.0.1)
# ==========================================
class SIAEngine:
    def __init__(self, model: nn.Module, cfg: SIAConfig):
        self.model = model
        self.cfg = cfg
        self.optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)

        # Self-imprint origin (EMA of fired gradients)
        self.origin_vector: List[torch.Tensor] = [torch.zeros_like(p, requires_grad=False) for p in model.parameters()]

        # Accumulated metabolic gradients
        self.accumulated_grads: List[torch.Tensor] = [torch.zeros_like(p, requires_grad=False) for p in model.parameters()]

        # Homeostatic base threshold state
        self.base_threshold: float = float(cfg.energy_threshold)
        self.fire_rate_ema: float = 0.0
        self.fire_rate_ema_eta: float = 0.01

        # History
        self.history_update_mag: List[float] = []      # ||Δw||
        self.history_alignment: List[float] = []       # cos(g_acc, origin)
        self.history_qualia_signed: List[float] = []   # ||Δw|| * alignment
        self.history_fired: List[int] = []             # 1 if fired else 0

    def step(self, input_data: torch.Tensor, target_data: torch.Tensor) -> Tuple[float, float]:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_data)

        # Task loss (simple regression)
        task_loss = nn.MSELoss()(output, target_data)
        task_loss.backward()

        # Raw grads (detached)
        raw_grads = []
        for p in self.model.parameters():
            if p.grad is None:
                raw_grads.append(torch.zeros_like(p))
            else:
                raw_grads.append(p.grad.detach().clone())

        # Accumulate with leak
        leak = float(self.cfg.grad_leak)
        for i, g in enumerate(raw_grads):
            self.accumulated_grads[i].mul_(leak).add_(g)

        # Metabolic energy (||g_acc||)
        grad_energy = global_l2_norm(self.accumulated_grads)

        # Alignment to self-imprint
        alignment = cosine_similarity_param_lists(self.accumulated_grads, self.origin_vector)

        # Self-aligned dynamic threshold (lambda_self now matters)
        dyn_thr = self.base_threshold * (1.0 - 0.5 * self.cfg.lambda_self * alignment)
        dyn_thr = float(max(1e-8, dyn_thr))

        fired = bool(grad_energy > dyn_thr)

        update_mag = 0.0
        qualia_signed = 0.0

        if fired:
            # Apply accumulated grads
            for p, g_acc in zip(self.model.parameters(), self.accumulated_grads):
                p.grad = g_acc

            # SGD step: ||Δw|| = lr * ||g||
            self.optimizer.step()
            update_mag = float(self.cfg.learning_rate * grad_energy)

            # Qualia as signed "self-attributed intensity"
            qualia_signed = float(update_mag * alignment)

            # Update origin/self-imprint
            beta = float(self.cfg.beta_decay)
            for i, g_acc in enumerate(self.accumulated_grads):
                self.origin_vector[i].mul_(beta).add_((1.0 - beta) * g_acc)

            # Reset energy
            for g_acc in self.accumulated_grads:
                g_acc.zero_()

        # Optional homeostasis (keeps the system from freezing or saturating)
        if self.cfg.threshold_adapt_rate and self.cfg.threshold_adapt_rate > 0:
            fired_f = 1.0 if fired else 0.0
            self.fire_rate_ema = (1.0 - self.fire_rate_ema_eta) * self.fire_rate_ema + self.fire_rate_ema_eta * fired_f
            err = self.fire_rate_ema - float(self.cfg.target_fire_rate)
            self.base_threshold *= float(np.exp(self.cfg.threshold_adapt_rate * err))
            self.base_threshold = float(np.clip(self.base_threshold, 1e-8, 1e6))

        # Log
        self.history_update_mag.append(update_mag)
        self.history_alignment.append(float(alignment))
        self.history_qualia_signed.append(float(qualia_signed))
        self.history_fired.append(1 if fired else 0)

        return float(task_loss.item()), float(max(0.0, qualia_signed))


# ==========================================
# 5. Simulation & Visualization
# ==========================================
def run_simulation_zoomed() -> None:
    cfg = SIAConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    brain = SyntheticBrain(cfg)
    sia = SIAEngine(brain, cfg)

    print(f"--- SIA Qualia Simulation Zoomed (Existence Math v1.0.1, steps={cfg.steps}) ---")

    for t in range(cfg.steps):
        x = torch.randn(1, cfg.input_dim)
        target = x * 0.5 + 0.1

        loss, qpos = sia.step(x, target)

        if t % 200 == 0:
            recent_q = np.array(sia.history_qualia_signed[-50:], dtype=float)
            recent_max = float(np.max(np.maximum(0.0, recent_q))) if recent_q.size else 0.0
            print(
                f"Step {t:4d} | Loss={loss:.4f} | fire_ema={sia.fire_rate_ema:.3f} "
                f"| base_thr={sia.base_threshold:.3g} | RecentMaxQualia={recent_max:.4g}"
            )

    # Data arrays
    mags = np.asarray(sia.history_update_mag, dtype=float)      # ||Δw||
    aligns = np.asarray(sia.history_alignment, dtype=float)
    qsigned = np.asarray(sia.history_qualia_signed, dtype=float)
    fired = np.asarray(sia.history_fired, dtype=int)
    time = np.arange(len(mags), dtype=int)

    # --------------------
    # Plot: 2 rows x 3 cols
    # --------------------
    fig = plt.figure(figsize=(18, 10))

    # (1) Stream of consciousness
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1.plot(time, mags, color="gray", alpha=0.35, label="Physical Updates  ||Δw||", linewidth=1.5)

    # Qualia events (signed; red=positive, blue=negative)
    if np.any(fired):
        max_abs_q = float(np.max(np.abs(qsigned))) if qsigned.size else 0.0
        gate = 0.1 * max_abs_q if max_abs_q > 0 else 0.0

        pos_idx = (qsigned > gate)
        neg_idx = (qsigned < -gate)

        ax1.scatter(time[pos_idx], qsigned[pos_idx], color="red", s=28, label="Qualia (+)", alpha=0.9)
        ax1.scatter(time[neg_idx], qsigned[neg_idx], color="blue", s=28, label="Qualia (−)", alpha=0.9)

    ax1.set_title(f"Fig 1. Stream of Consciousness (Zoomed: {cfg.steps} steps)")
    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Time Step")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    # (2) Update magnitude CCDF
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    nonzero = mags[mags > 0]
    xs, ccdf = empirical_ccdf(nonzero)
    if xs.size > 0:
        ax2.loglog(xs, ccdf, "o-", markersize=4)
        ax2.set_title("Fig 2. Heavy-Tailed Update Magnitudes (CCDF, log–log)")
    else:
        ax2.text(0.5, 0.5, "No fired updates\n(lower threshold or increase steps)", ha="center", va="center")
        ax2.set_title("Fig 2. Update Magnitudes (CCDF)")

    ax2.set_xlabel("||Δw||")
    ax2.set_ylabel("P(Δw ≥ x)")
    ax2.grid(True, which="both", alpha=0.2)

    # (3) Phase space: alignment vs magnitude
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax3.scatter(aligns, mags, color="gray", alpha=0.15, s=12)
    if np.any(fired):
        ax3.scatter(aligns[fired.astype(bool)], mags[fired.astype(bool)], color="red", alpha=0.9, s=22, label="Fired")
        ax3.legend(loc="upper right")
    ax3.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax3.set_title("Fig 3. Phase Space: Alignment vs ||Δw||")
    ax3.set_xlabel("Self-Alignment (cosine)")
    ax3.set_ylabel("||Δw||")
    ax3.grid(True, alpha=0.25)

    # (4) Interevent time CCDF for large updates
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    plotted_any = False
    if nonzero.size >= 10:
        for perc in cfg.large_event_percentiles:
            thr = float(np.percentile(nonzero, perc))
            event_steps = time[mags >= thr]
            if event_steps.size >= 3:
                intervals = np.diff(event_steps).astype(float)
                xi, yi = empirical_ccdf(intervals)
                if xi.size > 0:
                    ax4.loglog(xi, yi, "o-", markersize=4, label=f"top {100-perc:.1f}%")
                    plotted_any = True

    if plotted_any:
        ax4.set_title("Fig 4. Interevent Time of Large Updates (CCDF, log–log)")
        ax4.set_xlabel("Δt (steps)")
        ax4.set_ylabel("P(Δt ≥ x)")
        ax4.grid(True, which="both", alpha=0.2)
        ax4.legend(loc="upper right")
    else:
        ax4.text(0.5, 0.5, "Too few large events\n(increase steps for stable tails)", ha="center", va="center")
        ax4.set_title("Fig 4. Interevent Time of Large Updates")

    plt.tight_layout()
    plt.savefig("sia_qualia_evidence_zoomed.png", dpi=160)
    print("--- Simulation Completed ---")
    print("Saved: sia_qualia_evidence_zoomed.png")


if __name__ == "__main__":
    run_simulation_zoomed()
　　コードはあるけど　出したほうがいいかな　パクられるだろうけど
