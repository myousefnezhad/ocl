# -----------------------------------------------------------------------------
# Copyright (c) 2025 Learning By Machine
# Licensed under the MIT License. See LICENSE file in the project root.
#
# data.py — Simple synthetic data generator for OCL experiments
# -----------------------------------------------------------------------------
from __future__ import annotations
import math
import random
from typing import List, Tuple

import torch


def set_global_seed(seed: int = 0) -> None:
    """
    Make results as reproducible as practical for a demo.
    NOTE: Full determinism is not guaranteed across hardware/backends.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_synthetic_sequences(
    B: int = 3,             # number of sequences / subjects
    V: int = 128,           # input feature dimension (e.g., voxels)
    T_min: int = 80,        # minimum length per sequence
    T_max: int = 120,       # maximum length per sequence
    C: int = 4,             # number of stimulus classes
    snr_db: float = 6.0,    # signal-to-noise ratio in dB
    seed: int = 0,
    device: torch.device | None = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Create a list of variable-length sequences X_list and matching per-time labels y_list.

    Construction:
      - Build C base temporal prototypes (sine/cosine motifs).
      - For each subject b:
          * Sample length T_b in [T_min, T_max].
          * Create a class label per time t (cyclic or random).
          * Create a subject-specific mixing W_b ∈ R^{C×V}.
          * X_b[t] = sum_k 1{y[t]=k} * motif_k[t] * W_b[k] + noise.

    Returns:
      X_list: list of length B, each tensor ∈ R^{T_b × V}
      y_list: list of length B, each tensor ∈ {0..C-1}^T_b
    """
    set_global_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== [data] Generating synthetic sequences ===")
    print(f"[data] B (subjects)   : {B}")
    print(f"[data] V (features)   : {V}")
    print(f"[data] T range        : [{T_min}, {T_max}]")
    print(f"[data] C (classes)    : {C}")
    print(f"[data] SNR (dB)       : {snr_db:.1f}")
    print(f"[data] Device         : {device}")

    # Build C temporal motifs with different frequencies/phases
    max_T = T_max
    t = torch.arange(max_T, dtype=torch.float32, device=device)
    motifs = []
    for k in range(C):
        freq = 0.01 * (k + 1)
        phase = k * math.pi / 7.0
        wave = torch.sin(2 * math.pi * freq * t + phase) + 0.5 * torch.cos(
            2 * math.pi * (freq * 0.5) * t + 0.3 * phase
        )
        # Normalize motif amplitude
        wave = wave / (wave.std() + 1e-6)
        motifs.append(wave)  # shape [max_T]
    motifs = torch.stack(motifs, dim=0)  # [C, max_T]

    # Noise scaling from SNR (in dB): snr_db = 10*log10(sig_var/noise_var)
    noise_std = math.sqrt(1.0 / (10 ** (snr_db / 10.0)))

    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    for b in range(B):
        T_b = random.randint(T_min, T_max)
        # Class schedule (cyclic for clarity)
        y_b = torch.tensor([(t_ // 10) % C for t_ in range(T_b)], device=device, dtype=torch.long)

        # Subject-specific mixing (C→V)
        W_b = torch.randn(C, V, device=device) / math.sqrt(V)

        # Compose signal
        X_b = torch.zeros(T_b, V, device=device)
        for t_ in range(T_b):
            k = int(y_b[t_].item())
            # Use motif up to T_b
            s_t = motifs[k, t_]
            X_b[t_] = s_t * W_b[k]  # broadcast across V-features

        # Add noise
        X_b = X_b + noise_std * torch.randn_like(X_b)

        X_list.append(X_b)
        y_list.append(y_b)

        print(f"[data] Subject {b}: T={T_b} | y counts={torch.bincount(y_b, minlength=C).tolist()}")

    print("=== [data] Done ===\n")
    return X_list, y_list
