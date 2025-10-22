# -----------------------------------------------------------------------------
# Copyright (c) 2025 Learning By Machine
# Licensed under the MIT License. See LICENSE file in the project root.
#
# utils.py — Utilities: correlation checks, summaries, and small helpers
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch


def describe_batch(X_list: List[torch.Tensor], y_list: List[torch.Tensor]) -> None:
    """
    Print a concise summary of a variable-length batch for quick inspection.
    """
    B = len(X_list)
    V0 = X_list[0].shape[1]
    Ts = [x.shape[0] for x in X_list]
    print("=== [utils] Batch Description ===")
    print(f"[utils] B (subjects): {B}")
    print(f"[utils] V (features): {V0}")
    print(f"[utils] T lengths   : {Ts}")
    # Sanity check on label coverage
    unique_labels = sorted(set(torch.cat(y_list).tolist()))
    print(f"[utils] Unique labels across batch: {unique_labels}\n")


def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Pearson correlation between two 1D arrays with numerical safety.
    """
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))

def pretty_matrix(name: str, M: np.ndarray) -> None:
    """
    Nicely print a small numeric matrix.
    """
    print(f"--- {name} ---")
    for row in M:
        print("  " + " ".join(f"{v:+.3f}" for v in row))
    print()

def compute_pairwise_corr(
    X_list: List[torch.Tensor], Z_list: List[torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Pearson correlations between subjects for:
      (1) Raw input sequences (flattened over time×feature),
      (2) Learned embeddings (flattened over time×dim).

    IMPORTANT: Align lengths *per pair* (i, j) to avoid shape mismatches.
    """
    N = len(X_list)
    print("=== [utils] Pairwise Correlations (pairwise-aligned) ===")

    # Cache CPU numpy arrays and lengths (no flatten yet)
    X_np = []
    Z_np = []
    X_T = []
    Z_T = []
    for b in range(N):
        Xi = X_list[b].detach().cpu().numpy()       # [T_bx, V]
        Zi = Z_list[b].detach().cpu().numpy()       # [T_bz, d]
        X_np.append(Xi)
        Z_np.append(Zi)
        X_T.append(Xi.shape[0])
        Z_T.append(Zi.shape[0])
        print(f"[utils] Subject {b}: X shape={Xi.shape}, Z shape={Zi.shape}")

    corr_raw = np.eye(N, dtype=np.float64)
    corr_emb = np.eye(N, dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            # ----- RAW alignment -----
            Tij_raw = min(X_T[i], X_T[j])                 # common time across raw
            xi = X_np[i][:Tij_raw].reshape(-1)            # [Tij_raw * V]
            xj = X_np[j][:Tij_raw].reshape(-1)

            # ----- EMB alignment -----
            Tij_emb = min(Z_T[i], Z_T[j])                 # common time across embeddings
            zi = Z_np[i][:Tij_emb].reshape(-1)            # [Tij_emb * d]
            zj = Z_np[j][:Tij_emb].reshape(-1)

            print(
                f"[utils] Pair (i={i}, j={j}) | "
                f"Tij_raw={Tij_raw}, vecs: {xi.shape} vs {xj.shape} | "
                f"Tij_emb={Tij_emb}, vecs: {zi.shape} vs {zj.shape}"
            )

            # Safety: if either aligned length is zero, skip (shouldn't happen in normal data)
            if xi.size == 0 or xj.size == 0:
                r_raw = 0.0
                print(f"[utils]   WARN raw empty after align → r_raw=0.0")
            else:
                r_raw = _pearsonr(xi, xj)

            if zi.size == 0 or zj.size == 0:
                r_emb = 0.0
                print(f"[utils]   WARN emb empty after align → r_emb=0.0")
            else:
                r_emb = _pearsonr(zi, zj)

            corr_raw[i, j] = corr_raw[j, i] = r_raw
            corr_emb[i, j] = corr_emb[j, i] = r_emb
            print(f"[utils]   r_raw={r_raw:+.4f} | r_emb={r_emb:+.4f}")

    print("=== [utils] Done ===\n")
    return corr_raw, corr_emb
