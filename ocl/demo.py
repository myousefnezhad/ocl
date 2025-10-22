# -----------------------------------------------------------------------------
# Copyright (c) 2025 Learning By Machine
# Licensed under the MIT License. See LICENSE file in the project root.
#
# IMPLEMENTATION PARITY NOTE
# --------------------------
# This simple harness is NOT a bug-for-bug match to the internal distributed
# multi-machine/multi-GPU training stack used for our NeurIPS experiments.
# It implements the same algorithmic pieces for single-process, single-GPU/CPU
# use to keep the demo lightweight and readable.
#
# demo.py — End-to-end demo: generate data → train OCL → evaluate and correlate
# -----------------------------------------------------------------------------
from __future__ import annotations
import argparse
import time

import torch
import numpy as np

# IMPORTANT:
# This expects your previously provided research module as `model.py`
# with `train`, `test`, and the composite class `Model` (or `OCL` if you renamed).
# If your class is named `Model`, the import below of `OCL` is not required.
from model import train, test as eval_target
# from model import Model as OCL  # (Optional alias if you need it)

from data import make_synthetic_sequences
from utils import describe_batch, compute_pairwise_corr, pretty_matrix


def main(
    B: int = 3,
    V: int = 128,
    d: int = 32,
    C: int = 4,
    T_min: int = 80,
    T_max: int = 120,
    epochs: int = 50,
    lr: float = 1e-3,
    num_stages: int = 3,
    transformer_layers: int = 2,
    seed: int = 0,
):
    """
    Orchestrate a small OCL run on synthetic data, with extra prints to explain each step.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== [test] OCL single-GPU demo ===")
    print(f"[test] Device             : {device}")
    print(f"[test] Subjects (B)       : {B}")
    print(f"[test] Features (V)       : {V}")
    print(f"[test] Embedding dim (d)  : {d}")
    print(f"[test] Classes (C)        : {C}")
    print(f"[test] T range            : [{T_min}, {T_max}]")
    print(f"[test] Epochs             : {epochs}")
    print(f"[test] LR                 : {lr}")
    print(f"[test] Stages             : {num_stages}")
    print(f"[test] Transformer layers : {transformer_layers}")
    print(f"[test] Seed               : {seed}\n")

    # 1) Generate synthetic data
    X_list, y_list = make_synthetic_sequences(
        B=B, V=V, T_min=T_min, T_max=T_max, C=C, snr_db=6.0, seed=seed, device=device
    )
    describe_batch(X_list, y_list)

    # 2) Train the model (ONLINE encoders) and return TARGET embeddings for evaluation
    print("=== [test] Training begins ===")
    t0 = time.time()
    Z_list, model = train(
        X_list, y_list, V, d,
        epochs=epochs,
        lr=lr,
        tau_prop=None,                 # default heuristic 1/B
        num_stages=num_stages,
        transformer_layers=transformer_layers,
        tau=0.07,                      # optional: forward to contrastive via **mv_kwargs if desired
        margin=0.5,
        lam_between=0.1,
    )
    dt = time.time() - t0
    print(f"=== [test] Training complete in {dt:.2f}s ===\n")

    # 3) Evaluate (TARGET branch) on the same data to get final [T_i, d] per subject
    print("=== [test] Inference (TARGET encoders) ===")
    Z_eval = eval_target(X_list, model)
    for i, Zi in enumerate(Z_eval):
        print(f"[test] Subject {i}: Z_eval shape = {tuple(Zi.shape)}")
    print()

    # 4) Compute pairwise correlations before/after (raw vs embeddings)
    corr_raw, corr_emb = compute_pairwise_corr(X_list, Z_eval)
    pretty_matrix("Pairwise corr — RAW", corr_raw)
    pretty_matrix("Pairwise corr — EMB", corr_emb)

    print("=== [test] Done. ===")


if __name__ == "__main__":
    # Optional CLI
    parser = argparse.ArgumentParser(description="OCL single-GPU demo")
    parser.add_argument("--B", type=int, default=3)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--T_min", type=int, default=80)
    parser.add_argument("--T_max", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        B=args.B,
        V=args.V,
        d=args.d,
        C=args.C,
        T_min=args.T_min,
        T_max=args.T_max,
        epochs=args.epochs,
        lr=args.lr,
        num_stages=args.stages,
        transformer_layers=args.layers,
        seed=args.seed,
    )
