# -----------------------------------------------------------------------------
# Copyright (c) 2025 Learning By Machine
# Licensed under the MIT License. See LICENSE file in the project root.
#
# model.py — Core OCL model (QR decomposition, LSH hashing, Transformer encoders, contrastive loss). 
# -----------------------------------------------------------------------------
"""
NeurIPS 2025 Paper — Orthogonal Contrastive Learning (OCL) for Multi-Subject fMRI Alignment
Author: Tony Muhammad Yousefnezhad
Copyright (c) 2025 Learning By Machine
--------------------------------------------------------------------------------------

This module implements the core architectural components and training routine
used in our NeurIPS 2025 paper on Orthogonal Contrastive Learning (OCL). The goal is
to learn subject-invariant, time-resolved embeddings from variable-length
neuroimaging sequences (e.g., fMRI time series), while preserving temporal
structure and orthogonality constraints via a thin QR factorization.

IMPLEMENTATION PARITY NOTE
--------------------------
This reference implementation is **not** a bug-for-bug match to the internal codebase
we used to run the NeurIPS experiments. Our production stack trains/evaluates the
model in a **distributed, multi-machine, multi-GPU** setting (with data/model
parallelism, custom/fused ops, asynchronous input pipelines, gradient checkpointing,
and mixed precision). 

The code below captures the **same algorithmic components** described in the paper
(QR-based orthogonal projection, LSH conditioning, Transformer encoder refinements,
and EMA teacher/target updates) but is written for **single-process, single-GPU (or
CPU)** usage to maximize clarity and reproducibility. As a result, you may observe
small numerical and scheduling differences (e.g., kernel fusion, ordering of ops,
and reduction semantics). These should not alter the method’s training/inference
**semantics**; they only affect low-level execution details and throughput.

High-level pipeline per refinement stage:
1) QRModule:    Performs a thin QR decomposition on each subject’s sequence X.
                - Q (T×d) serves as an orthonormal temporal basis truncated to d.
                - R (d×V) captures subject-/voxel-specific loadings.
2) LSHModule:   Computes a 2-stable LSH signature over vec(R) to produce a
                compact, subject-level conditioning signal s.
3) Encoder:     Injects s into Q, applies positional encoding + Transformer
                blocks, and projects back to d to yield refined embeddings Z.
4) BYOL-style:  Maintains online/target encoders; target is an EMA of online.
5) Loss:        Contrastive objective over timepoints (within/between subjects),
                with an additional soft margin to separate negatives.

Inputs & Shapes
---------------
- X_list: list[Tensor], length B (batch of subjects/runs)
          Each X has shape [T_i, V_in], where T_i may differ across subjects.
          At stage 0, V_in = V (voxels/features); later stages use V_in = d.
- y_list: list[Tensor], one label per timepoint (e.g., stimulus/class IDs);
          concatenated length must be >= sum(T_i).

Key Dimensions
--------------
- B: batch size (# sequences / subjects)
- T_i: sequence length for item i (variable)
- T_max: max_i T_i
- V: input feature dimension at stage 0 (e.g., #voxels)
- d: embedding dimension retained by thin-QR and used throughout
- num_stages: number of refinement passes (QR→LSH→Encoder)

Notes
-----
- The QR factor enforces orthogonality in Q and constrains the embedding space.
- LSH(s) is a lightweight summary of R and acts as a conditioning “signature.”
- Transformer encoders are applied with key-padding masks for variable T.
- Target encoders are updated by EMA with rate `tau`; only ONLINE params are
  optimized by Adam.
- The provided contrastive loss is InfoNCE-like for positives (same labels)
  plus a soft-margin push for negatives.

Cite/Attribution
----------------
If you use or modify this code, please cite the NeurIPS paper associated with
OCL (Orthogonal Contrastive Learning) by the authors. This file is a compact,
commented reference implementation intended for research and reproducibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration (CUDA if available, otherwise CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# QR Decomposition with Static Output Dimension
# ──────────────────────────────────────────────────────────────────────────────
class QRModule(nn.Module):
    """
    Thin QR decomposition for variable-length inputs with fixed `d_out`.

    Given a batch of variable-length sequences X_list = [X_i], with
    X_i ∈ ℝ^{T_i × V_in}, this module:
      - Computes a thin QR: X_i = Q_i R_i with Q_i ∈ ℝ^{T_i × min(T_i, V_in)},
        R_i ∈ ℝ^{min(T_i, V_in) × V_in}.
      - Truncates Q_i and R_i to the first `d_out` rows/columns.
      - Pads Q across time to T_max and returns a key-padding mask.

    Args:
        d_out (int): Desired output embedding dimension d (≤ min(T_i, V_in)).
        in_dim (int): Input feature dimension V_in for this stage.

    Returns:
        Q (Tensor): [B, T_max, d_out]  time-padded orthonormal bases.
        R (Tensor): [B, d_out, V_in]   truncated R factors.
        mask (BoolTensor): [B, T_max]  True for valid time positions.
    """
    def __init__(self, d_out, in_dim):
        super().__init__()
        self.d_out = d_out
        self.in_dim = in_dim

    def forward(self, X_list):
        B = len(X_list)                       # number of sequences
        Ts = [x.shape[0] for x in X_list]     # individual lengths
        T_max = max(Ts)                       # max length for padding
        d = self.d_out                        # embedding dimension
        V = self.in_dim                       # input feature size at this stage

        # Allocate padded containers on the selected device
        Q = torch.zeros(B, T_max, d, device=device)
        R = torch.zeros(B, d, V, device=device)
        mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)

        # Process each sequence independently
        for i, X in enumerate(X_list):
            X = X.to(device)                           # [T_i, V]
            # Thin QR decomposition (reduced mode)
            Qi, Ri = torch.linalg.qr(X, mode='reduced')
            # Truncate to d (handles cases where min(T_i, V) > d)
            Qi = Qi[:, :d]                             # [T_i, d]
            Ri = Ri[:d]                                # [d, V]
            ti = Qi.size(0)                            # actual time length T_i
            # Time-pad Q and record valid positions in mask
            Q[i, :ti]    = Qi
            R[i]         = Ri
            mask[i, :ti] = True

        return Q, R, mask


# ──────────────────────────────────────────────────────────────────────────────
# LSH Hashing
# ──────────────────────────────────────────────────────────────────────────────
class LSHModule(nn.Module):
    """
    2-stable LSH signature of vec(R).

    Implements a single random projection + binning:
        h(v) = floor((v·a + b) / w),
    where a ~ N(0, I), b ~ Uniform(0, w).

    Args:
        D (int): Flattened dimensionality of R, i.e., d * V_in at this stage.
        w (float): Bucket width; larger w → coarser bins.

    Input:
        R: [B, d, V_in]

    Output:
        s: [B] integer-valued signatures (float tensor with floored values).
    """
    def __init__(self, D, w=1.0):
        super().__init__()
        # Random projection vector a ∈ ℝ^D and random offset b ∈ [0, w)
        self.register_buffer('a', torch.randn(D, device=device))
        self.register_buffer('b', torch.rand(1, device=device) * w)
        self.w = w

    def forward(self, R):
        B, d, V = R.shape
        v = R.view(B, d * V)                 # flatten R → [B, D]
        # Compute bucketed projection; returned as float (can keep as conditioning)
        return torch.floor((v @ self.a + self.b) / self.w)


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding (sinusoidal)
# ──────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017),
    precomputed up to `max_len` and added to inputs.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model, device=device)
        pos = torch.arange(max_len, device=device).unsqueeze(1).float()
        # Geometric progression of frequencies for even/odd dims
        div = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        # Store as [1, max_len, d_model] for easy broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            x + pe[:, :T, :]
        """
        return x + self.pe[:, :x.size(1), :]


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Block for OCL
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Sequence encoder that conditions Q on a subject/run signature s,
    then applies a Transformer encoder and projects back to d.

    Flow:
      Q  : [B, T, d]
      s  : [B] (scalar signature per item) → embed → [B, d] → broadcast to [B, T, d]
      x  = Q + s_emb
      x  → Linear(d→d_model) → PosEnc → LayerNorm → Transformer → Linear(d_model→d)

    Args:
      d (int): base embedding dimension (matches QR truncation)
      d_model (int): Transformer model dimension
      nhead (int): # attention heads
      layers (int): # TransformerEncoder layers
      max_len (int): max sequence length for positional encoding
    """
    def __init__(self, d, d_model=64, nhead=4, layers=2, max_len=5000):
        super().__init__()
        # Embed scalar signature s into a d-dim vector, then broadcast across time
        self.sig_embed = nn.Linear(1, d)
        # Project Q+s into Transformer dimension
        self.q_proj    = nn.Linear(d, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.norm      = nn.LayerNorm(d_model)

        # Transformer encoder stack (batch_first=True → [B, T, C])
        block = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model,
            batch_first=True
        )
        self.transf   = nn.TransformerEncoder(block, layers)
        # Project back to d to keep stage interface consistent
        self.out_proj = nn.Linear(d_model, d)

    def forward(self, Q, s, mask):
        """
        Args:
          Q    : [B, T, d]    (time-padded)
          s    : [B]          (LSH signature per sequence; float)
          mask : [B, T] bool  (True = valid timestep; used as key padding mask)

        Returns:
          Z    : [B, T, d] refined embeddings
        """
        B, T, _ = Q.shape
        # Embed scalar s → [B, d], then broadcast over time → [B, T, d]
        s_emb = self.sig_embed(s.unsqueeze(-1)).unsqueeze(1).expand(-1, T, -1)
        x = Q + s_emb                             # inject subject/run signature
        x = self.q_proj(x)                        # [B, T, d_model]
        x = self.pos_enc(x)                       # add sinusoidal positions
        x = self.norm(x)
        # Transformer uses False for valid tokens, True for pads; hence ~mask
        x = self.transf(x, src_key_padding_mask=~mask)
        return self.out_proj(x)                   # [B, T, d]


# ──────────────────────────────────────────────────────────────────────────────
# Composite OCL Model (multi-stage refinement + BYOL-style targets)
# ──────────────────────────────────────────────────────────────────────────────
class OCL(nn.Module):
    """
    Multi-stage OCL model with QR→LSH→Encoder refinements.

    - Each stage i:
        * QRModule(d_out=d, in_dim = V if i==0 else d)
        * LSHModule(D = d * (V if i==0 else d))
        * Online/Target Encoder pair (same architecture)
    - Only ONLINE encoders are optimized; TARGET encoders updated via EMA with rate tau.

    Args:
        d (int): embedding dimension retained by QR and encoders
        V (int): initial input feature dimension (e.g., voxels)
        num_stages (int): number of refinement stages
        transformer_layers (int): layers per encoder
        tau (float): EMA rate for target parameter updates (0<tau≤1)
    """
    def __init__(self, d, V, num_stages=5, transformer_layers=2, tau=0.1):
        super().__init__()
        self.d, self.V = d, V
        self.depth    = num_stages
        self.tau      = tau

        # Stage-wise QR modules; stage 0 consumes V, later stages consume d
        self.qr_modules = nn.ModuleList([
            QRModule(d_out=d, in_dim=(V if i == 0 else d))
            for i in range(self.depth)
        ])

        # Stage-wise LSH modules over vec(R) with appropriate flattened size
        self.lsh_modules = nn.ModuleList([
            LSHModule(D=(d * (V if i == 0 else d)))
            for i in range(self.depth)
        ])

        # Create paired online/target encoders per stage (same init; target ← online)
        self.online_encs = nn.ModuleList()
        self.target_encs = nn.ModuleList()
        for _ in range(self.depth):
            online = Encoder(d, d_model=64, nhead=4, layers=transformer_layers)
            target = Encoder(d, d_model=64, nhead=4, layers=transformer_layers)
            # Hard copy weights so target starts identical to online
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data.copy_(po.data)
            self.online_encs.append(online)
            self.target_encs.append(target)

    @torch.no_grad()
    def update_targets(self):
        """
        Exponential moving average (EMA) update for TARGET encoders:
            θ_t ← (1 − τ) θ_t + τ θ_o
        """
        for online, target in zip(self.online_encs, self.target_encs):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.data.mul_(1 - self.tau).add_(po.data * self.tau)

    def forward_online(self, X_list):
        """
        Forward pass through ONLINE encoders for training.

        Args:
          X_list: list of T_i×V (stage 0) then T_i×d (later stages)

        Returns:
          Z   : [B, T_max, d] final stage embeddings (padded)
          mask: [B, T_max]    validity mask
        """
        B = len(X_list)
        lengths = [x.shape[0] for x in X_list]  # keep original lengths per item
        inputs = X_list
        mask = None
        Z = None

        for i in range(self.depth):
            Q, R, mask = self.qr_modules[i](inputs)   # QR + mask
            s = self.lsh_modules[i](R)                # subject/run signature
            Z = self.online_encs[i](Q, s, mask)       # refine embeddings
            # Prepare inputs for next stage as a list of unpadded sequences
            inputs = [Z[b, :lengths[b]] for b in range(B)]

        return Z, mask

    def forward_target(self, X_list):
        """
        Forward pass through TARGET encoders (evaluation or teacher branch).

        Returns:
          Z_list: list of length B with unpadded [T_i, d] embeddings
          mask  : [B, T_max] mask from the last stage (for reference)
        """
        B = len(X_list)
        lengths = [x.shape[0] for x in X_list]
        inputs = X_list
        mask = None
        Z = None

        for i in range(self.depth):
            Q, R, mask = self.qr_modules[i](inputs)
            s = self.lsh_modules[i](R)
            Z = self.target_encs[i](Q, s, mask)
            inputs = [Z[b, :lengths[b]] for b in range(B)]

        return inputs, mask

    # ----------------------- Serialization helpers -----------------------------

    def save_online(self, path):
        """Save state_dicts of all ONLINE encoders to a single file."""
        state = {f"online_enc_{i}": enc.state_dict()
                 for i, enc in enumerate(self.online_encs)}
        torch.save(state, path)

    def save_target(self, path):
        """Save state_dicts of all TARGET encoders to a single file."""
        state = {f"target_enc_{i}": enc.state_dict()
                 for i, enc in enumerate(self.target_encs)}
        torch.save(state, path)

    def load_online(self, path, map_location=None):
        """Load ONLINE encoder weights from a file."""
        map_loc = map_location or device
        state = torch.load(path, map_location=map_loc)
        for i, enc in enumerate(self.online_encs):
            enc.load_state_dict(state[f"online_enc_{i}"])

    def load_target(self, path, map_location=None):
        """Load TARGET encoder weights from a file."""
        map_loc = map_location or device
        state = torch.load(path, map_location=map_loc)
        for i, enc in enumerate(self.target_encs):
            enc.load_state_dict(state[f"target_enc_{i}"])

    def save(self, path):
        """Save the full composite model state_dict (all modules)."""
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        """Load the full composite model state_dict and move to device."""
        loc = map_location or device
        state = torch.load(path, map_location=loc)
        self.load_state_dict(state)
        self.to(loc)


# ──────────────────────────────────────────────────────────────────────────────
# Loss, Train, Test
# ──────────────────────────────────────────────────────────────────────────────
def contrastive_loss(z, y, tau=0.1, margin=0.5, lam_between=0.1):
    """
    InfoNCE-style contrastive loss with an extra soft-margin term for negatives.

    Args:
      z (Tensor): [N, d] stacked embeddings (e.g., timepoints across batch)
      y (Tensor): [N]   integer labels; positives share the same label
      tau (float): temperature for similarity scaling
      margin (float): margin for negative separation (softplus on sim - margin)
      lam_between (float): weight for the negative push term

    Returns:
      scalar loss
    """
    device = z.device
    # Normalize to unit vectors; cosine similarity via dot product
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.t() / tau                    # [N, N]

    # Build positive mask (same labels), exclude self-pairs on diagonal
    mask = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float().to(device)
    pos_mask = mask * (1 - torch.eye(mask.size(0), device=device))

    # Numerator: sum over positives; Denominator: all pairs
    num = torch.exp(sim * pos_mask).sum(dim=1)
    den = torch.exp(sim).sum(dim=1)
    c_loss = -torch.log((num + 1e-12) / (den + 1e-12)).mean()

    # Additional negative separation with soft margin:
    neg_sim = sim * (1 - mask)                         # keep only negatives
    b_loss = torch.log1p(torch.exp(neg_sim - margin)).mean()

    return c_loss + lam_between * b_loss


def train(X_list, y_list, V, d,
          epochs=1000, lr=1e-3,
          tau_prop=None,
          num_stages=5,
          transformer_layers=2,
          **mv_kwargs):
    """
    Train ONLINE encoders end-to-end; update TARGET by EMA each step.

    Args:
      X_list (list[Tensor]): inputs; stage 0 expects [T_i, V]
      y_list (list[Tensor]): labels per timepoint; concatenated length ≥ ΣT_i
      V (int): input feature dimension at stage 0
      d (int): embedding dimension after QR/encoders
      epochs (int): training iterations
      lr (float): learning rate for Adam (ONLINE encoders only)
      tau_prop (float|None): EMA rate; defaults to 1/B (BYOL heuristic)
      num_stages (int): number of refinement stages
      transformer_layers (int): layers per encoder
      **mv_kwargs: extra args to contrastive_loss (e.g., tau, margin, lam_between)

    Returns:
      Z_list (list[Tensor]): TARGET embeddings per sequence (unpacked [T_i, d])
      model (Model): trained composite model
    """
    B   = len(X_list)
    tau = (1.0 / B) if tau_prop is None else tau_prop

    # Build model and move to device
    model = OCL(d, V,
                  num_stages=num_stages,
                  transformer_layers=transformer_layers,
                  tau=tau).to(device)

    # Optimize ONLINE encoder parameters only
    opt = torch.optim.Adam(
        [p for enc in model.online_encs for p in enc.parameters()],
        lr=lr
    )

    for e in range(epochs):
        # ONLINE forward
        Z, mask = model.forward_online(X_list)    # Z: [B, T_max, d], mask: [B, T_max]

        # Pack valid time steps across batch into [N, d]
        flat = mask.view(-1)                      # [B*T_max]
        zf   = Z.view(-1, d)[flat]               # [N, d]

        # Build labels [N]; ensure device alignment and truncate to N
        yf   = torch.cat(y_list).to(device)[:zf.size(0)]

        # Compute loss and update ONLINE; EMA update TARGET
        loss = contrastive_loss(zf, yf, **mv_kwargs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_targets()

        # Simple progress print (consider using a logger in production)
        print(f"Epoch: {e:5d} Loss: {loss}")

    # Return TARGET embeddings for evaluation/inference
    Z_list, _ = model.forward_target(X_list)
    return Z_list, model


def test(X_list, model):
    """
    Convenience evaluation: returns TARGET embeddings as a list of [T_i, d].
    """
    with torch.no_grad():
        return model.forward_target(X_list)[0]
