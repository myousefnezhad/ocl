# Orthogonal Contrastive Learning (OCL)

**Single-GPU Reference Implementation — NeurIPS 2025**  
**Author:** Tony Muhammad Yousefnezhad  
**Affiliation:** Learning By Machine, Edmonton AB Canada  
**Copyright © 2025 Learning By Machine**

---

This repository provides a **clean, single-GPU version** of the core algorithm
described in the paper:

> **Orthogonal Contrastive Learning for Multi-Subject fMRI Alignment**  
> *Tony Muhammad Yousefnezhad, Learning By Machine (© 2025)*  

It reproduces the OCL training and inference pipeline with a modular design:

- [**`model.py`**](./ocl/model.py) — Core OCL model (QR decomposition, LSH hashing, Transformer encoders, contrastive loss).  
- [**`data.py`**](./ocl/data.py) — Synthetic data generator (variable-length, multi-subject time series).  
- [**`utils.py`**](./ocl/utils.py) — Helper functions (correlation analysis, formatting, summaries).  
- [**`demo.py`**](./ocl/demo.py) — Entry point for training, evaluation, and visualizing subject-wise correlations.

---

## 🔬 Algorithm Overview

OCL enforces **orthogonal subspace alignment** across multiple subjects using
a combination of:

1. **QR decomposition** to maintain orthonormal temporal bases  
2. **2-stable LSH signatures** for subject-specific conditioning  
3. **Transformer encoders** for cross-subject temporal refinement  
4. **EMA online/target encoders** for contrastive self-supervision  

Each stage learns progressively aligned representations while preserving
temporal order and orthogonality constraints.

---

## ⚠️ Implementation Parity Note

This reference code is **not a bug-for-bug match** to the internal multi-machine,
multi-GPU version used in the NeurIPS experiments.  
It reproduces the **same algorithmic flow**—QR, LSH, Transformer refinement, and
EMA target updates—but runs in a single process on **CPU or one GPU** for
clarity and reproducibility.

---

## 🧩 Installation

```bash
git clone https://github.com/yourusername/ocl.git
cd ocl
pip install -r requirements.txt
cd ocl
python demo.py
```
