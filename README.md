# bid-sage

GPU-accelerated AdTech machine learning pipeline for real-time bidding (RTB), built as a hands-on portfolio project exploring deep learning and GPU-accelerated data engineering in an AdTech context.

---

## Motivation

Real-time bidding systems make sub-100ms decisions on billions of ad impressions per day. The models driving those decisions — CTR predictors, bid optimizers — sit at the intersection of large-scale data engineering and applied ML. This project exists to build and demonstrate practical fluency in that stack: GPU-accelerated data processing, deep learning for tabular AdTech data, and eventually an AI-powered campaign optimization layer.

Documented publicly on Substack: [The Inference Layer](https://substack.com/@connerlewis)

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | CTR prediction model (Deep & Cross Network) trained on Criteo Click Logs | 🔨 In Progress |
| 2 | Simulated real-time bidding pipeline | 📋 Planned |
| 3 | Claude-powered campaign analyst agent layer | 📋 Planned |

---

## Phase 1: CTR Prediction

### Model
**Deep & Cross Network (DCN)** — a dual-tower architecture combining:
- **Cross network**: learns explicit, bounded-degree feature interactions via a series of cross layers anchored to the input vector `x₀`
- **Deep network**: learns implicit, non-linear feature representations via stacked fully-connected layers

Both towers share the same input embedding and their outputs are concatenated before a final sigmoid prediction.

### Dataset
[Criteo Click Logs](https://www.kaggle.com/c/criteo-display-ad-challenge) — ~45M rows of real display ad impressions with binary click labels.

Key characteristics:
- 13 numerical features, 26 categorical features
- ~97% negative / ~3% positive labels (severe class imbalance)
- Class imbalance is handled via `BCEWithLogitsLoss` with `pos_weight=32.0`

### Data Pipeline
GPU-accelerated preprocessing via **RAPIDS cuDF**:
- Loads raw TSV data directly into GPU memory
- Log-normalizes numerical features using `cupy.log1p()`
- Label-encodes categorical features
- Returns PyTorch-compatible tensors

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| GPU compute | NVIDIA RTX 4080 SUPER, CUDA Toolkit |
| Data processing | RAPIDS cuDF, CuPy |
| Modeling | PyTorch (CUDA-enabled) |
| Environment | WSL2 (Ubuntu 22.04), conda `rapids-26.02` |
| Infrastructure | Docker + NVIDIA Container Toolkit |
| Dev tooling | VS Code (WSL2 remote), GitHub CLI |

---

## Project Structure
```
bid-sage/
├── src/
│   ├── data/
│   │   ├── loader.py        # cuDF-based GPU data pipeline
│   │   └── test_loader.py
│   └── models/
│       ├── dcn.py           # Deep & Cross Network (PyTorch)
│       └── test_dcn.py
└── .gitignore
```

---

## Environment Setup

Requires WSL2 with Ubuntu 22.04 and an NVIDIA GPU with CUDA support.
```bash
# Activate the conda environment (required each session)
conda activate rapids-26.02

# Verify GPU is accessible
nvidia-smi
```

> **Note:** If `nvidia-smi` segfaults in WSL2, run `wsl --update` from a Windows terminal.

---

## Known Learnings & Gotchas

- `cuDF` has no native `.log()` — use `cupy.log1p()` instead
- Python lambda closures in loops require explicit binding: `lambda x, enc=encoder: enc.transform(x)`
- Large data files must be excluded via `.gitignore` before the first commit
- WSL2 `nvidia-smi` segfault → fix with `wsl --update`
- AdTech CTR data is heavily imbalanced; without explicit loss weighting, models converge to predicting all negatives (AUC ~0.5)
