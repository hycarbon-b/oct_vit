# OCT-ViT: Vision Transformer for Retinal Oxygen Saturation Prediction

A Vision Transformer (ViT) pipeline for classifying and regressing **oxygen saturation** from Optical Coherence Tomography Angiography (OCTA) scans. The model supports 2D slice-based inference, 3D volumetric inference with majority-vote decoding, and transfer learning from a pretrained OCT backbone.

The ViT backbone is adapted from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch).

---

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Model Zoo](#model-zoo)
- [MixLoss](#mixloss)
- [Results](#results)

---

## Background

Retinal oxygen saturation is an important biomarker linked to various systemic and ophthalmic diseases. This project applies Vision Transformers to OCTA volumes (paired OS/OD scans) to jointly:

1. **Classify** saturation into 4 levels: `<89`, `89–92`, `93–95`, `≥96`
2. **Regress** the continuous saturation value

A custom **MixLoss** combines `BCEWithLogitsLoss` (classification) and `MSELoss` (regression), with optional per-batch positive-weight balancing.

---

## Project Structure

```
oct_vit/
├── train.py              # Vanilla ViT training script (no pretrain)
├── train2.py             # Fine-tuning script with pretrained backbone
├── util/
│   └── utilize.py        # Datasets, model builders, training loop, losses
├── vit_pytorch/
│   ├── vit.py            # Standard 2D ViT
│   ├── vit_3D.py         # 3D ViT (ViT3)
│   ├── deepvit.py        # DeepViT
│   └── ...               # Other ViT variants from lucidrains
├── log/                  # Experiment logs and metrics
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/hycarbon-b/oct_vit.git
cd oct_vit
pip install -r requirements.txt
```

---

## Data Preparation

The pipeline expects OCTA volumes in `.npy` format and a 3-fold cross-validation split. Organise your data as follows:

```
images_npy/
├── <patient_id>/
│   ├── <scan>_OS_<timestamp>.npy   # Left eye volume (H × W × D)
│   └── <scan>_OD_<timestamp>.npy   # Right eye volume (H × W × D)
...
train_0   # text file — one patient ID per line (fold 0 train)
test_0
train_1
test_1
train_2
test_2
Sleep-results.xlsx  # label file with columns [patient_id, sat_avg]
```

Update the `path`, split-file prefix, and Excel path inside `get_dataUNI()` in `util/utilize.py` to match your directory layout.

---

## Training

### Vanilla ViT (no pretrained backbone)

Edit `train.py` to set `save_path`, `wandb` credentials, and `metric_path`, then run:

```bash
python train.py
```

### Fine-tuning with a pretrained backbone

Edit `train2.py` to set `path` inside `get_model_oct_withpretrain(...)`, `save_path`, and `wandb` credentials, then run:

```bash
python train2.py
```

### Key configuration options in the `args` dict

| Key | Description |
|-----|-------------|
| `device` | PyTorch device (auto-detected by default) |
| `model` | Model builder — see `util/utilize.py` |
| `save_path` | Directory to save checkpoints |
| `epochs` | Number of training epochs |
| `lr` | Initial learning rate |
| `batch_size` | Batch size |
| `bce_weight` | Weight `α` for BCE term in MixLoss (0–1) |
| `is_MIX` | Enable MixLoss (classification + regression) |
| `wandb` | `[entity, project, run_name]` for W&B logging |
| `metric_path` | CSV file to accumulate experiment metrics |

---

## Model Zoo

| Function | Description |
|----------|-------------|
| `get_vani(outsize, dropout)` | Plain ViT with random initialisation |
| `get_model_oct_withpretrain(pretrain_out, outsize, path, dropout)` | ViT fine-tuned from a pretrained checkpoint |
| `get_model_octa_resume(outsize, path, dropout)` | Resume training from a saved OCTA checkpoint |
| `get_model_conv(pretrain_out, outsize, path, dropout)` | ViT with a 1D-conv classification head |

---

## MixLoss

`mixloss` implements a combined classification + regression criterion:

$$\mathcal{L}_\text{mix} = \alpha \cdot \text{BCEWithLogits} + (1 - \alpha) \cdot \text{MSE}$$

- Supports per-batch **positive-weight balancing** via `get_pos_weight`.
- The model output shape is `(batch, 5)`: first 4 logits for the 4-class head, last value for regression.

```python
from util.utilize import mixloss

criterion = mixloss(bce_weight=0.5)   # α = 0.5

# output: (batch, 5)  — 4 class logits + 1 regression value
# label:  (batch, 5)  — 4-class one-hot + continuous sat value
loss = criterion(output, label)
```

---

## Results

3-fold cross-validation on the OCTA saturation dataset:

| Method | Val Acc | Sensitivity | Specificity |
|--------|---------|-------------|-------------|
| Vanilla ViT | ~0.69 | — | — |
| ViT + Pretrain + MixLoss | ~0.79 | ~0.82 | — |
| ViT + Pretrain + MixLoss + 3D Vote | best split | — | — |

> Detailed per-run logs are stored in `log/`.
