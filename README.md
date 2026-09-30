# MambaNIDS

> Robust Real-Time Cyberattack Detection via **Selective State-Space
> Modelling** and **Frequency-Domain Attention**

Reference implementation and reproducibility package accompanying the
manuscript submitted to the *International Journal of Machine Learning
and Cybernetics (IJMLC, Springer)*.

📄 **Manuscript (PDF):** [`docs/MambaNIDS_manuscript.pdf`](docs/MambaNIDS_manuscript.pdf)

---

## What this repository contains

| Path | Contents |
|------|----------|
| [`src/`](src/) | Library code — `models/`, `data/`, `training/`, `evaluation/` |
| [`scripts/`](scripts/) | Reproduction entry points (train, ablation, adversarial, SHAP …) |
| [`results/`](results/) | Raw result JSONs backing every table and figure in the manuscript |
| [`figures/`](figures/) | Publication-ready PNG figures |
| [`docs/`](docs/) | Manuscript PDF |
| [`config.py`](config.py) | Hyperparameters and paths |
| [`requirements.txt`](requirements.txt) | Python dependencies |

---

## Headline numbers (TON_IoT)

| Metric | Value |
|--------|-------|
| Accuracy (mean of 3 seeds) | **93.74 ± 0.65 %** |
| Macro F1 (mean of 3 seeds) | **88.94 ± 0.66 %** |
| Parameters | 1.63 M (60 % smaller than CNN-BiLSTM-MHA) |
| Inference latency (batch 512, RTX 5070) | 5.86 ms |
| McAllester PAC-Bayes bound (m = 179 386) | R(Q) ≤ **0.07** (non-vacuous) |
| Adversarial accuracy after Madry training (PGD-10, ε = 0.05) | **68.75 %** |

---

## Quick start

```bash
# 1. Create environment
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch the three public datasets (paths in config.py)
#    - TON_IoT   : https://research.unsw.edu.au/projects/toniot-datasets
#    - N-BaIoT   : https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT
#    - IoTNet24  : release URL in the manuscript

# 3. Reproduce the main table (TON_IoT, single seed)
python scripts/train.py --model mamba_nids --dataset ton_iot

# 4. Reproduce the ablation (Table 7 of the manuscript)
python scripts/ablation_mamba.py

# 5. Adversarial sweep (Fig. 4, Table 5)
python scripts/run_robustness.py --max-batches 6 --pgd-steps 10

# 6. Madry-style adversarial training (Table 6)
python scripts/adv_train.py --epsilon 0.05 --pgd-steps 7 --epochs 10

# 7. SHAP feature attribution (Fig. 5)
python scripts/run_shap.py
```

---

## How the code maps to the manuscript

| Manuscript artefact | Command |
|---|---|
| Table 1 (main TON_IoT) | `python scripts/train.py --model mamba_nids --dataset ton_iot` |
| Table 2 (multi-seed) | `python scripts/multiseed.py --models mamba_nids cnn_bilstm_mha transformer_ids cnn_lstm` |
| Table 3 (cross-dataset) | `python scripts/run_q1_extensions.py` |
| Table 4 (McNemar / Holm) | `python scripts/advanced_compare.py` |
| Table 5 + Fig. 4 (robustness) | `python scripts/run_robustness.py` |
| Table 6 (adv-trained) | `python scripts/adv_train.py` |
| Table 7 (ablation) | `python scripts/ablation_mamba.py` |
| Table 8 (complexity) | `python scripts/advanced_compare.py --complexity` |
| Fig. 5 (SHAP) | `python scripts/run_shap.py` |

Random seeds are fixed at `{7, 42, 123}` throughout — set via the
`MULTISEED_SEED` environment variable to reproduce a specific run.

---

## Architecture

```
Input window (B, T=10, F)
        │
        ▼
   1D-CNN front end   (64 → 128 → 256, kernel 3)
        │
        ▼
   Mamba stack        (2 pre-RMSNorm layers, N=16, expand=2)
        │
        ▼
  ┌── Time-domain MHA ──┐    ┌── Frequency-domain Attention ──┐
  │  8-head, d=256      │    │  rFFT → MHA on magnitude       │
  └─────────┬───────────┘    └───────────────┬────────────────┘
            │                                │
            └──── learnable per-token gate ──┘
                             │
                             ▼
                  Global average pool
                             │
                             ▼
              FC (256 → 128 → C classes)
```

See [`src/models/mamba_nids.py`](src/models/mamba_nids.py) for the
implementation and [`src/models/mamba.py`](src/models/mamba.py) for the
pure-PyTorch Mamba (selective-scan) block.

---

## Hardware used in the paper

* **GPU**: NVIDIA RTX 5070 Laptop (8.5 GB VRAM)
* **CPU**: Intel Core i7-13700H
* **Wall-clock**: ~12 minutes per MambaNIDS variant for 15 epochs on TON_IoT
  (179 k training windows after SMOTE)

The pure-PyTorch selective scan runs on any GPU or CPU; a hardware-aware
CUDA kernel (e.g., `mamba-ssm`) would push inference to sub-millisecond
latency.

---

## Citation

If you use this code or the accompanying benchmarks, please cite:

```bibtex
@article{alezzi2026mambanids,
  title   = {MambaNIDS: Robust Real-Time Cyberattack Detection via
             Selective State-Space Modelling and Frequency-Domain Attention},
  author  = {Al-Ezzi, Sufyan Jabbar Ahmed},
  journal = {International Journal of Machine Learning and Cybernetics},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

Released under the [MIT License](LICENSE).
