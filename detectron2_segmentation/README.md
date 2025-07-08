# Detectron2 Segmentation Pipeline

This module implements the segmentation stage of **RCAMNet** using Detectron2.

## 📂 Files
- `train_detectron2.py` — Script to train an instance segmentation model (Detectron2).
- `config.yaml` — Config file for Detectron2 hyperparameters.
- `utils.py` — Utility functions (e.g., data preprocessing, evaluation).

## 🚀 Run Training
```bash
python train_detectron2.py --config config.yaml
