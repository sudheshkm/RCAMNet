### 📄 `classification_cbam/README.md`

```markdown
# CBAM-MobileNetV2 Classification Pipeline

This module implements the classification stage of **RCAMNet**:  
using CBAM-enhanced MobileNetV2 on RGB+Mask fusion.

## 📂 Files
- `train_cbam_mobilenet.py` — Script to train the classifier.
- `cbam.py` — CBAM attention module implementation.
- `dataset.py` — Custom PyTorch dataset that loads RGB images and corresponding masks.

## 🚀 Run Training
```bash
python train_cbam_mobilenet.py --epochs 50 --batch-size 32
