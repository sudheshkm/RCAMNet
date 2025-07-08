**RCAMNet: Rice BLB Severity Analysis with Dual-Path Attention and Segmentation**
🌾 **What is RCAMNet?**
RCAMNet is a novel multi-task deep learning framework for accurate classification and severity analysis of BLB.
It integrates segmentation-driven feature enhancement with a lightweight, attention-augmented classifier to achieve robust and interpretable predictions.

📋 **Key Features**

✅ **Multi-class segmentation to isolate diseased regions:**


Detectron2 is preferred for its instance segmentation capability.

✅ **Dual-path attention mechanism:**

The Convolutional Block Attention Module (CBAM) is applied independently on:

Raw RGB image

Segmentation mask

Highlights key visual and spatial features.

✅ **Feature fusion & classification:**

Enhanced features are fused and passed into a lightweight MobileNetV2 classifier.


📁**Repository Structure**
bash
Copy
Edit
RCAMNet/
│
├── detectron2_segmentation/
│   ├── train_detectron2.py      # Train Detectron2 segmentation model
│   ├── config.yaml              # Detectron2 config file
│   ├── utils.py                 # Helper functions
│   └── README.md                # Instructions for segmentation pipeline
│
├── classification_cbam/
│   ├── train_cbam_mobilenet.py  # Train CBAM + MobileNetV2 classifier
│   ├── cbam.py                  # CBAM implementation
│   ├── dataset.py               # Dataset loader
│   └── README.md                # Instructions for classification pipeline
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files/folders to ignore
└── README.md                    # This file
🛠️ **Installation & Usage**
bash
Copy
Edit
# Clone the repo
git clone https://github.com/sudheshkm/RCAMNet.git
cd RCAMNet

# Install dependencies
pip install -r requirements.txt

# Run segmentation training
cd detectron2_segmentation
python train_detectron2.py

# Run classification training
cd ../classification_cbam
python train_cbam_mobilenet.py
See the respective README.md files in each subfolder for detailed instructions.

## Pretrained Weights
We have not included trained weights in this repository to avoid large file storage. Please contact the corresponding author to request pretrained weights.

📜 **Citation**
If you use this work, please cite:
Sudhesh K M et al., "RCAMNet: Segmentation-driven Dual-path Attention Framework for Rice BLB Severity Analysis", 2025.
