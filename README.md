## Getting Started

### Install dependencies
```bash
pip install -r requirements.txt

This version:
# Paddy Pathologist: Multimodal Fusion for Rice Bacterial Leaf Blight Severity Assessment 🌾

## What is this repository?

This repository contains the source code associated with the research work:

**"Multimodal Fusion of Visual and Lesion Features for Mobile-Based Severity Assessment of Rice Bacterial Leaf Blight"**

The work presents a multimodal framework for stage-wise assessment of Bacterial Leaf Blight (BLB) severity in rice. The framework combines learned visual features with structured lesion descriptors to support severity classification and mobile deployment.

The pipeline integrates lightweight rice leaf segmentation, HSV-based lesion masking, MobileNetV2 with CBAM attention, handcrafted lesion descriptors, and feature-level multimodal fusion.

## Key Features

- 🌱 **Lightweight leaf segmentation** using U-Net
- 🎨 **HSV-based lesion masking** for extracting disease-affected regions
- 🧠 **MobileNetV2 with CBAM** for visual feature extraction
- 📊 **Structured lesion descriptors** representing lesion area, colour, and texture characteristics
- 🔗 **Feature-level multimodal fusion** of visual and lesion descriptors
- 📱 **TensorFlow Lite deployment** for on-device inference
- 🌾 **Stage-wise BLB severity assessment** across five severity stages
- 🔍 **Grad-CAM-based interpretability** for visual analysis of model predictions

## Framework

The proposed pipeline consists of the following major stages:

1. Input rice leaf image
2. Rice leaf segmentation using lightweight U-Net
3. HSV-based lesion-mask generation
4. Visual feature extraction using MobileNetV2 with CBAM
5. Extraction of structured lesion descriptors
6. Feature-level multimodal fusion
7. Five-stage BLB severity classification
8. TensorFlow Lite conversion for mobile deployment


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

## Framework

The proposed pipeline consists of the following major stages:

1. Input rice leaf image
2. Rice leaf segmentation using lightweight U-Net
3. HSV-based lesion-mask generation
4. Visual feature extraction using MobileNetV2 with CBAM
5. Extraction of structured lesion descriptors
6. Feature-level multimodal fusion
7. Five-stage BLB severity classification
8. TensorFlow Lite conversion for mobile deployment


## Code availability

The code is archived in Zenodo and is associated with the GitHub
repository through a versioned release.


## Dataset

The experiments use **BLBVisionDB**, a rice Bacterial Leaf Blight progression dataset introduced in our previous work.

Dataset information is available through the project webpage:

https://sudheshkm.github.io/Bacterial-Leaf-Blight-disease-progression/

The dataset is available from the corresponding author upon reasonable request.

## Model Weights

The trained model weights used for inference are available from the corresponding author upon reasonable request.

For requests regarding the dataset or model weights, please contact the corresponding author.

## Mobile Application

The trained models were integrated into the **Paddy Pathologist** Android application for on-device rice disease assessment.

The application supports:

- Automatic leaf detection
- Manual image capture
- Gallery-based analysis
- BLB severity assessment
- Multilingual user interface
- Stage-wise remedial recommendations
- Agricultural officer communication through SMS

The severity assessment model described in the associated manuscript focuses specifically on **Bacterial Leaf Blight**, based on the stage-labelled BLB dataset used in this study.

## Repository Scope

This repository is associated with the multimodal BLB severity assessment work described in the manuscript above.

The repository name **RCAMNet** is retained for continuity with the earlier implementation and previous work. The present study extends that earlier visual framework by incorporating structured lesion descriptors and multimodal feature fusion.

## Citation

If you use the methodology, source code, or related work from this repository, please cite:

Sudhesh K M, Aarthi R, Sainamole Kurian P, Sikha O K.

**"Multimodal Fusion of Visual and Lesion Features for Mobile-Based Severity Assessment of Rice Bacterial Leaf Blight."**

## Contact

For access to the BLBVisionDB dataset or trained model weights, please contact the corresponding author.
