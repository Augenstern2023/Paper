# Light_asf_net: Fine-grained Disease Identification Network of Crop Leaves Based on Local Feature Enhancement

This repository contains the implementation of **Light_asf_net**, a lightweight dual-branch network for fine-grained classification of crop leaf diseases.

## Abstract

Fine-grained classification of crop leaf diseases faces challenges such as large intra-class variation, inter-class ambiguity, and background interference. We propose **Light_asf_net**, which features:
- **Global Branch**: ASF-former Transformer to capture leaf morphology.
- **Local Branch**: MBConv module to extract lesion microstructures.
- **Adaptive Feature Fusion (AFF)**: Enables collaborative local-global enhancement.

**Results**:
- **Cassava Leaf Disease Dataset**: 70.45% mAcc (37.8% improvement over ResNet-101) with only 6.03M parameters.
- **Rice Leaf Disease Dataset**: 100% accuracy within 12 epochs.
- **Inference Speed**: 35 FPS on mobile devices (Snapdragon 865).

## Directory Structure

```
.
├── net/                # Network definitions (Light_asf_net, etc.)
├── models/             # ASF-former dependencies
├── tools/              # Utilities and Dataset classes
├── dataset_toolbox/    # Scripts for data processing
├── ASF-former-main/    # Upstream ASF-former reference
├── train.py            # Training script
├── predict.py          # Inference/Evaluation script
├── config.py           # Configuration file
└── requirements.txt    # Dependencies
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.8.0
- torchvision
- albumentations
- torchmetrics
- tensorboard

## Data Preparation

### Rice Leaf Disease Dataset
1.  Place your images in `Data/Rice Leaf Disease Images/<CategoryName>/`.
2.  Use scripts in `dataset_toolbox/` to generate the `processed_data/` folder and `train.csv`.
    - Example: Run `dataset_toolbox/3.dataset_division_new.py` (you may need to adjust paths).

### Cassava Leaf Disease Dataset
Similar structure applies. Ensure your CSV file format matches:
```csv
图片名,标签
image_name.jpg,label_index
```

**Note**: Update `config.py` with your `DATA_DIR` and `LABEL_DIR` paths.

## Usage

### Training

Configure parameters in `config.py`, then run:

```bash
python train.py
```

### Inference / Evaluation

To evaluate trained models:

```bash
python predict.py
```

## Configuration

Modify `config.py` to change:
- `DATA_DIR`: Path to dataset images.
- `LABEL_DIR`: Path to training labels CSV.
- `MODEL_NAME`: "light_asf_net" (default).
- `BATCH_SIZE`, `LR`, `MAX_EPOCH`: Training hyperparameters.

## Reference

*Research on Fine-grained Disease Identification Network of Crop Leaves Based on Local Feature Enhancement*
Liwei Fan, Siyan Liu, Kai Cai, Sixing Lu, Ke Zhang, Huixian Chen.
School of Electronic Information and Control Engineering, Guangzhou University of Software.
