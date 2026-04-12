# ViTUNet vs SwinUNet on Medical Image Segmentation

## 1. Project Overview

This project compares ViTUNet and SwinUNet on Kvasir-SEG with unified
pipeline.

## 2. Key Results

  Model      IoU      Dice     AP
  ---------- -------- -------- --------
  ViTUNet    0.8493   0.9096   0.9645
  SwinUNet   0.8390   0.8970   0.9520

## 3. Core Findings

-   Swin: better validation, worse test
-   ViT: more robust on hard samples
-   Swin suffers long-tail failure

## 4. Structure

Swin-ViT/ ├── ViTUnet vs SwinUnet/ ├── Dataset/ ├── ViTUnet/ ├──
SwinUnet/

## 5. Training

python src/train_vitunet.py python src/train_swinunet.py

## 6. Testing

python src/test_vitunet.py --checkpoint PATH python src/test_swinunet.py
--checkpoint PATH

## 7. Metrics

IoU / Dice / AP

## 8. Author

Brian, SCU VCL318
