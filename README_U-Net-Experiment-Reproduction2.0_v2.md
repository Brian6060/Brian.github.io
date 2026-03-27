# U-Net Experiment Reproduction 2.0

A paper-faithful PyTorch reproduction of **U-Net: Convolutional Networks for Biomedical Image Segmentation** with a full experiment pipeline covering:

- data preprocessing
- data loading
- model implementation
- training
- checkpoint loading and testing
- final inference packaging and result analysis

This project focuses on **understanding the original U-Net design logic**, not only running a modern segmentation baseline.

---

## 1. Project Goal

This reproduction has two core goals:

1. Reconstruct the original U-Net pipeline as faithfully as possible in **PyTorch**.
2. Build a complete and reusable segmentation experiment framework for later research work.

The implementation keeps the main ideas of the original paper:

- valid convolution
- encoder-decoder U-shape
- skip connection with copy-and-crop
- overlap-tile inference
- weighted loss for boundary separation
- elastic deformation based augmentation

---

## 2. Dataset

Main dataset used in this project:

- **PhC-C2DH-U373** from the Cell Tracking Challenge

Training data processing follows the original U-Net route:

- image source: `01/`, `02/`
- supervision source: `01_GT/SEG/`, `02_GT/SEG/`
- image format kept as `.tif`
- masks and weight maps cached as `.npy`

Two processed dataset branches are used:

- `processed_unet_train` or `processed_unet_strict`: training / GT-bearing data
- `processed_unet_test`: inference-only test data

---

## 3. Project Structure

```text
U-Net ER2.0/
├── src/
│   ├── DataProcessor.py
│   ├── DataLoader.py
│   ├── Model.py
│   ├── Train.py
│   ├── Test.py
│   └── Inference.py
├── processed_unet_train/
├── processed_unet_test/
├── outputs_train_formal_A/
├── outputs_test_formal_A_best/
└── README.md
```

Core scripts:

- `DataProcessor.py`  
  Unified preprocessing for train and infer modes.

- `DataLoader.py`  
  Patch-based data loader for paper-style U-Net training.

- `Model.py`  
  Original valid-convolution U-Net implementation in PyTorch.

- `Train.py`  
  Paper-style training with weighted pixel-wise cross entropy.

- `Test.py`  
  Pure checkpoint loading and overlap-tile inference.

- `Inference.py`  
  Final output packaging, probability-map export, visualization preparation, and per-image statistics.

---

## 4. Reproduction Design

### 4.1 DataLoader

The training loader is designed around the original U-Net setting:

- large input tile and smaller supervised output tile
- typical size: `572 -> 388`
- mirror padding for boundary context
- elastic deformation augmentation
- optional gray-value variation
- patch-based sampling with batch size 1

### 4.2 Model

The model keeps the original paper structure:

- unpadded `3×3` convolutions
- `2×2` max pooling in the encoder
- `2×2` up-convolution in the decoder
- copy-and-crop skip connections
- final `1×1` convolution for pixel classification

### 4.3 Train

Training follows the main paper logic:

- SGD optimizer
- momentum `0.99`
- batch size `1`
- weighted pixel-wise cross entropy
- boundary-aware weight map
- optional bottleneck dropout

### 4.4 Test

Testing uses:

- checkpoint loading from `.pt`
- overlap-tile inference
- whole-image prediction export
- probability map saving

### 4.5 Final Inference Output

The final inference stage packages each test sample into an organized folder and exports:

- original image
- predicted mask
- probability map
- visualization files
- metadata files

---

## 5. Training Output

Formal training outputs are stored in:

```text
outputs_train_formal_A/
```

Typical files include:

- `best_train_loss.pt`
- `latest.pt`
- `epoch_002.pt`, `epoch_004.pt`, ...
- `train_log.csv`

These checkpoints are later used by `Test.py` for inference.

---

## 6. Test Output

Test outputs are stored in:

```text
outputs_test_formal_A_best/
```

Typical files include:

- `pred_masks_tif/`
- `pred_masks_npy/`
- `prob_maps_npy/`
- `checkpoint_meta.json`
- `inference_manifest.json`
- `summary.json`

---

## 7. Final Output Packaging

Final packaged inference results are stored in:

```text
outputs_test_formal_A_best/final_inference_output/
```

Each sample is organized into its own folder for easier visualization and later reporting.

A typical packaged folder may include:

- `image.tif`
- `pred_mask_vis.tif`
- `prob_heatmap.tif`
- `segmentation_overlay.tif`
- `pred_mask.npy`
- `prob_map.npy`
- `meta.json`

---

## 8. Notes on Metrics

For the official test split, ground-truth masks are not provided.
Therefore:

- real segmentation metrics such as **IoU** and **Dice** cannot be computed directly on the hidden test split
- prediction-side statistics and visualization outputs are still available
- true quantitative evaluation should be run on a GT-bearing evaluation split

Metrics used in GT-bearing evaluation may include:

- IoU
- Dice
- Pixel Accuracy
- Precision
- Recall
- Specificity

---

## 9. Environment

Recommended environment:

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- tifffile
- Anaconda

Example environment activation:

```bash
conda activate ragseg_torch
```

---

## 10. Research Value of This Reproduction

This project is not only an implementation exercise. It is also used to:

- understand the technical logic of classical CV model development
- analyze how a paper turns problem setting into method design
- build a complete segmentation experiment pipeline from data to report
- prepare for later research on more advanced segmentation tasks

---

## 11. Reference

Ronneberger O, Fischer P, Brox T. **U-Net: Convolutional Networks for Biomedical Image Segmentation**. MICCAI 2015.

---

## 12. Status

Current status:

- data preprocessing completed
- paper-style model implemented
- formal training completed
- checkpoints generated
- test inference completed
- final inference output packaging completed

This repository will continue to be used for further evaluation, visualization, and report generation.
