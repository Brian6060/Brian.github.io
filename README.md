# U-Net Experiment Reproduction

A lightweight reproduction of U-Net for biomedical image segmentation in a CPU-only local environment.

This project implements the core U-Net pipeline in **Caffe**, using the **PhC-C2DH-U373** microscopy dataset in a simplified local validation setting. The goal is not to reproduce the official benchmark numbers from the original paper, but to build a complete and analyzable reproduction workflow covering:

- dataset preparation
- preprocessing
- weighted training in Caffe
- checkpointing and validation
- deployment with `deploy.prototxt`
- inference and evaluation through **OpenCV DNN**

## Final reported setting

The final stable local result in this repository corresponds to:

- framework: **Caffe**
- task: **binary semantic segmentation**
- dataset: **PhC-C2DH-U373**
- image size: **256 x 256**
- final checkpoint: `unet_weighted_iter_1500.caffemodel`
- inference threshold: **0.56**
- local validation result:
  - **Mean IoU = 0.252561**
  - **Mean Pixel Error = 0.054223**

This result is based on a simplified local validation split with 7 validation samples. It is not directly comparable to the official challenge-level evaluation in the original U-Net paper.

## Repository structure

```text
U-Net Experiment Reproduction/
├── README.md
├── requirements.txt
├── caffe/                     # Caffe source tree and compiled binaries
├── data/
│   ├── raw/                   # raw images and raw instance masks
│   ├── processed/             # resized grayscale images and binary masks
│   ├── splits/                # train.txt and val.txt
│   ├── h5/                    # earlier HDF5 exports
│   └── h5_weighted/           # weighted HDF5 used by final training
├── outputs/
│   ├── checkpoints/           # .caffemodel and .solverstate files
│   ├── logs/                  # Caffe training logs
│   └── inference*/            # saved inference outputs
├── reports/
│   ├── figures/               # figures used in the report
│   └── *.tex                  # report source files
├── scripts/                   # data export, plotting, inference, evaluation
└── src/                       # Caffe prototxt files and src-specific docs
    └── README.md
```

## Environment

### Hardware

- MacBook Pro
- Intel i5 CPU
- 8 GB RAM
- no GPU

### Software

- macOS
- Python 3.9
- Caffe
- Miniconda
- OpenCV DNN for inference

## Dataset

This project uses the **PhC-C2DH-U373** dataset from the Cell Tracking Challenge.

In this reproduction:

- all inputs are converted to **single-channel grayscale**
- instance masks are converted into **binary semantic masks**
- foreground is defined as `pixel > 0`
- all images and masks are resized to **256 x 256**
- the local split uses:
  - **28 training samples**
  - **7 validation samples**

## Training variants used during the project

Several training variants were explored:

1. single-channel output with `SigmoidCrossEntropyLoss`
2. two-channel output with `SoftmaxWithLoss`
3. weighted two-channel output with a custom `WeightedSoftmaxWithLoss`

The final reported result comes from the **weighted** version.

## Important implementation notes

### 1. Training is done in Caffe

The core training pipeline is defined in `src/train_val.prototxt` and launched through Caffe solver files.

### 2. Inference is not done with pycaffe

`pycaffe` caused runtime library conflicts on the local macOS setup. To keep deployment stable, inference is performed with **OpenCV DNN** instead.

### 3. Weighted training uses HDF5 with three blobs

The final weighted training pipeline reads:

- `data`
- `label`
- `weight`

from weighted HDF5 files.

### 4. Threshold calibration matters

The weighted model does not produce a sharply separated foreground probability distribution at the default threshold of 0.5. The final local result uses a calibrated threshold of **0.56**.

## Main workflow

### A. Prepare data

1. place raw images and raw segmentation maps under `data/raw/`
2. preprocess them into `data/processed/`
3. write `train.txt` and `val.txt` into `data/splits/`
4. export weighted HDF5 files with the scripts under `scripts/`

### B. Train the model

Use Caffe to train with the solver under `src/`.

Typical command:

```bash
cd caffe
./.build_release/tools/caffe.bin train \
  --solver="/path/to/src/solver.prototxt"
```

### C. Run inference

Use the OpenCV DNN scripts under `scripts/`.

Typical command:

```bash
python scripts/infer_single_cv2_weighted.py \
  --image data/processed/images/01_t049.tif \
  --gt data/processed/segmentation_maps/01_t049.tif \
  --deploy src/deploy.prototxt \
  --weights outputs/checkpoints/unet_weighted_iter_1500.caffemodel \
  --outdir outputs/inference_final \
  --threshold 0.56
```

### D. Evaluate on the validation split

```bash
python scripts/eval_val_cv2_weighted.py \
  --split data/splits/val.txt \
  --image_dir data/processed/images \
  --gt_dir data/processed/segmentation_maps \
  --deploy src/deploy.prototxt \
  --weights outputs/checkpoints/unet_weighted_iter_1500.caffemodel \
  --threshold 0.56
```

## Reports and figures

The report source and generated figures are stored under `reports/`.

Representative qualitative figures include:

- `reports/figures/01_t049_input.png`
- `reports/figures/01_t049_gt.png`
- `reports/figures/01_t049_pred.png`
- `reports/figures/training_curve.png`

## Requirements

The Python dependencies used by scripts are listed in `requirements.txt`.

Note that **Caffe itself is not installed through `requirements.txt`**. It must be built separately in the local environment.

## Limitations

This reproduction has several practical limitations:

- CPU-only training and inference constraints
- very small local train/validation split
- simplified local protocol instead of the original challenge benchmark
- threshold-sensitive inference behavior
- incomplete reproduction of the full original augmentation and benchmark pipeline

## Reference

- Ronneberger, O., Fischer, P., and Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
