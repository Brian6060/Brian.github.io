# MultiOrgan Seg

## Overview
This directory contains the resources for the **multi-organ segmentation comparison experiments** in the **TransUNet vs. U-Net** project.

The main goal of this part is to compare different segmentation architectures under a multi-organ setting, with emphasis on the performance difference between convolution-based and transformer-based designs.

## Experiment Goal
The overall objective is to evaluate how **U-Net** and **TransUNet-related variants** behave on multi-organ segmentation tasks, especially in terms of:
- segmentation accuracy
- organ-wise performance differences
- generalization behavior
- training stability
- sensitivity to task complexity and organ scale

## Possible Comparison Dimensions
This experiment directory is intended to support comparisons across:
- **Model architecture**
  - U-Net
  - TransUNet
  - other controlled variants if included
- **Segmentation target setting**
  - multi-class segmentation
  - binary or organ-specific reformulations if applicable
- **Training configuration**
  - input size
  - loss design
  - optimizer and scheduler settings
  - augmentation strategy
- **Evaluation**
  - Dice
  - IoU
  - organ-wise metrics
  - qualitative prediction results

## Directory Role
This directory serves as the main workspace for the multi-organ segmentation study, including code, configs, scripts, logs, or related resources that support the comparison experiments.

## Research Purpose
This part of the project is important for understanding:
1. whether TransUNet brings clear benefits over U-Net in complex anatomical segmentation
2. which types of organs benefit more from global-context modeling
3. where transformer-based designs may show weaknesses such as optimization difficulty or weaker generalization

## Notes
- This repository version preserves the experiment workspace structure.
- Large datasets or heavy intermediate outputs should be managed separately when necessary.
- Future updates can further document:
  - dataset source
  - exact train/val/test split
  - model variants
  - final quantitative comparison tables
