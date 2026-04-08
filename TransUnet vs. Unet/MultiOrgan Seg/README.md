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
- model architecture
- segmentation target setting
- training configuration
- evaluation metrics
- qualitative prediction behavior

## Directory Role
This directory serves as the main workspace for the multi-organ segmentation study, including code, configs, scripts, logs, and related resources that support the comparison experiments.

## Research Purpose
This part of the project is important for understanding:
1. whether TransUNet brings clear benefits over U-Net in complex anatomical segmentation
2. which types of organs benefit more from global-context modeling
3. where transformer-based designs may show weaknesses such as optimization difficulty or weaker generalization

## Notes
- This repository version preserves the experiment workspace structure.
- Large datasets or heavy intermediate outputs should be managed separately when necessary.
- Future updates can further document dataset source, exact split, model variants, and final comparison tables.
