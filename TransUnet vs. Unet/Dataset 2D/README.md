# Dataset 2D

## Overview
This directory contains the 2D dataset-related resources used in the **TransUNet vs. U-Net** project.

At the current stage, the repository only includes the **DataProcessor** module rather than the full dataset itself. The purpose is to preserve the preprocessing and dataset-construction logic while avoiding direct upload of large raw or processed data files.

## Included Content
- `DataProcessor/`: scripts and utilities for preparing, organizing, converting, or preprocessing the 2D dataset.

## Why the full dataset is not included
The original dataset files are excluded from the repository for storage and transfer efficiency. This repository focuses on:
- preprocessing pipeline preservation
- reproducibility of dataset preparation logic
- project structure management

## Intended Use
Use the files in `DataProcessor/` to:
1. inspect dataset preprocessing logic
2. reproduce the 2D data preparation pipeline
3. support downstream segmentation experiments comparing U-Net and TransUNet

## Notes
- Raw datasets and large generated files are not included in this repository.
- Dataset sources and preprocessing requirements can be documented separately in future updates.
