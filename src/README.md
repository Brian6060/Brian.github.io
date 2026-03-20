# `src/` Directory Guide

This directory stores the core **Caffe-side experiment configuration files** used in the U-Net reproduction.

It mainly contains:

- network definitions in `.prototxt`
- solver definitions
- deployment configuration
- auxiliary experiment configuration files created during debugging

## Purpose of this directory

The `src/` folder is the configuration center of the experiment.

It defines:

- how the network is built
- how training data are connected to the graph
- which loss function is used
- how optimization is performed
- how deployment and inference are configured

This directory does **not** contain the Caffe source code itself. The Caffe framework is stored under the top-level `caffe/` directory.

## Main files

### `train_val.prototxt`

This is the main training and validation network definition.

It typically includes:

- HDF5 input layers
- encoder and decoder blocks of U-Net
- skip connections
- final `score` layer
- the training loss layer

In the final weighted setting, the input pipeline reads three blobs:

- `data`
- `label`
- `weight`

and uses the custom layer:

- `WeightedSoftmaxWithLoss`

### `deploy.prototxt`

This file defines the inference-time graph.

It should contain:

- the network structure only
- no HDF5 input layers
- no `label`
- no loss layer

The deploy graph is used together with a trained `.caffemodel` file for inference through OpenCV DNN.

### `solver.prototxt`

This file defines the optimizer and training schedule.

Typical contents include:

- learning rate
- learning rate policy
- momentum
- weight decay
- test interval
- max iterations
- snapshot interval
- checkpoint prefix

Different solver variants may appear during debugging and ablation.

### `solver_*_debug.prototxt`

These files, if present, are lightweight debug solvers used to verify that:

- the network graph is valid
- the custom loss layer is recognized
- HDF5 input can be loaded correctly
- the training loop runs at least one iteration without immediate failure

## Final training configuration used in the report

The final reported weighted model is based on:

- a two-channel output `score` layer
- a custom weighted pixel-wise loss
- HDF5 input with `data`, `label`, and `weight`
- a 1500-iteration training schedule
- the checkpoint:
  - `unet_weighted_iter_1500.caffemodel`

## Relationship with other directories

### `caffe/`

Contains the actual Caffe framework source tree and compiled binaries.

### `scripts/`

Contains helper scripts for:

- HDF5 export
- plotting training curves
- single-image inference
- validation evaluation
- checkpoint and threshold search

### `data/`

Stores raw data, processed data, split files, and HDF5 exports used by the graphs defined here.

### `outputs/`

Stores training logs, checkpoints, and inference outputs produced from the configurations in this folder.

## Typical usage

### Train

```bash
cd ../caffe
./.build_release/tools/caffe.bin train \
  --solver="/path/to/src/solver.prototxt"
```

### Deploy and infer

```bash
python ../scripts/infer_single_cv2_weighted.py \
  --image /path/to/image.tif \
  --gt /path/to/mask.tif \
  --deploy /path/to/src/deploy.prototxt \
  --weights /path/to/checkpoint.caffemodel \
  --outdir /path/to/output_dir \
  --threshold 0.56
```

## Notes

1. The files in this directory are tightly coupled to the local project paths. If the project is moved, some absolute paths inside `.prototxt` files may need to be updated.
2. The final inference pipeline does not rely on `pycaffe`, because `pycaffe` caused local runtime library conflicts on macOS.
3. If a training run fails, first check:
   - HDF5 list paths
   - number of HDF5 tops
   - final loss layer type
   - checkpoint prefix
   - deploy output channel count
