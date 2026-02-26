"""Metric Visualizer Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from anomalib.models.components import AnomalyModule

from .visualizer_base import BaseVisualizerCallback


class MetricVisualizerCallback(BaseVisualizerCallback):
    """Callback that visualizes the metric results of a model by plotting the corresponding curves.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def on_test_end(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Log images of the metrics contained in pl_module.

        In order to also plot custom metrics, they need to have implemented a `generate_figure` function that returns
        tuple[matplotlib.figure.Figure, str].

        Args:
            trainer (pl.Trainer): pytorch lightning trainer.
            pl_module (AnomalyModule): pytorch lightning module.
        """

        if self.save_images or self.log_images:
            for metrics in (pl_module.image_metrics, pl_module.pixel_metrics):
                for metric in metrics.values():
                    # `generate_figure` needs to be defined for every metric that should be plotted automatically
                    if hasattr(metric, "generate_figure"):
                        fig, log_name = metric.generate_figure()
                        file_name = f"{metrics.prefix}{log_name}"
                        if self.log_images:
                            self._add_to_logger(fig, pl_module, trainer, file_name)

                        if self.save_images:
                            fig.canvas.draw()
                            # convert figure to np.ndarray for saving via visualizer
                            fig.canvas.draw()
                            w, h = fig.canvas.get_width_height()
                            if hasattr(fig.canvas, 'buffer_rgba'):
                                buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
                                img = buf.reshape(h, w, 4)[..., :3]
                            elif hasattr(fig.canvas, 'tostring_argb'):
                                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                                argb = buf.reshape(h, w, 4)
                                img = argb[..., 1:]
                            else:
                                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                                img = buf.reshape(h, w, 3)
                            self.visualizer.save(Path(self.image_save_path.joinpath(f"{file_name}.png")), img)
                        plt.close(fig)
        super().on_test_end(trainer, pl_module)
