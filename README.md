# Partial Diffusion Suffices: Solving General Inverse Problems via Score Evolution

## Overview

This project/repository contains our work. This version is primarily intended for peer review.

## Core Algorithm

The core contribution of our work, the main implementation of which is located in the following directory:

* `util/algo/pdse/`

## Code and Model Release Plan

We are committed to promoting openness and reproducibility in our research field.

The complete, runnable code for this project, including all necessary scripts, utilities, and the fine-tuned model checkpoints specifically for MRI and CT tasks, will be made publicly available on GitHub.

Thank you for your understanding and patience.

## Code Structure

The project's code is organized as follows:

* `configs/`: Contains YAML configuration files for experiments.
* `guided_diffusion/`: Core diffusion model components (often based on or adapted from existing guided-diffusion repositories).
* `util/`: Contains various utility functions:
  * `algo/`: Implementations of various algorithms.
    * `pdse/`: The core algorithm implementation is located here.
    * `dmplug.py`: DMPlug algorithm.
    * `mpgd.py`: MPGD  algorithms.
    * `dps.py`: DPS algorithms.
    * `utils.py`: Helper functions related to algorithms (e.g., metrics calculation, logging, early stopping strategies).
  * `img_utils.py`: Image processing-related utilities.
  * `logger.py`: Logging setup.
  * `tools.py`:Other general tools.
* `outputs/`: Default directory for saving experimental results, logs, and images.
* `image_train.py`:  Main script for general model MRI training.
* `sample_condition.py` : Main script(s) for general image generation/reconstruction/testing.


​                                                                                                                                                                      
