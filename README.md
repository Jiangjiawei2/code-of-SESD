# Score Evolution Guided Shortcut Diffusion for General Inverse Problems

## Overview

This project/repository contains our work. This version is primarily intended for peer review.

## Core Algorithm

The core contribution of our work, the main implementation of which is located in the following directory:

* `util/algo/sesd/`

## Code and Model Release Plan

We are committed to promoting openness and reproducibility in our research field.

The complete, runnable code for this project, including all necessary scripts, utilities, and the fine-tuned model checkpoints specifically for MRI (Magnetic Resonance Imaging) tasks, will be made publicly available on a code hosting platform (e.g., GitHub) upon the formal conclusion of the peer-review process for the associated academic publication.

At that time, we will provide clear instructions and resources to ensure that the research community can smoothly access and run the code, and reproduce our experimental results.

Thank you for your understanding and patience.

## Code Structure

The project's code is organized as follows:

* `configs/`: Contains YAML configuration files for experiments.
* `guided_diffusion/`: Core diffusion model components (often based on or adapted from existing guided-diffusion repositories).
* `util/`: Contains various utility functions:
  * `algo/`: Implementations of various algorithms.
    * `sesd/`: The core algorithm implementation is located here.
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
