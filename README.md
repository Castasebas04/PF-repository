# PreprocessingImages_FinalProject

This repository contains the tools and scripts developed for the preprocessing stage of histological images from renal biopsies, as part of the final degree project focused on the detection and segmentation of tubular atrophy.

## Project Objective
The main purpose of this module is to clean, normalize, and prepare the medical image dataset to optimize the subsequent training of the Deep Learning model (U-Net).

## Main Features
* **Color Normalization:** Adjustment and standardization of histological stains to reduce variability between samples.
* **Resolution Adjustment and Resizing:** Adaptation of image sizes to the input format required by the neural network.
* **Patch Generation:** Division of high-resolution images (WSI) into smaller sub-images to facilitate processing.
* **Data Augmentation (Optional):** Rotations, flips, and contrast adjustments to enrich the training dataset.

## Prerequisites
To run the scripts in this repository, make sure you have Python installed along with the following main libraries:
* `opencv-python`
* `numpy`
* `pillow` (PIL)
* `scikit-image`

## Basic Usage
1. Clone this repository:
```bash
   git clone [https://github.com/Castasebas04/PF-repository.git](https://github.com/Castasebas04/PF-repository.git)
