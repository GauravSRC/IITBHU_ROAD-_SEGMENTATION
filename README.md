# Road Segmentation Project

This repository contains the implementation of a road segmentation pipeline using a MobileNetV3-Large backbone with custom segmentation heads. It is designed to train on paired RGB images and annotated masks, and to perform real-time segmentation on images and videos.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dependencies](#dependencies)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training Procedure](#training-procedure)
7. [Evaluation Metrics and Results](#evaluation-metrics-and-results)
8. [Inference (Image & Video)](#inference-image--video)
9. [Usage Instructions](#usage-instructions)
10. [Saving and Loading Models](#saving-and-loading-models)
11. [License](#license)

---

## Project Overview

This project implements an end-to-end road segmentation pipeline:

* **Data Loading & Augmentation**: Reads JPEG images and PNG masks, applies spatial and photometric augmentations.
* **Model**: Adapts a pretrained MobileNetV3-Large model by adding custom `second_last` and `last` convolutional heads for binary segmentation.
* **Training**: Fine-tunes the last few layers along with select intermediate blocks (conv\_up2, conv\_up3, block4) using a BCEWithLogitsLoss objective.
* **Evaluation**: Computes average Intersection-over-Union (IoU) and tracks train/validation losses over epochs.
* **Inference**: Provides scripts to segment single images and entire video streams, with optional post-processing (erosion, contour filtering).

This README documents all steps to reproduce the experiments, run inference, and extend the code to new datasets.

## Repository Structure

```
├── data/
│   ├── train3/1/            # Original input folder with paired images and masks
│   ├── jpeg_images/         # Copied and renamed JPEG images
│   └── png_masks/           # Copied and renamed PNG masks
│
├── saved_models/            # Checkpoints for each training epoch
│   ├── last_layer+aug1.pth
│   └── last_layer+aug25.pth
│
├── notebooks/               # Optional Kaggle notebook files
├── src/                     # Source code (can be refactored from notebook)
│   ├── dataset.py           # Dataset_Builder implementation
│   ├── model.py             # Model definition and modification
│   ├── train.py             # Training loop and metrics logging
│   ├── inference_image.py   # Single-image segmentation script
│   ├── inference_video.py   # Video segmentation + overlay
│   └── postprocess.py       # Erosion and contour filtering methods
│
├── README.md                # Project documentation (this file)
└── requirements.txt         # Python dependencies
```

## Dependencies

Create a Python 3 environment and install the required packages:

```bash
pip install torch torchvision
pip install fastseg
pip install geffnet
pip install albumentations
pip install torchinfo
pip install tqdm
pip install opencv-python-headless
pip install pillow
```

Alternatively, install from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Original Data Directory**: Place the raw dataset in `data/train3/1/`, containing alternating image (`.jpeg`/`.jpg`) and mask (`.png`) files.

2. **Copy & Rename**: The script in the notebook copies images and masks into separate folders `jpeg_images/` and `png_masks/`, renaming them as `image_1.jpeg`, `mask_1.png`, etc.

   ```python
   python src/data_prep.py \
     --input-dir data/train3/1/ \
     --output-img-dir data/jpeg_images/ \
     --output-mask-dir data/png_masks/
   ```

3. **Train/Validation Split**: 90% of the dataset is used for training, and 10% for validation via PyTorch’s `random_split`.

## Model Architecture

* **Backbone**: `fastseg.MobileV3Large.from_pretrained()` with LRASPP head removed.
* **Custom Heads**:

  * `second_last`: 256 → 128 → 64 channels (1×1 Convs + ReLU + BatchNorm)
  * `last`: 128 → 1 channel through a cascade of 1×1 convolutions, ReLU, and BatchNorm layers.
* **Freezing Strategy**:

  1. Freeze all original backbone parameters.
  2. Unfreeze `second_last` and `last` heads.
  3. Unfreeze intermediate upsampling layers `conv_up2`, `conv_up3`.
  4. Unfreeze entire `block4` in the trunk for deeper fine-tuning.

Refer to `src/model.py` for full definitions.

## Training Procedure

1. **Hyperparameters**:

   * Epochs: 25
   * Batch size: 8
   * Optimizer: `AdamW` on all unfrozen parameters
   * Loss: `BCEWithLogitsLoss`
   * Learning rate: default (you may adjust via CLI)
2. **Augmentations**:

   * **Spatial**: Random rotation (±180°)
   * **Photometric**: Color jitter, Gaussian blur, motion blur, sun flare
3. **Metrics**:

   * Training/Validation loss per epoch
   * **IoU**: Binary Intersection-over-Union computed per batch and averaged

Launch training with:

```bash
python src/train.py \
  --img-dir data/jpeg_images/ \
  --mask-dir data/png_masks/ \
  --epochs 25 \
  --batch-size 8 \
  --save-dir saved_models/
```

Training logs will display progress bars, batch losses, epoch averages, and IoU scores.

## Evaluation Metrics and Results

* **Final Train Loss**: 0.1135
* **Final Validation Loss**: 0.0009
* **Peak Avg IoU**: 0.9008

Plot example training curves (`src/train.py` can save loss/IoU logs for visualization).

Sample Model Output on Test Image:

![Sample Segmentation](./assets/sample_prediction.png)

Segmentation mask shows accurate delineation of road regions, even under complex lighting.

## Inference (Image & Video)

### Single Image

Use `inference_image.py` to segment an individual image:

```bash
python src/inference_image.py \
  --model-path saved_models/last_layer+aug25.pth \
  --image-path test-data/test_image.jpg \
  --output-path output/image_result.png
```

### Video Segmentation

1. **Basic Overlay**:

```bash
python src/inference_video.py \
  --model-path saved_models/last_layer+aug25.pth \
  --input-video test-video-3/test_video.mp4 \
  --output-video output_video.mp4
```

2. **Post-Processing (Erosion + Filtering)**:

```bash
python src/postprocess.py \
  --input-video output_video.mp4 \
  --output-video output_video_erosion.mp4 \
  --min-area 500 \
  --kernel-size 5
```

This pipeline applies morphological erosion and removes small contours to reduce noise, then overlays the filtered mask in red.

## Usage Instructions

1. **Prepare Data** as described above.
2. **Train** the model or download a pretrained checkpoint.
3. **Run Inference** on new images or videos.
4. **Adjust Hyperparameters** in the training and inference scripts to suit your dataset.

All scripts accept additional CLI arguments; run `python script.py --help` for detailed options.

## Saving and Loading Models

* Models are saved as PyTorch state dictionaries along with optimizer state.

* To resume training or run inference, load via:

  ```python
  checkpoint = torch.load('saved_models/last_layer+aug25.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  ```

* For inference only, `optimizer` state can be ignored.

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
