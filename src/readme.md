# OculoNet: Dual-Stage Glaucoma Detection Pipeline 👁️

A production-ready, dual-stage deep learning pipeline for automated Glaucoma diagnosis from fundus imagery. 

This project transitions a complex diagnostic workflow from research notebooks into a modular, local codebase. It utilizes a **TransUNet** for precise Optic Disc segmentation, followed by a dynamically cropped evaluation via a fine-tuned **DenseNet121** classifier.

##  Clinical Objective & Performance

In medical screening, false negatives (missing a positive Glaucoma diagnosis) carry severe consequences, potentially leading to irreversible vision loss. 

To meet strict clinical requirements:
- **Weighted Loss Fine-Tuning:** The DenseNet classifier was fine-tuned for 3 additional epochs utilizing a custom weighted loss (`pos_weight = 1.6`) to heavily penalize false negatives.
- **Calibrated Thresholding:** The decision boundary was calibrated from the standard `0.50` to **`0.42`**, guaranteeing a **$\ge 90\%$ Recall** on the test set.
- **Final ROC-AUC:** `0.9417`

## System Architecture

The inference pipeline operates in two distinct phases to mimic human ophthalmologic evaluation:

1. **Phase 1: Optic Disc Localization (TransUNet)**
   - **Role:** Analyzes the full fundus image to predict the exact location of the Optic Disc.
   - **Mechanism:** Uses a Vision Transformer + U-Net hybrid architecture (15 epochs) to generate a binary mask. The system dynamically calculates a bounding box around the predicted disc and extracts a localized Region of Interest (ROI).

2. **Phase 2: Clinical Classification (DenseNet121)**
   - **Role:** Analyzes the extracted optic nerve head for structural damage.
   - **Mechanism:** A fine-tuned DenseNet121 backbone extracts deep features from the localized crop, outputting a probability score mapped to the calibrated `0.42` clinical threshold.

## 📂 Repository Structure

```text
oculonet-glaucoma-pipeline/
├── models/
│   ├── densenet.py                        # DenseNet121 Architecture
│   ├── transunet.py                       # TransUNet Architecture
│   ├── transunetsegmentationmodel(1).pth # [Download from Releases]
│   └── finetuned_densenet_glaucoma.pth    # [Download from Releases]
├── src/
│   ├── extract_roi.py                     # Dynamic OpenCV cropping logic
│   └── inference.py                       # Master execution script
├── notebooks/                             # Historical training notebooks
├── .gitignore
└── README.md

1. Clone the Repository
    git clone [https://github.com/fireflash27/oculonet-glaucoma-pipeline.git](https://github.com/fireflash27/oculonet-glaucoma-pipeline.git)
    cd oculonet-glaucoma-pipeline
2. Install Dependencies
    pip install torch torchvision opencv-python albumentations numpy self-attention-cv
3. Download Model Weights
    Because the model weights exceed GitHub's standard file limits, they are hosted in GitHub Releases.

    Navigate to the Releases tab of this repository.

    Download both .pth files:

    transunetsegmentationmodel(1).pth

    finetuned_densenet_glaucoma.pth

    Place both files directly into the models/ directory.
4. Running Inference
    To run a diagnosis on a new fundus image, place your image in the root directory (e.g., test_eye.jpg) and update the SAMPLE_IMAGE path in src/inference.py.

    Execute the pipeline from the root directory:

    Bash
    python src/inference.py
5. Expected Output
    [*] Initializing pipeline on: CUDA
    [*] Building model architectures...
    [*] Loading trained weights...
    [*] Reading image: test_eye.jpg
    [*] Phase 1: Locating Optic Disc (TransUNet)...
    [*] Phase 2: Analyzing nerve damage (DenseNet)...

    ==================================================
     CLINICAL DIAGNOSIS REPORT
    ==================================================
    File             : test_eye.jpg
    Class 1 Prob.    : 0.9093 (90.9%)
    Active Threshold : 0.4200
    Final Diagnosis  : Glaucoma Detected (1)
    ==================================================
6. Labels
    Class 0: Healthy

    Class 1: Glaucoma Detected

7. Author:  Developed by Rishi Kumar
            Linkedin: www.linkedin.com/in/rishi-kumar-202287387
