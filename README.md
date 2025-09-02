
# Generative Image Models for Enhancing License Plate Images Captured at Extreme Viewing Angles

This repository contains the official code for a final year project for the Department of Electrical Engineering at  Afeka - The Academic College of Engineering in Tel Aviv, focused on reconstructing distorted license plate images using deep learning models.



## Overview
License plate (LP) images from non-dedicated cameras (e.g., security, ATM) often suffer from extreme geometric distortion and noise, rendering them unreadable and, beyond a certain point, irrecoverable. This project pinpoints the critical threshold where this information is permanently lost. To achieve this, we developed an end-to-end framework to synthetically generate distorted data and benchmark a diverse set of deep learning models (discriminative and generative). The result is a clear 'recoverability boundary' that defines the maximum viewing angles for successful image restoration.

Concept: Restoring a license plate from a synthetically distorted image that simulates an extreme angle capture.
## Key Features

- **Synthetic Data with Realistic Distortions:** Creates clean/distorted image pairs by applying 3D perspective warps and a multi-stage noise simulation pipeline.
- **Advanced Angle Sampling:** Uses Sobol sequences to ensure dense and uniform coverage of extreme viewing angles.
- **Diverse Model Benchmark:** Systematically compares leading deep learning paradigms (CNNs, Transformers, GANs, Diffusion).
- **Novel Evaluation Metrics:** Introduces Boundary-AUC and a Reliability Score (F) to precisely quantify recoverability limits.

<img width="953" height="110" alt="image" src="https://github.com/user-attachments/assets/cce3f683-348b-4ed2-b3d6-1ce0f4f041e5" />
<img width="937" height="477" alt="image" src="https://github.com/user-attachments/assets/34eae85b-ccab-4944-8191-0f297bb89f1b" />


## Models Investigated
We evaluated five distinct architectures representing several leading deep learning paradigms (CNNs, Transformers, GANs, and Diffusion Models) for image restoration.

| **Category** | **Model**    | **Description**                      |
| :-------- | :------- | :-------------------------------- |
| Discriminative      | U-Net | A robust CNN-based encoder-decoder, serving as our baseline. |
|  | U-Net Conditional| An enhanced U-Net using FiLM layers to condition features on input distortion. |
|  | Restormer | An efficient Transformer-based model for high-resolution image restoration|
| Generative | Pix2Pix GAN | A conditional GAN that learns an image-to-image mapping adversarially. |
|  | Diffusion-SR3 | A denoising diffusion probabilistic model adapted for conditional restoration.|

<img width="604" height="510" alt="image" src="https://github.com/user-attachments/assets/f9a65f7a-489b-4c2b-8bf3-e7ef62391ff9" />

## Key Results
- **Discriminative Models Outperform Generative Ones:** Discriminative architectures (e.g., Restormer) consistently delivered superior restoration quality, while generative models often struggled with severe distortions and "hallucinated" incorrect digits.
- **A "Maximal Recoverability Boundary" Was Identified:** We found that ~93.4% of the angle space is recoverable. Beyond ~80° on both axes, information is permanently lost. Furthermore, yaw rotation (α angle) is consistently harder to restore than pitch rotation (β angle).
- **PSNR is a Reliable Proxy for OCR Accuracy:** PSNR demonstrates a strong linear correlation (R^2~0.98) with final OCR accuracy, making it a robust metric for guiding the training process.
<img width="1218" height="525" alt="image" src="https://github.com/user-attachments/assets/d9e11cfe-ffb5-4487-bdaf-3a9e083d95b4" />


## Getting Started
Prerequisites
-	Anaconda or Miniconda
-	NVIDIA GPU with CUDA support
-	Tesseract OCR Engine

## Installation & Setup

**Clone the Repository Using Git**

```
bash
  git clone [https://github.com/aiigoradam/LPR-Project.git](https://github.com/aiigoradam/LPR-Project.git) cd LPR-Project
```
(Alternatively, download and unzip the ZIP file and navigate into the LPR-Project-main directory)


**Activate the Environment**
```
bash
  conda env create -f environment.yml
  conda activate lpr2-env
```

## Usage and Workflow

**The project is designed as a sequential pipeline. Follow the steps below to generate data, train the models, and evaluate the results.**

1. **Data Generation** (datasets A, B and C, and full grid evaluation)

```
bush
  python scripts/lp_processing.py
```
This will populate the *data/* directory with the necessary datasets, including clean/distorted image pairs and their corresponding metadata.json files.

2. **Model Training** - train the desired model on a specific dataset and experiment name for MLflow

- *model_name: unet_base, unet_conditional, restormer, pix2pix, diffusion_sr3*
- *data-dir: data/A, data/B, data/C*
```
bush
  Python src/train_{model_name}.py  #default parameters 
  # Example: Training Restormer on Dataset B
  # python src/train_restormer.py --data-dir data/B --experiment-name "LPR_Restormer_B" 
```
The trained model and its associated run data (metrics, artifacts) will be saved in a local *mlruns* directory. You can view the results using the MLflow UI 

3. **Full-Scale Evaluation** (Inference & Metrics)

After training, the *run_all.py* script automates the entire evaluation process. It loads the best models logged in MLflow, runs inference on the *full_grid* dataset, and computes all performance metrics.

4. **Analysis and Visualization**

Open *results.ipynb* Jupyter Notebook to load the CSV files and generate the heatmaps, plots, and tables from our final report.



## Contributors
- Igor Adamenko
- Orpaz Ben Aharon
- Mentor: Dr. Sasha Apartsin

