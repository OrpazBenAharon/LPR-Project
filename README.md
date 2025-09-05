
# Generative Image Models for Enhancing License Plate Images Captured at Extreme Viewing Angles

## Overview
License plate (LP) images from non-dedicated cameras (e.g., security, ATM) often suffer from extreme geometric distortion and noise, rendering them unreadable and, beyond a certain point, irrecoverable. This project pinpoints the critical threshold where this information is permanently lost. To achieve this, we developed an end-to-end framework to synthetically generate distorted data and benchmark a diverse set of deep learning models (discriminative and generative). The result is a clear 'recoverability boundary' that defines the maximum viewing angles for successful image restoration.

Concept: Restoring a license plate from a synthetically distorted image that simulates an extreme angle capture.
## Synthetic Data with Realistic Distortions

The pipeline generates clean license plate images, applies 3D perspective warping to simulate different viewing angles, and adds realistic noise effects (blur, compression, brightness/contrast changes, color shifts, edge artifacts, and Gaussian noise). The images are then realigned, cropped, and downsampled.

<img width="953" height="110" alt="image" src="https://github.com/user-attachments/assets/cce3f683-348b-4ed2-b3d6-1ce0f4f041e5" />


## Advanced Angle Sampling

We created a custom probability density function (PDF) with emphasis on extreme angles and applied a 3D Sobol sequence to generate evenly distributed samples. The Sobol cube was divided into four regions and then mapped to the angle space according to the PDF.



<img width="2902" height="814" alt="image" src="https://github.com/user-attachments/assets/3b2407ec-5d24-4ff5-9fd0-b464a24ff991" />


## Models Investigated
We evaluated five distinct architectures representing several leading deep learning paradigms (CNNs, Transformers, GANs, and Diffusion Models) for image restoration.

| **Category** | **Model**    | **Key Adaptations for This Project** |**Orignal paper**|
| :-------- | :------- | :-------------------------------- |:-------------------------------- |
| Discriminative      | U-Net | Adapted from its original segmentation purpose for image-to-image regression. We modified it to handle 3-channel RGB images and incorporated batch normalization for stable training.| [U-Net](https://arxiv.org/abs/1505.04597)|
|  | U-Net Conditional| Extends the U-Net with FiLM (Feature-wise Linear Modulation) layers. A separate encoder processes the distorted input into a latent vector, which FiLM then uses to generate adaptive channel-wise scale and shift parameters at each stage of the network. This allows the model to dynamically adjust its features to suppress artifacts specific to each image. |[FiLM](https://arxiv.org/abs/1709.07871)|
|  | Restormer | The original high-resolution Transformer architecture was applied directly. In our implementation, the model is trained to predict the residual (the difference between the clean and distorted images), which is then added back to the input to produce the final restored image|[Restormer](https://arxiv.org/abs/2111.09881)|
| Generative | Pix2Pix GAN | We followed the original framework, pairing a U-Net-based generator with a PatchGAN discriminator. The discriminator evaluates local image patches to enforce high-frequency realism, and the training objective combines an adversarial loss with an L1 reconstruction loss for pixel-level accuracy. |[Pix2Pix GAN](https://arxiv.org/abs/1611.07004)|
|  | Diffusion-SR3 | Adapted from super-resolution to restoration by conditioning the reverse diffusion process on our distorted image (concatenated as a 6-channel input). For improved stability, we implemented v-prediction instead of the standard ε-prediction and used DDIM for faster, deterministic sampling.|[Diffusion-SR3](https://arxiv.org/abs/2104.07636)|

<img width="604" height="510" alt="image" src="https://github.com/user-attachments/assets/f9a65f7a-489b-4c2b-8bf3-e7ef62391ff9" />

## Results
- **Model evaluation** - Using plate-level OCR accuracy, Restormer achieved the best performance across the full angle grid, showing the highest and most stable recognition rates. In contrast, Diffusion-SR3 delivered the weakest results, with significantly lower OCR accuracy and less consistent restoration quality.

  <img width="2919" height="620" alt="image" src="https://github.com/user-attachments/assets/13faabc0-edec-401a-b80c-22f097df386f" />


- **Maximal recoverability boundary and rotation difficulty** - The recoverability boundary was defined by mapping all license plates with OCR accuracy above 90%. For each α, we took the maximal β, and for each β, we took the maximal α; the union of these limits defined the boundary for each model. Combining all model boundaries gave a maximal recoverability region covering ~93% of the angle space. From the same graph, it is clear that no restoration is possible when both α and β exceed 80°, and that extreme yaw (α) rotations are harder to recover than extreme pitch (β) rotations.
  
  <img width="1828" height="1000" alt="image" src="https://github.com/user-attachments/assets/f8358341-d430-4973-86d0-c948a838cebe" />

- **Reliability (F-score)** – Within the recoverability boundary, F-scores revealed “holes” where models failed to reconstruct despite meeting the angle limits. Discriminative models showed fewer failures, while generative models had more.
  
  <img width="598" height="392" alt="image" src="https://github.com/user-attachments/assets/d534e595-7a93-4739-a7f5-506d329a6fd3" />



- **PSNR–OCR correlation** – Across all models, PSNR and OCR accuracy were strongly correlated (R² ≈ 0.98), confirming PSNR as a reliable metric for guiding training and evaluation.



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

