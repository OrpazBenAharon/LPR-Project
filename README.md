
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

| **Category** | **Model**    | **Key Adaptations for This Project** |**Original paper**|
| :-------- | :------- | :-------------------------------- |:-------------------------------- |
| Discriminative      | U-Net | Adapted from its original segmentation purpose for image-to-image regression. We modified it to handle 3-channel RGB images and incorporated batch normalization for stable training.| [U-Net](https://arxiv.org/abs/1505.04597)|
|  | U-Net Conditional| Extends the U-Net with FiLM (Feature-wise Linear Modulation) layers. A separate encoder processes the distorted input into a latent vector, which FiLM then uses to generate adaptive channel-wise scale and shift parameters at each stage of the network. This allows the model to dynamically adjust its features to suppress artifacts specific to each image. |[FiLM](https://arxiv.org/abs/1709.07871)|
|  | Restormer | The [original](https://github.com/swz30/Restormer) high-resolution Transformer architecture was applied directly. The model is trained to predict the residual (the difference between the clean and distorted images), which is then added back to the input to produce the final restored image|[Restormer](https://arxiv.org/abs/2111.09881)|
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

## Repository Structure
```
LPR-Project/
├── data/                 # Stores all generated datasets
├── models/               # Model architecture definitions
├── scripts/              # Standalone scripts for processing, inference, and evaluation
├── src/                  # Model training scripts and shared utilities
├── environment.yml       # Conda environment definition
├── run_all.py            # Master script to automate the evaluation pipeline
├── results.ipynb         # Jupyter notebook for final analysis and visualization
└── README.md
```

## Getting Started
Prerequisites
-	Anaconda or Miniconda
-	Python 3.9+
-	NVIDIA GPU with CUDA support
-	Tesseract OCR Engine


## Installation & Setup

**Clone the Repository Using Git**

Open your terminal and clone the project repository:
```
bash
  git clone [https://github.com/aiigoradam/LPR-Project.git](https://github.com/aiigoradam/LPR-Project.git) cd LPR-Project
```
(Alternatively, download and unzip the ZIP file and navigate into the LPR-Project-main directory)


**Activate the Environment**

All code is run in a dedicated Anaconda environment, which includes PyTorch, OpenCV, Tesseract OCR, MLflow, and other required libraries. Activating this environment ensures all dependencies are available for every script.
```
bash
  conda env create -f environment.yml
  conda activate lpr2-env
```

## Usage and Workflow

The project is designed as a sequential pipeline. Follow the steps below to generate data, train the models, and evaluate the results.

**Step 1. Data Generation**
```
bush
  python scripts/lp_processing.py
```
This script runs the full dataset generation pipeline, from creating clean plates to saving them in the *data/* folder. In the main, we can set all the related parameters of the clean plates’ dimensions, font, the sampling PDF,
the number of samples, etc. By default, three datasets are created and saved in *data/{dataset}/split/* folders as paired *original_{index}.png* and *distorted_{index}.png* images. Each split includes a *metadata.json* file that records the plate text, 
distortion angles, and bounding boxes for every digit, linking each distorted image with its clean original.

**Step 2. Model Training** - train the desired model on a specific dataset and experiment name for MLflow

*NOTE: To process the results correctly later on, all models that are trained on the same dataset need to be saved under the **same experiment name**.*
```
bush
  # command-line arguments: --model_name, --data-dir, ----experiment-name
  # Example: Training Restormer on Dataset B
    python src/train_restormer.py --data-dir data/B --experiment-name "LPR_B" 
```
- **Data loading:** The custom LicensePlateDataset class reads all image pairs into memory once and applies any necessary transforms (tensor conversion, normalization). PyTorch DataLoaders turn the dataset class into an iterable over batches, shuffle batches (only for the training set), and deliver them to the model in the training loop.
- **Model setup:** The selected model is moved to the GPU. The script defines a loss (e.g., MSE) and an optimizer (e.g., AdamW). It defines a learning rate scheduler (e.g., CosineAnnealingLR). A fixed seed fixes data order and weight initialization.
- **Training loop:** For U-Net and GANs, each epoch shuffles batches, runs forward pass, computes loss, back-propagates, updates weights, then runs a no-gradient validation pass. Diffusion follows the same cycle, but first adds noise to each input and predicts that noise across scheduled timesteps. After validation, the script logs loss, MSE, SSIM, and PSNR to MLflow and keeps the model state with the best validation SSIM.
- **Post-training:** After the final epoch or step, the best model weights are loaded. Evaluated on the unseen test set. Test MSE, SSIM, and PSNR are logged. The trained model is logged with an environment file for dependencies.
- **MLflow tracking:** The script starts an MLflow run. It logs hyperparameters and copies the training script and model file as artifacts. Metrics and sample images are logged during training. Those artifacts are stored locally and can be accessed through the MLflow UI.
  ```
  
  ```
**Step 3. Full-Scale Evaluation** - Inference

```
bush
  # command-line arguments: --experiment-name, --models, --steps, --batch-size
  # Example: Inference all models trained on dataset B
    python scripts/run_inference.py --experiment-name "LPR_B" --models "unet_base,unet_conditional,restormer,pix2pix,diffusion_sr3" --steps 1000 --batch-size 16
```
The *run_inference.py* script loads trained models from MLflow and applies them to the data/full_grid/ set. It loads each model under the chosen dataset name, moves the model to the GPU, and enables evaluation mode. For U-Net, U-Net Conditional, Restormer, and Pix2Pix, 
each distorted image is fed through the network, and the restored output is saved in results/{dataset}/{model}/. The diffusion model performs the scheduled denoising loop before saving the image. The script measures the average inference time per 
image and writes it to inference_times.csv. Command line flags control parameters such as model names, batch size, and diffusion sampling steps.

**Step 4. Full-Scale Evaluation** - Metrics

```
bush
  # command-line arguments: --experiment-name, --models
  # Example: Compute metrics for all models trained on dataset B
    python scripts/compute_metrics.py --experiment-name "LPR_B" --models "unet_base,unet_conditional,restormer,pix2pix,diffusion_sr3" 
```
The *compute_metrics.py* script compares each reconstructed image with its clean original. It looks up the plate text, angle pair, and digit boxes in metadata.json. For each image, it reports plate-level and digit-level metrics. It runs Tesseract in digit mode, 
extracts the digit patch, preprocesses it, and applies recognition. If recognition fails or is incomplete, the script tries alternative preprocessing or attempts to identify the digit from a full plate. All values (raw data) are saved 
to a CSV at results/{dataset}/{model_name}.csv.

**2+3+4 STEPS AT ONCE**

 The *run_all.py* script automates the workflow for any dataset. It activates the project’s environment. It runs the training scripts for each model in sequence. After training, it calls *run_inference.py*. It then calls *compute_metrics.py*. 
 The script can be configured to enable or skip specific models and steps. It uses subprocess to launch each script in the correct order. **This allows running the entire pipeline with a single command once the data is available**.

## Checking, Validation & Figures
- *ocr_test.ipynb* – In this notebook, we test OCRs. We check that the OCR and preprocessing steps can read digits from distorted plates. Different preprocessing options are tried. The final approach is used in the main pipeline. This notebook confirms that OCR will work before it is used in *compute_metrics.py*.
- *report_artifacts.ipynb* – This notebook makes the figures and sample images used in the project report. Here, we break down our code and plot intermediate results and other relevant illustrations.
- *sampling.ipynb* – In this notebook, we test the sampling distribution for rotation angles. We select parameters and plot PDFs and how rotation angles are sampled from the PDF for each dataset. We compare sampling methods. This helps us visually verify correctness of our distribution and sampling. We use those figures in the report. It is an extension to *report_artifacts.ipynb*.
- *results.ipynb* – In this notebook, we load the CSV files (raw data) from *compute_metrics.py* for each model and dataset as pandas dataframes. We compute means over all samples for each angle pair to get a general overview of each model. We transform the dataframes to draw statistics and make different plots, such as heatmaps, bar charts, scatter plots, and tables. This notebook is used to prepare the main summary figures and tables for the final report.


## Contributors
- Igor Adamenko
- Orpaz Ben Aharon
- Mentor: Dr. Sasha Apartsin

