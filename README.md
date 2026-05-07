# Dataset Setup

This project uses the **CelebA dataset** for synthetic face image generation.

The full dataset is **not included in this GitHub repository** because it is large and contains real face images. The dataset should be downloaded locally before running the preprocessing, training, or evaluation scripts.

---

## Dataset Used

**Dataset:** CelebA  
**Task:** Face image generation  
**Use in this project:** Training and comparing VAE, DCGAN, and diffusion models for synthetic face generation.

CelebA includes aligned celebrity face images and additional metadata such as facial attributes, bounding boxes, landmarks, and train/validation/test split information.

---

## Expected Local Folder Structure

After downloading and extracting CelebA, the local dataset should follow this structure:

```text
data/
  raw/
    celeba/
      img_align_celeba/
        000001.jpg
        000002.jpg
        000003.jpg
        ...
      list_attr_celeba.txt
      list_bbox_celeba.txt
      list_eval_partition.txt
      list_landmarks_align_celeba.txt
```

---

## Project Structure

```text
celeba-generative-comparison/
│
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── samples/
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_vae_baseline.ipynb
│   ├── 04_dcgan_baseline.ipynb
│   ├── 05_diffusion_model.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_app_demo_test.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
│
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   └── assets/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_vae.py
│   ├── train_dcgan.py
│   ├── train_diffusion.py
│   ├── generate_samples.py
│   └── evaluate_models.py
│
├── reports/
│   ├── figures/
│   └── results_summary.md
│
├── checkpoints/
├── outputs/
└── runs/
```