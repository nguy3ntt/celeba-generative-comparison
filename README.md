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
