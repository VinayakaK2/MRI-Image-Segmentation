# Robustness of a Lightweight 2D U-Net for Cardiac MRI Segmentation Under Controlled Image-Quality Degradations

## 1. Project Overview

This research mini-project investigates the performance degradation and failure modes of a lightweight 2D U-Net trained from scratch on the **ACDC (Automated Cardiac Diagnosis Challenge)** dataset when subjected to controlled image-quality corruptions. 

Beyond evaluating clean-image baseline segmentation, the project systematically assesses:
1. **Model Robustness:** Degradation curves under three controlled corruption families:
   - **Additive Noise:** Gaussian noise across mild, moderate, and severe levels.
   - **Spatial Blur:** Gaussian smoothing kernels modeling spatial detail loss.
   - **Contrast Reduction:** Intensity range attenuation modeling poor RF contrast.
2. **Mitigation Interventions:**
   - **Preprocessing Ablation:** Denoising and contrast enhancement filters applied before inference on corrupted inputs.
   - **Post-Processing Ablation:** Morphological filtering and largest connected component analysis to eliminate spurious island predictions.

---

## 2. Anatomical Classes & Task Definition

The segmentation pipeline solves a 4-class multi-class semantic segmentation task on short-axis cine cardiac MRI slices (specifically End-Diastole [ED] and End-Systole [ES] phases):
- **Class 0:** Background
- **Class 1:** Right Ventricular (RV) cavity
- **Class 2:** Left Ventricular Myocardium (MYO)
- **Class 3:** Left Ventricular (LV) cavity

---

## 3. Repository Structure

```text
.
├── checkpoints/              # Saved model weights (.pt / .pth)
├── configs/                  # Experiment configuration files (YAML)
│   └── baseline_unet.yaml    # Baseline lightweight U-Net configuration
├── data/
│   ├── raw/                  # Original ACDC NIfTI archives (.nii.gz)
│   └── processed/            # Extracted 2D slices and metadata
├── experiments/              # Run metadata and paired trial logs
├── reports/                  # Generated figures, tables, and metric summaries
├── scripts/                  # Command-line runners and utilities
│   └── check_env.py          # Environment diagnostics & hardware checker
├── src/                      # Core research package
│   ├── corruptions/          # Controlled image degradation modules
│   ├── data/                 # Dataset loader, slicing & patient-level split
│   ├── evaluation/           # Dice, IoU, HD95, and paired testing metrics
│   ├── models/               # Lightweight 2D U-Net architecture
│   ├── postprocessing/       # Component filtering & morphological ops
│   ├── preprocessing/        # Normalization, clipping, and restoration
│   ├── training/             # Loss functions (Dice+CE) & training loop
│   └── utils/                # Config parser, reproducibility, and logging
├── tests/                    # Unit and integration test suite
├── pyproject.toml            # Package configuration and tool settings
├── requirements.txt          # Python dependency specifications
└── README.md                 # Project documentation
```

---

## 4. Setup & Installation

### Local Environment (CPU / Local Testing)
```bash
# Optional: create a virtual environment
python -m venv .venv
# Activate:
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Compatibility
The training and evaluation scripts automatically detect CUDA devices when running on Google Colab or external GPU clusters:
```python
# In Colab notebook cell:
# !git clone <repo-url>
# %cd cardiac-mri-robustness
# !pip install -r requirements.txt
# !python scripts/check_env.py
```

---

## 5. Environment Diagnostics

Run the environment diagnostic tool to verify PyTorch version, hardware acceleration, library availability, and dataset configuration status:

```bash
python scripts/check_env.py
```

---

## 6. Running Tests

Run the test suite to verify configuration schema parsing, directory structure validation, and reproducibility utilities:

```bash
pytest -v tests/
```

---

## 7. Experimental Protocol & Rules

- **Strict Patient-Level Split:** Data is split by patient ID (not random slices) into 70% Train, 15% Validation, and 15% Test. Slices from the same patient never cross splits.
- **Frozen-Model Robustness:** The model trained on clean data is evaluated frozen against degraded test sets to ensure paired comparability across corruption severities.
- **Data Segregation:** Raw NIfTI volumes in `data/raw/` remain unmodified; processed 2D arrays reside strictly in `data/processed/`.
