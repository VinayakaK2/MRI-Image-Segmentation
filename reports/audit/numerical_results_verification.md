# Numerical Results Verification Report: Traceability & Independent Recalculation

**Project**: ACDC Cardiac MRI Segmentation — LightweightUNet2D  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Audit Date**: November 6, 2026  
**Auditor**: Senior Medical Imaging Research Scientist, ML Engineer & Statistical Auditor  
**Audit Scope**: End-to-end numerical traceability across Phase 3, Phase 4, Phase 5, and the CLAHE Forensic Audit  

---

## 1. Executive Summary

Every empirical numerical result reported across the project lifecycle was audited directly against raw machine-readable JSON and CSV outputs. Where prediction arrays and ground-truth masks were available, summary statistics and patient-level metrics were recomputed from first principles.

### Key Numerical Findings:
1. **100% Traceability**: All reported means, standard deviations, paired differences, and $p$-values match repository outputs to at least five significant digits.
2. **Zero Inaccuracies in Current Raw Artifacts**: Current raw data files (`reports/phase4_50epochs/robustness_results.json`, `experiments/phase5_validation_locked_mitigation/mitigation_results.json`, and `experiments/phase5_clahe_invariance_audit/patient_level_prediction_comparison.csv`) are 100% internally consistent and mathematically reproducible.
3. **Transcription Corrections in Historical Documentation**: Two historical transcription errors in prior markdown discussion documents were identified and reconciled:
   - 3-epoch severe contrast: Historical markdown discussions cited $0.5794 \pm 0.1017$; the actual raw ground truth in `reports/robustness_results.json` is **$0.0405 \pm 0.0446$**.
   - 3-epoch severe blur: Historical markdown discussions cited $0.6121 \pm 0.0825$; actual raw ground truth is **$0.6718 \pm 0.1147$**.
   - 3-epoch clean standard deviation: Historical markdown cited $0.0886$; actual clean standard deviation is **$0.1074$** ($0.0886$ was the severe contrast score of `patient009` mistakenly copied).

---

## 2. Phase 3: Baseline Model Training & Convergence Verification

### 2.1 Checkpoint Cryptographic Identity
Both model checkpoints were verified by computing independent SHA-256 hashes directly from binary bytes:

| Checkpoint Path | Architecture | Param Count | Epoch Selected | Training Origin | SHA-256 Hash | Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| `checkpoints/best_model_frozen.pt` | `LightweightUNet2D` | 1,963,860 | 3 | CPU (Local) | `e7bfa884bef0f4eafce30cd6948c8d60fc4e7c8c8ca2d0bb0e83f2977fc29c02` | **VERIFIED INTACT** |
| `checkpoints/50epochs/best_model_50epochs.pt` | `LightweightUNet2D` | 1,963,860 | 44 | GPU (Tesla T4) | `a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38` | **VERIFIED MATCH** |

*Verification*: The 50-epoch checkpoint was initialized from scratch (random seed `42`), not fine-tuned or resumed from the 3-epoch checkpoint.

### 2.2 Training Progression & Validation Milestones
Source: `reports/training_history_50epochs.json` (50 recorded epoch dictionaries).

- **Total Completed Epochs**: 50
- **Best Validation Epoch**: Epoch 44
- **Best Validation Foreground Dice**: `0.8871069795435059` ($\approx 0.8871$)
  - Val RV Dice (Epoch 44): `0.8601`
  - Val MYO Dice (Epoch 44): `0.8624`
  - Val LV Dice (Epoch 44): `0.9382`
- **Total Training Wall Clock Time**: 819.9 seconds (~13.6 minutes) on NVIDIA Tesla T4 GPU.

### 2.3 Clean Held-Out Test Evaluation
Source: `reports/baseline_test_evaluation_50epochs.json` (15 patients, 131 2D slices).

| Anatomical Structure | 2D Slice-Level Dice | 2D Slice-Level IoU | Patient 3D Volume Dice | Patient 3D Volume IoU |
| :--- | :---: | :---: | :---: | :---: |
| **Background (Class 0)** | 0.997541 | 0.995126 | 0.997488 ± 0.002858 | 0.995004 ± 0.005640 |
| **Right Ventricle (RV, Class 1)** | 0.813571 | 0.746451 | 0.901925 ± 0.045515 | 0.824125 ± 0.071361 |
| **Myocardium (MYO, Class 2)** | 0.791422 | 0.694209 | 0.843378 ± 0.054073 | 0.732559 ± 0.077688 |
| **Left Ventricle (LV, Class 3)** | 0.884022 | 0.844382 | 0.941952 ± 0.030138 | 0.891665 ± 0.052385 |
| **Macro Foreground (Classes 1–3)** | **0.829672** | **0.761681** | **0.895752 ± 0.037229** | **0.816116 ± 0.057651** |

*Verification*: Slice-level foreground Dice ($0.8297$) and patient-level 3D foreground Dice ($0.8958 \pm 0.0372$) evaluate the exact same 15 test patients and 131 slices without exclusions.

---

## 3. Phase 4: Controlled Robustness Evaluation Verification

Source: `reports/phase4_50epochs/robustness_results.json` and `reports/phase4_50epochs/robustness_results.csv`.  
Test Cohort: 15 patients (`patient001`, `004`, `009`, `027`, `038`, `040`, `044`, `052`, `059`, `062`, `063`, `068`, `086`, `098`, `100`), 131 end-diastolic slices (`frame01.nii.gz`).

### 3.1 Master 10-Condition Evaluation Table

| Condition ID | Family | Severity | Param ($\sigma$ or $\alpha$) | Slice FG Dice | Patient 3D FG Dice ($\mu \pm \sigma$) | Mean Paired $\Delta$ | Patient Drop ($\overline{R}_{\text{patient}}$) | Cohort Drop ($R_{\text{cohort}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_baseline` | Clean | Benchmark | $0.0$ | 0.8297 | 0.8958 ± 0.0372 | $0.0000$ | $0.00\%$ | $0.00\%$ |
| `gaussian_noise_mild` | Additive Noise | Mild | $\sigma=0.02$ | 0.8300 | 0.8957 ± 0.0374 | -0.0001 | -0.01% | -0.01% |
| `gaussian_noise_moderate` | Additive Noise | Mod | $\sigma=0.05$ | 0.8263 | 0.8948 ± 0.0379 | -0.0010 | -0.11% | -0.11% |
| `gaussian_noise_severe` | Additive Noise | Severe | $\sigma=0.10$ | 0.8213 | 0.8913 ± 0.0393 | -0.0044 | -0.51% | -0.50% |
| `gaussian_blur_mild` | Spatial Blur | Mild | $\sigma=0.75$ | 0.8259 | 0.8860 ± 0.0345 | -0.0098 | -1.07% | -1.09% |
| `gaussian_blur_moderate` | Spatial Blur | Mod | $\sigma=1.50$ | 0.7958 | 0.8540 ± 0.0428 | -0.0418 | -4.62% | -4.66% |
| `gaussian_blur_severe` | Spatial Blur | Severe | $\sigma=2.50$ | 0.7133 | 0.7775 ± 0.0800 | -0.1182 | -13.15% | -13.20% |
| `contrast_reduction_mild` | Contrast | Mild | $\alpha=0.75$ | 0.8350 | 0.8956 ± 0.0368 | -0.0002 | -0.01% | -0.02% |
| `contrast_reduction_moderate` | Contrast | Mod | $\alpha=0.50$ | 0.8337 | 0.8935 ± 0.0349 | -0.0023 | -0.24% | -0.26% |
| `contrast_reduction_severe` | Contrast | Severe | $\alpha=0.25$ | 0.7273 | 0.8158 ± 0.0677 | -0.0800 | -8.85% | -8.93% |

### 3.2 Class-Wise Patient 3D Volumetric Dice Verification

| Condition ID | RV Mean | RV Std | MYO Mean | MYO Std | LV Mean | LV Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_baseline` | 0.9019 | 0.0455 | 0.8434 | 0.0541 | 0.9420 | 0.0301 |
| `gaussian_noise_mild` | 0.9019 | 0.0454 | 0.8432 | 0.0543 | 0.9420 | 0.0302 |
| `gaussian_noise_moderate` | 0.9012 | 0.0456 | 0.8414 | 0.0547 | 0.9418 | 0.0304 |
| `gaussian_noise_severe` | 0.8970 | 0.0469 | 0.8356 | 0.0556 | 0.9413 | 0.0308 |
| `gaussian_blur_mild` | 0.8932 | 0.0441 | 0.8287 | 0.0538 | 0.9361 | 0.0306 |
| `gaussian_blur_moderate` | 0.8663 | 0.0494 | 0.7820 | 0.0645 | 0.9137 | 0.0340 |
| `gaussian_blur_severe` | 0.7821 | 0.0863 | 0.6879 | 0.0984 | 0.8625 | 0.0664 |
| `contrast_reduction_mild` | 0.9019 | 0.0452 | 0.8436 | 0.0538 | 0.9413 | 0.0298 |
| `contrast_reduction_moderate` | 0.8974 | 0.0458 | 0.8423 | 0.0520 | 0.9408 | 0.0298 |
| `contrast_reduction_severe` | 0.7482 | 0.0827 | 0.7774 | 0.0772 | 0.9219 | 0.0526 |

*Finding*: Spatial blur damages all three classes progressively (severe blur MYO drops to $0.6879$, RV to $0.7821$, LV to $0.8625$). Contrast reduction disproportionately impairs the thin, curved Right Ventricle ($0.9019 \to 0.7482$, $-17.0\%$) and Myocardium ($0.8434 \to 0.7774$, $-7.8\%$), while the high-intensity Left Ventricle blood pool remains relatively spared ($0.9420 \to 0.9219$, $-2.1\%$).

---

## 4. Phase 5: Mitigation Interventions & Pre-Registered Hypotheses Verification

Source: `experiments/phase5_validation_locked_mitigation/primary_hypotheses_results.csv`, `patient_level_results.csv`, `negative_control_results.csv`.

### 4.1 Family 1: Morphological Post-Processing Alone ($m=10$)
Intervention: Per-class Largest Connected Component (LCC) with `max_hole_size = 16` safeguard on myocardium.

| Condition ID | Baseline 3D Dice | Post-Proc 3D Dice | Delta ($\Delta$) | 95% CI of $\Delta$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `clean_baseline` | 0.8958 | 0.8980 | +0.0022 | [-0.0031, +0.0076] | 0.2294 |
| `gaussian_noise_mild` | 0.8957 | 0.8978 | +0.0021 | [-0.0033, +0.0075] | 0.2181 |
| `gaussian_noise_moderate` | 0.8948 | 0.8970 | +0.0022 | [-0.0033, +0.0076] | 0.2196 |
| `gaussian_noise_severe` | 0.8913 | 0.8942 | +0.0029 | [-0.0026, +0.0083] | 0.2905 |
| `gaussian_blur_mild` | 0.8860 | 0.8894 | +0.0034 | [-0.0037, +0.0105] | 0.2631 |
| `gaussian_blur_moderate` | 0.8540 | 0.8617 | +0.0076 | [-0.0034, +0.0187] | 0.3824 |
| `gaussian_blur_severe` | 0.7775 | 0.7915 | +0.0140 | [-0.0006, +0.0286] | 0.5297 |
| `contrast_reduction_mild` | 0.8956 | 0.8982 | +0.0026 | [-0.0033, +0.0085] | 0.2401 |
| `contrast_reduction_moderate` | 0.8935 | 0.8949 | +0.0014 | [-0.0033, +0.0061] | 0.1661 |
| `contrast_reduction_severe` | 0.8158 | 0.8156 | -0.0002 | [-0.0019, +0.0016] | -0.0492 |

### 4.2 Family 2: CLAHE Alone on Contrast Conditions ($m=3$)
Intervention: `apply_clahe` (`clip_limit = 0.01`, `grid_size = (16, 16)`, normalization to $[0, 1]$, re-standardization).

| Condition ID | Baseline 3D Dice | CLAHE 3D Dice | Delta ($\Delta$) | 95% CI of $\Delta$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `contrast_reduction_severe` | 0.8158 | 0.8896 | +0.0738 | [+0.0354, +0.1121] | 1.0647 |
| `contrast_reduction_moderate` | 0.8935 | 0.8896 | -0.0039 | [-0.0121, +0.0044] | -0.2616 |
| `contrast_reduction_mild` | 0.8956 | 0.8896 | -0.0060 | [-0.0124, +0.0004] | -0.5232 |

### 4.3 Family 3: Combined Contrast Pipeline (CLAHE + Post-Processing) ($m=3$)

| Condition ID | Baseline 3D Dice | Comb 3D Dice | Delta ($\Delta$) | 95% CI of $\Delta$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `contrast_reduction_severe` | 0.8158 | 0.8921 | +0.0763 | [+0.0375, +0.1150] | 1.0907 |
| `contrast_reduction_moderate` | 0.8935 | 0.8921 | -0.0014 | [-0.0074, +0.0046] | -0.1297 |
| `contrast_reduction_mild` | 0.8956 | 0.8921 | -0.0035 | [-0.0093, +0.0023] | -0.3383 |

### 4.4 Family 4: Combined Noise Pipeline (Median + Post-Processing) ($m=3$)
Intervention: Median filter ($3 \times 3$) + Morphological post-processing.

| Condition ID | Baseline 3D Dice | Comb 3D Dice | Delta ($\Delta$) | 95% CI of $\Delta$ | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `gaussian_noise_mild` | 0.8957 | 0.8928 | -0.0029 | [-0.0112, +0.0055] | -0.1896 |
| `gaussian_noise_moderate` | 0.8948 | 0.8922 | -0.0026 | [-0.0116, +0.0064] | -0.1607 |
| `gaussian_noise_severe` | 0.8913 | 0.8910 | -0.0003 | [-0.0111, +0.0106] | -0.0133 |

### 4.5 Negative Controls Verification (Off-Target Preprocessing)

| Negative Control Arm | Tested Condition | Unmitigated Dice | Treated Dice | Delta ($\Delta$) | Empirical Outcome |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CLAHE Alone** | `clean_baseline` | 0.8958 | 0.8896 | -0.0062 | Intensity histogram over-redistribution on clean images |
| **CLAHE Alone** | `gaussian_noise_mild` | 0.8957 | 0.8880 | -0.0076 | Mild thermal noise amplification |
| **CLAHE Alone** | `gaussian_noise_moderate` | 0.8948 | 0.8839 | -0.0109 | Progressive noise amplification |
| **CLAHE Alone** | `gaussian_noise_severe` | 0.8913 | 0.8744 | -0.0169 | Severe noise spike amplification |
| **CLAHE Alone** | `gaussian_blur_mild` | 0.8860 | 0.8829 | -0.0031 | Negligible change on mild blur |
| **CLAHE Alone** | `gaussian_blur_moderate` | 0.8540 | 0.8598 | +0.0058 | Slight edge contrast boost, non-significant |
| **CLAHE Alone** | `gaussian_blur_severe` | 0.7775 | 0.8051 | +0.0276 | Boundary contrast boost, nominal only ($p_{\text{raw}} = 0.0397$) |
| **Combined Contrast**| `clean_baseline` | 0.8958 | 0.8921 | -0.0037 | Mild degradation on clean data |
| **Combined Noise** | `clean_baseline` | 0.8958 | 0.8929 | -0.0029 | Median filter blurs sharp myocardial borders on clean data |

---

## 5. CLAHE Forensic Investigation Numerical Audit

Source: `experiments/phase5_clahe_invariance_audit/` (`patient_level_prediction_comparison.csv`, `tensor_comparison_results.csv`, `preprocessing_stage_analysis.json`).

### 5.1 Patient-by-Patient Predictions Under 5 Conditions

| Patient ID | Clean Baseline | Severe Unmitigated | Clean + CLAHE | Severe + CLAHE | $\Delta$ (Clean vs Sev CLAHE) | Severe + Z-Score Alone | $\Delta$ (Clean vs Sev Z-Score) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `patient001` | 0.8932 | 0.8727 | 0.8998 | 0.8998 | **0.0000** | 0.8932 | **0.0000** |
| `patient004` | 0.9178 | 0.8745 | 0.9157 | 0.9157 | **0.0000** | 0.9178 | **0.0000** |
| `patient009` | 0.8827 | 0.8926 | 0.8847 | 0.8847 | **0.0000** | 0.8827 | **0.0000** |
| `patient027` | 0.9088 | 0.8836 | 0.8916 | 0.8916 | **0.0000** | 0.9088 | **0.0000** |
| `patient038` | 0.8026 | 0.7058 | 0.7766 | 0.7766 | **0.0000** | 0.8026 | **0.0000** |
| `patient040` | 0.9291 | 0.6647 | 0.9268 | 0.9268 | **0.0000** | 0.9291 | **0.0000** |
| `patient044` | 0.9131 | 0.8230 | 0.9062 | 0.9062 | **0.0000** | 0.9131 | **0.0000** |
| `patient052` | 0.8857 | 0.8763 | 0.8826 | 0.8826 | **0.0000** | 0.8857 | **0.0000** |
| `patient059` | 0.9361 | 0.7798 | 0.9260 | 0.9260 | **0.0000** | 0.9361 | **0.0000** |
| `patient062` | 0.9383 | 0.8797 | 0.9318 | 0.9318 | **0.0000** | 0.9383 | **0.0000** |
| `patient063` | 0.8830 | 0.7785 | 0.8851 | 0.8851 | **0.0000** | 0.8830 | **0.0000** |
| `patient068` | 0.9207 | 0.8086 | 0.9257 | 0.9257 | **0.0000** | 0.9207 | **0.0000** |
| `patient086` | 0.8345 | 0.8047 | 0.8183 | 0.8183 | **0.0000** | 0.8345 | **0.0000** |
| `patient098` | 0.8810 | 0.7777 | 0.8647 | 0.8647 | **0.0000** | 0.8810 | **0.0000** |
| `patient100` | 0.9096 | 0.8148 | 0.9079 | 0.9079 | **0.0000** | 0.9096 | **0.0000** |
| **Cohort Mean** | **0.8958** | **0.8158** | **0.8896** | **0.8896** | **0.0000** | **0.8958** | **0.0000** |
| **Cohort Std** | **0.0372** | **0.0677** | **0.0429** | **0.0429** | **0.0000** | **0.0372** | **0.0000** |

### 5.2 Tensor-Level Stage Measurements

- **Max absolute difference between Clean Unit ($C$) and Corrupt Unit ($D$) across 131 slices**: $2.3842 \times 10^{-7}$ (strictly bounded by float32 machine epsilon $2^{-24} \approx 5.96 \times 10^{-8}$ to $1.19 \times 10^{-7}$).
- **Mean absolute difference between $C$ and $D$**: $3.12 \times 10^{-10}$.
- **Percentage of matching pixels between $C$ and $D$ within $10^{-5}$ tolerance**: $100.00\%$.
- **Max absolute difference between Clean CLAHE ($E$) and Corrupt CLAHE ($F$)**: $0.0000 \times 10^0$ (exact numerical array identity).
- **Model prediction pixel agreement between $E$ and $F$**: $100.00\%$ across all 8,585,216 evaluated voxels ($131 \times 256 \times 256$).

### 5.3 Causal Decomposition Summary
$$\Delta_{\text{total}} = 0.8896 - 0.8158 = +0.0738$$
- **Linear Normalization Share**: $+0.0800$ ($108.4\%$ of total change).
- **CLAHE Share**: $-0.0062$ ($-8.4\%$ penalty).
- **Empirical Reality**: Pure linear standardization restores $100.0\%$ of the clean baseline performance ($0.8958$). Adding CLAHE reduces performance by $-0.0062$.
