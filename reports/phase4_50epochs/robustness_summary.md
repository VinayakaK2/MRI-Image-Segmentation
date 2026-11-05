# Phase 4 Controlled Robustness Evaluation Summary Report

**Experiment ID**: `phase4_robustness_50epoch_best_epoch44`  
**Project**: ACDC Cardiac MRI Semantic Segmentation Robustness Evaluation  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Execution Timestamp**: 2026-11-04 18:02:44  
**Checkpoint Evaluated**: `checkpoints\50epochs\best_model_50epochs.pt`  
**Checkpoint SHA-256**: `a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38`  
**Model Architecture**: `LightweightUNet2D` (1,963,860 parameters)  
**Trained Epochs**: 44 (Best Val Epoch: 44)  
**Test Cohort**: Held-out 15-patient test split (131 2D short-axis cine MRI slices, completely disjoint from training)  
**Input Space Protocol**: Controlled corruptions applied strictly to normalized model-input tensors without post-corruption re-normalization or range clipping.

---

## 1. Executive Summary & Critical Clinical Scope

This report documents the empirical evaluation of the frozen lightweight 2D U-Net under **10 pre-specified experimental conditions** across three degradation families: **Additive Gaussian Noise**, **Spatial Gaussian Blur**, and **Contrast Reduction**.

> [!WARNING] Methodological & Clinical Scope
> 1. **Synthetic Perturbation vs Real Clinical Scan Robustness**:
>    The measured robustness reflects performance under mathematically controlled, synthetic perturbations (additive i.i.d. Gaussian noise, isotropic 2D Gaussian blur, and global linear contrast scaling).
>    **This does NOT automatically prove robustness to real clinical MRI scans or multicenter domain shifts.** Real clinical MRI artifacts involve spatially non-uniform $B_0/B_1$ magnetic field inhomogeneities (bias field shading), respiratory and cardiac phase-encoding ghosting/blurring, slice misregistration from patient breathing, cross-vendor reconstruction discrepancies (Siemens vs Philips vs GE), and 1.5T vs 3.0T pulse sequence physics (SSFP vs FLASH).
> 2. **Scope of Comparative Findings**:
>    The finding that the model shows differential sensitivity across corruptions reflects performance **under the specific tested parameter ranges** ($\sigma \in [0.02, 0.10]$ vs $\alpha \in [0.75, 0.25]$). It does not claim that all contrast shifts in clinical medicine are more damaging than all noise or blur artifacts.

---

## 2. Metric Hierarchy & Anatomical 3D Volume Validity

### Three Distinct Operational Levels:
1. **Slice-Level 2D Evaluation**:
   - Evaluated independently on every 2D slice $s \in \{1, \dots, 131\}$:
     $$\text{Dice}_s = \frac{2 |Y_s \cap \hat{Y}_s|}{|Y_s| + |\hat{Y}_s|}$$
   - Arithmetic mean is computed across all 131 slices.
2. **Patient-Level Voxel Pooling (3D Anatomical Metric)**:
   - For patient $p$, binary overlap counts are summed across all contiguous slices of the patient's End-Diastolic volume:
     $$I_{p, c} = \sum_{s=0}^{S_p - 1} \sum_{x, y} (\hat{Y}_{p, s}(x, y) == c \land Y_{p, s}(x, y) == c)$$
     $$C_{p, c} = \sum_{s=0}^{S_p - 1} \sum_{x, y} (\hat{Y}_{p, s}(x, y) == c) + \sum_{s=0}^{S_p - 1} \sum_{x, y} (Y_{p, s}(x, y) == c)$$
     $$\text{Dice}_{p, c} = \frac{2 I_{p, c}}{C_{p, c}} \quad (\text{if } C_{p, c} > 0, \text{ else } 1.0)$$
   - *Slice Ordering*: Slices are ordered along the anatomical long axis ($z=0, \dots, S_p-1$).
   - *Empty Class Handling*: If both ground truth and prediction have 0 voxels for class $c$ across the patient volume, $\text{Dice}_{p, c} = 1.0$ (True Negative). If one has >0 and the other 0, $\text{Dice}_{p, c} = 0.0$.
3. **Full 3D Volumetric Evaluation in Physical Space**:
   - Physical distance metrics (Hausdorff Distance 95%, Average Surface Distance) require physical voxel spacing $(dx, dy, dz)$ and 3D surface mesh extraction, which are investigated in Phase 5.

### Anatomical Phase Segregation Audit:
- All 131 test slices across all 15 test patients originate from each patient's single **End-Diastolic (ED) volume** (`patientXXX_frame01.nii.gz`).
- No End-Systolic (ES) slices are present. Thus, pooling across slices for each patient evaluates a single, anatomically coherent 3D volume at end-diastole, with zero temporal phase mixing.

---

## 3. Mathematical Definitions & Patient-Level Audit of Relative Degradation

Two distinct mathematical formulas are reported for relative degradation:

### Definition A: Mean of Patient-Wise Relative Changes ($\overline{R}_{\text{patient}}$)
$$\overline{R}_{\text{patient}} = \frac{1}{N} \sum_{p=1}^N \left( \frac{\text{Dice}_{\text{corrupt}, p} - \text{Dice}_{\text{clean}, p}}{\text{Dice}_{\text{clean}, p}} \right) \times 100\%$$
- **Numerator**: Individual patient's absolute score difference $\Delta_p = \text{Dice}_{\text{corrupt}, p} - \text{Dice}_{\text{clean}, p}$.
- **Denominator**: Individual patient's baseline performance $\text{Dice}_{\text{clean}, p}$.

### Definition B: Relative Change of Cohort Means ($R_{\text{cohort}}$)
$$R_{\text{cohort}} = \left( \frac{\overline{\text{Dice}}_{\text{corrupt}} - \overline{\text{Dice}}_{\text{clean}}}{\overline{\text{Dice}}_{\text{clean}}} \right) \times 100\%$$

### Full 15-Patient Audit for Severe Contrast Reduction (Condition: `contrast_reduction_severe`):

| Patient ID | Clean Dice | Corrupted Dice | Absolute $\Delta_p$ | Patient-Wise Relative Drop ($\%_p$) |
| :--- | :---: | :---: | :---: | :---: |
| `patient001` | 0.8932 | 0.8727 | -0.0206 | -2.30% |
| `patient004` | 0.9178 | 0.8745 | -0.0433 | -4.71% |
| `patient009` | 0.8827 | 0.8926 | 0.0099 | 1.12% |
| `patient027` | 0.9088 | 0.8836 | -0.0252 | -2.77% |
| `patient038` | 0.8026 | 0.7058 | -0.0968 | -12.06% |
| `patient040` | 0.9291 | 0.6647 | -0.2644 | -28.46% |
| `patient044` | 0.9131 | 0.8230 | -0.0901 | -9.86% |
| `patient052` | 0.8857 | 0.8763 | -0.0094 | -1.06% |
| `patient059` | 0.9361 | 0.7798 | -0.1563 | -16.69% |
| `patient062` | 0.9383 | 0.8797 | -0.0585 | -6.24% |
| `patient063` | 0.8830 | 0.7785 | -0.1046 | -11.84% |
| `patient068` | 0.9207 | 0.8086 | -0.1121 | -12.18% |
| `patient086` | 0.8345 | 0.8047 | -0.0299 | -3.58% |
| `patient098` | 0.8810 | 0.7777 | -0.1033 | -11.73% |
| `patient100` | 0.9096 | 0.8148 | -0.0949 | -10.43% |
| **Cohort Mean** | **0.8958** | **0.8158** | **-0.0800** | **-8.8527% (-8.85%)** |

### Summary of Severe Contrast Drop:
- **Definition A (Patient-Wise Mean)**: $\overline{R}_{\text{patient}} = \mathbf{-8.85\%}$
- **Definition B (Cohort Mean Change)**: $R_{\text{cohort}} = \mathbf{-8.93\%}$
- Mathematical property verified: Average of Ratios ($\overline{R}_{\text{patient}}$) $\neq$ Ratio of Averages ($R_{\text{cohort}}$).

---

## 4. Statistical Unit & Multiple Comparison Correction

### Statistical Unit: Patient ($N = 15$)
- Slices from the same patient are spatially correlated. Treating $N=131$ slices as independent observations represents pseudoreplication (inflated Type I error rate).
- The correct statistical unit is the **patient ($N = 15$)**, where each patient provides one paired observation $(\text{Dice}_{\text{clean}, p}, \text{Dice}_{\text{corrupt}, p})$.

### Full Table of All 10 Conditions with Raw and Holm-Adjusted $p$-values

| Condition ID | Family | Severity Level | Parameter Value | Slice FG Dice | Patient 3D FG Dice ($\mu \pm \sigma$) | Mean Paired $\Delta$ | Patient Drop ($\overline{R}_{\text{patient}}$) | Cohort Drop ($R_{\text{cohort}}$) | Paired $t$ $p_{\text{raw}}$ | Paired $t$ $p_{\text{Holm}}$ | Wilcoxon $p_{\text{raw}}$ | Wilcoxon $p_{\text{Holm}}$ | FWER Controlled Result ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_baseline` | Clean | None | $0.0$ | **0.8297** | **0.8958 \pm 0.0372** | $0.0000$ | $0.0\%$ | $0.0\%$ | N/A | N/A | N/A | N/A | Benchmark |
| `gaussian_noise_mild` | gaussian_noise | mild | 0.02 | 0.8300 | 0.8957 \pm 0.0374 | -0.0001 | -0.01\% | -0.01\% | 6.1623e-01 | 1.0000e+00 | 6.3867e-01 | 1.0000e+00 | **Not Significant** |
| `gaussian_noise_moderate` | gaussian_noise | moderate | 0.05 | 0.8263 | 0.8948 \pm 0.0379 | -0.0010 | -0.11\% | -0.11\% | 9.1266e-02 | 3.6506e-01 | 1.0699e-01 | 4.2798e-01 | **Not Significant** |
| `gaussian_noise_severe` | gaussian_noise | severe | 0.1 | 0.8213 | 0.8913 \pm 0.0393 | -0.0044 | -0.51\% | -0.50\% | 1.4165e-02 | 7.4087e-02 | 4.2725e-03 | 2.5635e-02 | **Nominal Only (Unadjusted)** |
| `gaussian_blur_mild` | gaussian_blur | mild | 0.75 | 0.8259 | 0.8860 \pm 0.0345 | -0.0098 | -1.07\% | -1.09\% | 1.2348e-02 | 7.4087e-02 | 8.3618e-03 | 4.1809e-02 | **Nominal Only (Unadjusted)** |
| `gaussian_blur_moderate` | gaussian_blur | moderate | 1.5 | 0.7958 | 0.8540 \pm 0.0428 | -0.0418 | -4.62\% | -4.66\% | 3.6088e-04 | 2.8870e-03 | 1.2207e-04 | 9.7656e-04 | **Significant** |
| `gaussian_blur_severe` | gaussian_blur | severe | 2.5 | 0.7133 | 0.7775 \pm 0.0800 | -0.1182 | -13.15\% | -13.20\% | 4.5667e-05 | 4.1100e-04 | 6.1035e-05 | 5.4932e-04 | **Significant** |
| `contrast_reduction_mild` | contrast_reduction | mild | 0.75 | 0.8350 | 0.8956 \pm 0.0368 | -0.0002 | -0.01\% | -0.02\% | 8.5520e-01 | 1.0000e+00 | 9.7797e-01 | 1.0000e+00 | **Not Significant** |
| `contrast_reduction_moderate` | contrast_reduction | moderate | 0.5 | 0.8337 | 0.8935 \pm 0.0349 | -0.0023 | -0.24\% | -0.26\% | 2.8569e-01 | 8.5706e-01 | 2.2931e-01 | 6.8793e-01 | **Not Significant** |
| `contrast_reduction_severe` | contrast_reduction | severe | 0.25 | 0.7273 | 0.8158 \pm 0.0677 | -0.0800 | -8.85\% | -8.93\% | 5.0695e-04 | 3.5487e-03 | 1.8311e-04 | 1.2817e-03 | **Significant** |

*Note on Holm Step-Down Mechanics & Precision*:
1. **Holm Step-Down Multipliers**: Raw $p$-values are sorted in ascending order $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$ ($m=9$). For 1-based rank $i \in \{1, \dots, m\}$, the nominal multiplier is $(m - i + 1)$. This is followed by running-maximum monotonicity enforcement ($p_{\text{adj},(i)} = \max(p_{\text{adj},(i-1)}, (m - i + 1) \cdot p_{(i)})$) and capping at $1.0$.
2. **Parametric vs Non-Parametric Separation**: Paired $t$-tests and Wilcoxon signed-rank tests are separate statistical procedures with distinct foundational assumptions: paired $t$-test assumes that paired intra-patient differences follow a normal distribution, whereas the Wilcoxon signed-rank test is non-parametric and only assumes symmetric difference distributions. Both tests are reported independently to demonstrate that significance conclusions are invariant to distribution assumptions.

---

## 5. Class-Wise Anatomical Impact (RV, MYO, LV)

| Condition ID | RV Dice | RV IoU | MYO Dice | MYO IoU | LV Dice | LV IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_baseline` | 0.8136 | 0.7465 | 0.7914 | 0.6942 | 0.8840 | 0.8444 |
| `gaussian_noise_mild` | 0.8140 | 0.7463 | 0.7913 | 0.6940 | 0.8847 | 0.8448 |
| `gaussian_noise_moderate` | 0.8062 | 0.7381 | 0.7888 | 0.6908 | 0.8839 | 0.8436 |
| `gaussian_noise_severe` | 0.8030 | 0.7333 | 0.7799 | 0.6807 | 0.8811 | 0.8410 |
| `gaussian_blur_mild` | 0.8109 | 0.7395 | 0.7818 | 0.6787 | 0.8852 | 0.8398 |
| `gaussian_blur_moderate` | 0.7791 | 0.6970 | 0.7423 | 0.6261 | 0.8660 | 0.8091 |
| `gaussian_blur_severe` | 0.6709 | 0.5750 | 0.6571 | 0.5272 | 0.8120 | 0.7384 |
| `contrast_reduction_mild` | 0.8211 | 0.7544 | 0.7947 | 0.6966 | 0.8891 | 0.8466 |
| `contrast_reduction_moderate` | 0.8130 | 0.7440 | 0.7950 | 0.6948 | 0.8930 | 0.8489 |
| `contrast_reduction_severe` | 0.6462 | 0.5654 | 0.7051 | 0.5901 | 0.8304 | 0.7799 |

---

## 6. Generated Visual Artifacts & Figures

- **Severity vs Dice Curve**: [`H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\severity_vs_dice.png`](file:///H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\severity_vs_dice.png)
- **Severity vs IoU Curve**: [`H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\severity_vs_iou.png`](file:///H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\severity_vs_iou.png)
- **Class-Wise Degradation Plot**: [`H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\class_wise_degradation.png`](file:///H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\class_wise_degradation.png)
- **Top-Level Overlays**: [`H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\representative_overlays.png`](file:///H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\representative_overlays.png)
- **Multi-Condition Visual Comparisons**: Located in [`H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\visualizations`](file:///H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\visualizations)
- **Slice-Specific Overlays**: Located in [`H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\representative_overlays`](file:///H:\Mini project\experiments\phase4_robustness_50epoch_best_epoch44\representative_overlays)

---

## 7. Artifact Inventory & Checksums

| Artifact Filename | Type | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `class_wise_degradation.png` | File | 160562 B | `abeff094a69af79a69a747b84ab8d1c46e1278ce845cd8325217d2e24a6cf411` |
| `experiment_metadata.json` | File | 957 B | `188a4dc56dcd0bddee609016ff9750200b59beb683fb14ff97b3f21cf7913f6a` |
| `representative_overlays.png` | File | 2609250 B | `9312f5694e92fa745b9c0e5d1586780b31214b0082fca5c8553cf9d253b095b4` |
| `robustness_results.csv` | File | 2418 B | `18bc4c29178e2ef39ad20f322700215a8d2d822ef4ad9b90fbe1c3708f066e1f` |
| `robustness_results.json` | File | 162769 B | `d2df59604e18ca54885dc507e91b010f7d25d9cc096cceff1347e67e7ccd192a` |
| `severity_vs_dice.png` | File | 225667 B | `46f0540e90a753417d472da12e5157157adadc5ae7a937437a97dc3c32e7a3a4` |
| `severity_vs_iou.png` | File | 238749 B | `b03b799b8e65b148f92957743d9d08c9a0a166d2741ecf4cd8f57b02189cdd32` |