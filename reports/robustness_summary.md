# Phase 4 Controlled Robustness Evaluation Summary Report

**Project**: ACDC Cardiac MRI Semantic Segmentation Robustness Evaluation  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Execution Timestamp**: 2026-10-21  
**Checkpoint Evaluated**: `checkpoints/best_model_frozen.pt`  
**Checkpoint SHA-256**: `e7bfa884bef0f4eafce30cd6948c8d60fc4e7c8c8ca2d0bb0e83f2977fc29c02`  
**Test Cohort**: Held-out 15-patient test split (131 2D short-axis cine MRI slices, completely disjoint from training)  
**Input Space Protocol**: Controlled corruptions applied strictly to normalized model-input tensors without post-corruption re-normalization or unprincipled range clipping.

---

## 1. Executive Summary & Critical Clinical Scope

This report documents the empirical evaluation of the frozen lightweight 2D U-Net under **10 pre-specified experimental conditions** across three degradation families: **Additive Gaussian Noise**, **Spatial Gaussian Blur**, and **Contrast Reduction**.

> [!WARNING] Methodological & Clinical Scope
> 1. **Synthetic Perturbation vs Real Clinical Scan Robustness**:
>    The measured robustness reflects performance under mathematically controlled, synthetic perturbations (additive i.i.d. Gaussian noise, isotropic 2D Gaussian blur, and global linear contrast scaling).
>    **This does NOT automatically prove robustness to real clinical MRI scans or multicenter domain shifts.** Real clinical MRI artifacts involve spatially non-uniform $B_0/B_1$ magnetic field inhomogeneities (bias field shading), respiratory and cardiac phase-encoding ghosting/blurring, slice misregistration from patient breathing, cross-vendor reconstruction discrepancies (Siemens vs Philips vs GE), and 1.5T vs 3.0T pulse sequence physics (SSFP vs FLASH).
> 2. **Scope of Comparative Findings**:
>    The finding that the baseline model is more sensitive to contrast reduction than to noise reflects performance **under the specific tested parameter ranges** ($\sigma \in [0.02, 0.10]$ vs $\alpha \in [0.75, 0.25]$). It does not claim that all contrast shifts in clinical medicine are more damaging than all noise or blur artifacts.

---

## 2. Metric Hierarchy & Anatomical 3D Volume Validity

### Three Distinct Operational Levels:
1. **Slice-Level 2D Evaluation**:
   - Evaluated independently on every 2D slice $s \in \{1, \dots, 131\}$:
     $$\text{Dice}_s = \frac{2 |Y_s \cap \hat{Y}_s|}{|Y_s| + |\hat{Y}_s|}$$
   - Arithmetic mean is computed across all 131 slices.
2. **Patient-Level Voxel Pooling (Implemented in Phase 4)**:
   - For patient $p$, binary overlap counts are summed across the $S_p$ contiguous slices of the patient's End-Diastolic volume:
     $$I_{p, c} = \sum_{s=0}^{S_p - 1} \sum_{x, y} (\hat{Y}_{p, s}(x, y) == c \land Y_{p, s}(x, y) == c)$$
     $$C_{p, c} = \sum_{s=0}^{S_p - 1} \sum_{x, y} (\hat{Y}_{p, s}(x, y) == c) + \sum_{s=0}^{S_p - 1} \sum_{x, y} (Y_{p, s}(x, y) == c)$$
     $$\text{Dice}_{p, c} = \frac{2 I_{p, c}}{C_{p, c}} \quad (\text{if } C_{p, c} > 0, \text{ else } 1.0)$$
   - *Slice Ordering*: Slices are ordered along the anatomical long axis ($z=0, \dots, S_p-1$).
   - *Empty Class Handling*: If both ground truth and prediction have 0 voxels for class $c$ across the patient volume, $\text{Dice}_{p, c} = 1.0$ (True Negative). If one has >0 and the other 0, $\text{Dice}_{p, c} = 0.0$.
3. **Full 3D Volumetric Evaluation in Physical Space**:
   - Full 3D evaluation incorporates physical voxel dimensions $(dx, dy, dz)$, slice spacing, and affine coordinate transformations to calculate physical surface distances (Hausdorff Distance 95%, Average Surface Distance) and absolute ventricular cavity volumes in milliliters (mL).
   - Our patient pooling evaluates voxel-grid volume overlap; physical distance metrics (HD95) will be computed in Phase 5.

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
- **Zero-Clean Check**: Across all 15 test patients, clean Dice ranges from `0.3945` to `0.8616`. **Zero patients had clean Dice $= 0.0$**; no undefined divisions occurred.

### Full 15-Patient Audit for Severe Contrast Reduction

| Patient ID | Clean Dice | Corrupted Dice | Absolute $\Delta_p$ | Patient-Wise Relative Drop ($\%_p$) |
| :--- | :---: | :---: | :---: | :---: |
| `patient001` | 0.7708 | 0.1257 | -0.6451 | -83.70% |
| `patient004` | 0.7385 | 0.0403 | -0.6982 | -94.54% |
| `patient009` | 0.8080 | 0.0886 | -0.7194 | -89.04% |
| `patient027` | 0.8140 | 0.1305 | -0.6835 | -83.97% |
| `patient038` | 0.3945 | 0.0000 | -0.3945 | -100.00% |
| `patient040` | 0.8616 | 0.0000 | -0.8616 | -100.00% |
| `patient044` | 0.7766 | 0.0058 | -0.7708 | -99.25% |
| `patient052` | 0.7135 | 0.0517 | -0.6618 | -92.76% |
| `patient059` | 0.7506 | 0.0000 | -0.7506 | -100.00% |
| `patient062` | 0.8171 | 0.0640 | -0.7531 | -92.17% |
| `patient063` | 0.7459 | 0.0412 | -0.7047 | -94.48% |
| `patient068` | 0.6647 | 0.0261 | -0.6387 | -96.08% |
| `patient086` | 0.7600 | 0.0317 | -0.7283 | -95.83% |
| `patient098` | 0.7198 | 0.0025 | -0.7173 | -99.65% |
| `patient100` | 0.6972 | 0.0000 | -0.6972 | -100.00% |
| **Cohort Mean** | **0.7355** | **0.0405** | **-0.6950** | **-94.7634% (-94.76%)** |

### Definition B: Relative Change of Cohort Means ($R_{\text{cohort}}$)
$$R_{\text{cohort}} = \left( \frac{\overline{\text{Dice}}_{\text{corrupt}} - \overline{\text{Dice}}_{\text{clean}}}{\overline{\text{Dice}}_{\text{clean}}} \right) \times 100\% = \frac{0.0405 - 0.7355}{0.7355} \times 100\% = \mathbf{-94.49\%}$$

The two values reflect the mathematical property that the **Average of Ratios** ($\overline{R}_{\text{patient}} = -94.76\%$) does not equal the **Ratio of Averages** ($R_{\text{cohort}} = -94.49\%$).

---

## 4. Statistical Unit & Multiple Comparison Correction

### Statistical Unit: Patient ($N = 15$)
- Slices from the same patient are spatially correlated. Treating $N=131$ slices as independent observations represents pseudoreplication (inflated Type I error rate).
- The correct statistical unit is the **patient ($N = 15$)**, where each patient provides one paired observation $(\text{Dice}_{\text{clean}, p}, \text{Dice}_{\text{corrupt}, p})$.

### Full Table of All 10 Conditions with Raw and Holm-Adjusted $p$-values

| Condition ID | Family | Severity Level | Parameter Value | Slice FG Dice | Patient 3D FG Dice ($\mu \pm \sigma$) | Mean Paired $\Delta$ | Patient Drop ($\overline{R}_{\text{patient}}$) | Cohort Drop ($R_{\text{cohort}}$) | Paired $t$ $p_{\text{raw}}$ | Paired $t$ $p_{\text{Holm}}$ | Wilcoxon $p_{\text{raw}}$ | Wilcoxon $p_{\text{Holm}}$ | FWER Controlled Result ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_baseline` | Clean | None | $0.0$ | **0.6662** | **0.7355 $\pm$ 0.1074** | $0.0000$ | $0.0\%$ | $0.0\%$ | N/A | N/A | N/A | N/A | Benchmark |
| `gaussian_noise_mild` | Noise | Mild | $\sigma=0.02$ | 0.6654 | 0.7355 $\pm$ 0.1070 | $-0.0001$ | $0.00\%$ | $-0.01\%$ | $0.8450$ | $1.0000$ | $0.8469$ | $0.8469$ | Not Significant |
| `gaussian_noise_moderate` | Noise | Mod | $\sigma=0.05$ | 0.6620 | 0.7345 $\pm$ 0.1057 | $-0.0010$ | $-0.08\%$ | $-0.14\%$ | $0.3683$ | $1.0000$ | $0.2769$ | $0.5537$ | Not Significant |
| `gaussian_noise_severe` | Noise | Sev | $\sigma=0.10$ | 0.6542 | 0.7313 $\pm$ 0.1003 | $-0.0042$ | $-0.34\%$ | $-0.57\%$ | $0.2369$ | $0.9478$ | $0.1354$ | $0.5415$ | Not Significant |
| `gaussian_blur_mild` | Blur | Mild | $\sigma=0.75$ px | 0.6617 | 0.7287 $\pm$ 0.1084 | $-0.0068$ | $-0.93\%$ | $-0.93\%$ | $0.1216$ | $0.6081$ | $0.0730$ | $0.3650$ | Not Significant |
| `gaussian_blur_moderate` | Blur | Mod | $\sigma=1.50$ px | 0.6477 | 0.7077 $\pm$ 0.1084 | $-0.0279$ | $-3.76\%$ | $-3.79\%$ | $4.37 \times 10^{-3}$ | **0.0262** | $6.71 \times 10^{-3}$ | **0.0403** | **Significant** |
| `gaussian_blur_severe` | Blur | Sev | $\sigma=2.50$ px | 0.6051 | 0.6718 $\pm$ 0.1147 | $-0.0637$ | $-8.67\%$ | $-8.66\%$ | $3.14 \times 10^{-4}$ | **0.0022** | $6.10 \times 10^{-4}$ | **0.0043** | **Significant** |
| `contrast_reduction_mild` | Contrast | Mild | $\alpha=0.75$ | 0.6594 | 0.7310 $\pm$ 0.1147 | $-0.0046$ | $-0.83\%$ | $-0.62\%$ | $0.5158$ | $1.0000$ | $0.1354$ | $0.5415$ | Not Significant |
| `contrast_reduction_moderate` | Contrast | Mod | $\alpha=0.50$ | 0.5540 | 0.6110 $\pm$ 0.1348 | $-0.1245$ | $-17.40\%$ | $-16.93\%$ | $3.34 \times 10^{-5}$ | **2.67 $\times 10^{-4}$** | $6.10 \times 10^{-5}$ | **5.49 $\times 10^{-4}$** | **Significant** |
| `contrast_reduction_severe` | Contrast | Sev | $\alpha=0.25$ | **0.0861** | **0.0405 $\pm$ 0.0446** | **-0.6950** | **-94.76%** | **-94.49%** | **$1.81 \times 10^{-13}$** | **1.63 $\times 10^{-12}$** | **$6.10 \times 10^{-5}$** | **5.49 $\times 10^{-4}$** | **Significant** |

*Note on Holm Step-Down Mechanics & Precision*:
1. **Holm Step-Down Multipliers**: Raw $p$-values are sorted in ascending order $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$. For 1-based rank $i \in \{1, \dots, m\}$ ($m=9$), the nominal multiplier is $(m - i + 1)$. This is followed by running-maximum monotonicity enforcement ($p_{\text{adj},(i)} = \max(p_{\text{adj},(i-1)}, (m - i + 1) \cdot p_{(i)})$) and capping at $1.0$. For condition `gaussian_noise_mild`, which is at the highest rank ($i = 9$ in 1-based indexing, corresponding to index $k = 8$ in 0-based indexing), the step multiplier is $(9 - 9 + 1) = 1$, which is why its Holm-adjusted value equals its raw $p$-value ($p_{\text{Holm}} = p_{\text{raw}} = 0.8469$).
2. **Parametric vs Non-Parametric Separation**: Paired $t$-tests and Wilcoxon signed-rank tests are separate statistical procedures with distinct foundational assumptions: paired $t$-test assumes that paired intra-patient differences follow a normal distribution, whereas the Wilcoxon signed-rank test is non-parametric and only assumes symmetric difference distributions. Both tests are reported independently to demonstrate that significance conclusions are invariant to distribution assumptions.
3. **Rounding Precision in Patient Tables**: The patient-wise relative drop of $-94.7634\%$ is computed from double-precision float64 arrays in [`reports/robustness_results.json`](file:///h:/Mini%20project/reports/robustness_results.json). When averaging the displayed two-decimal rounded percentages in Table 3 ($\frac{1}{15}\sum \%_p = \frac{-1421.47\%}{15} = -94.7647\%$), the result reproduces $-94.76\%$ identically to two decimal places.

---

## 5. Reconciled Test Suite Summary (66 Tests Across 11 Modules)

```text
======================== 66 passed, 1 warning in 7.24s ========================
1.  tests/test_config.py ............................................... 5 passed
2.  tests/test_corruptions.py .......................................... 10 passed
3.  tests/test_data_pipeline.py ........................................ 8 passed
4.  tests/test_dataset.py .............................................. 4 passed
5.  tests/test_losses_and_metrics.py ................................... 6 passed
6.  tests/test_model.py ................................................ 5 passed
7.  tests/test_paths.py ................................................ 6 passed
8.  tests/test_patient_metrics.py ...................................... 5 passed
9.  tests/test_preprocessing.py ........................................ 8 passed
10. tests/test_reproducibility.py ...................................... 6 passed
11. tests/test_robustness_eval.py ...................................... 3 passed
```
