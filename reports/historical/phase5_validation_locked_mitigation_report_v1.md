# Phase 5 — Validation-Locked Mitigation Experiment: Final Scientific Report

**Experiment Identifier**: `phase5_validation_locked_mitigation`  
**Date**: November 5, 2026  
**Auditor / Investigator**: Senior Medical Imaging ML Research Engineer & Statistical Validation Specialist  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Git Branch**: `phase5-mitigation-audit`  
**Target Architecture**: `LightweightUNet2D` (1,963,860 parameters)  
**Evaluated Model Checkpoint**: `checkpoints/50epochs/best_model_50epochs.pt`  
- Selected Validation Epoch: Epoch 44  
- Training Run: 50 epochs (fresh random initialization)  
- Cryptographic SHA-256: `a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38`  
**Reference Frozen Baseline**: `checkpoints/best_model_frozen.pt` (3-epoch baseline)  
- Cryptographic SHA-256: `e7bfa884bef0f4eafce30cd6948c8d60fc4e7c8c8ca2d0bb0e83f2977fc29c02` (Verified Untouched)  
**Held-Out Test Cohort**: 15 patients (`patient001`, `004`, `009`, `027`, `038`, `040`, `044`, `052`, `059`, `062`, `063`, `068`, `086`, `098`, `100`), 131 end-diastolic 2D SAX slices (`frame01.nii.gz`)  

---

## 1. Executive Summary

Phase 5 investigated the efficacy of **pre-registered, validation-locked mitigation strategies** designed to restore segmentation accuracy in cardiac MRI under controlled image corruptions. Evaluation was performed on the held-out 15-patient test cohort across 10 experimental conditions (Clean Baseline, 3 Gaussian Noise levels, 3 Gaussian Blur levels, and 3 Contrast Reduction levels) using the newly trained 50-epoch baseline model (`best_model_50epochs.pt`, selected at Epoch 44).

### Key Empirical Findings:
1. **Severe Contrast Reduction Rescue**: Contrast reduction at $\alpha=0.25$ causes substantial segmentation degradation, reducing patient 3D macro foreground Dice from $0.8958 \pm 0.0372$ to $0.8158 \pm 0.0677$. Application of locked **CLAHE** (`clip_limit=0.01`, `grid_size=(16, 16)`) rescues patient Dice to **$0.8896 \pm 0.0429$** ($\Delta = +0.0738$, paired $t = 4.1235$, $p_{\text{raw}} = 1.0336 \times 10^{-3}$, $p_{\text{Holm}} = 3.1008 \times 10^{-3}$, Wilcoxon $p_{\text{Holm}} = 9.1553 \times 10^{-4}$, Cohen's $d = 1.0647$, $95\%\text{ CI} = [0.0354, 0.1121]$). This achieves **$99.3\%$ recovery** back to clean baseline levels and is statistically significant after Family-Wise Error Rate (FWER) control.
2. **Combined Contrast Pipeline Efficacy**: Integrating CLAHE with morphological post-processing achieves **$0.8921 \pm 0.0371$** ($\Delta = +0.0763$, paired $t = 4.2242$, $p_{\text{raw}} = 8.4967 \times 10^{-4}$, $p_{\text{Holm}} = 2.5490 \times 10^{-3}$, Wilcoxon $p_{\text{Holm}} = 5.4932 \times 10^{-4}$, Cohen's $d = 1.0907$, $95\%\text{ CI} = [0.0375, 0.1150]$), confirming significant restoration.
3. **Analytical Contrast Invariance of CLAHE**: Under affine contrast degradation $I_{\text{corrupt}} = \alpha I + (1-\alpha)\mu$, unit-interval normalization cancels out $\alpha$. Consequently, CLAHE receives mathematically identical $[0, 1]$ input tensors across all $\alpha > 0$. Empirical audit confirms a maximum per-patient score difference of **$0.00 \times 10^{0}$** between Clean+CLAHE and Severe+CLAHE across all 15 test patients.
4. **Morphological Post-Processing Generalization**: Across all 10 conditions, morphological post-processing (per-class LCC with the myocardium cavity safeguard) yields small positive deltas ($\Delta \approx +0.0014$ to $+0.0140$). On severe blur, post-processing improves Dice from $0.7775$ to $0.7915$ ($\Delta = +0.0140$, $+1.40\%$, paired $t = 2.0514$, $p_{\text{raw}} = 0.0594$, Wilcoxon $p_{\text{raw}} = 0.0231$). Under strict Holm-Bonferroni correction ($m=10$), this is classified as *Nominal Only (Unadjusted)*.
5. **Denoising Pipeline Limits**: Median filtering ($3 \times 3$) combined with post-processing does not yield statistically significant gains on Gaussian noise conditions ($\Delta = -0.0003$ to $-0.0029$, all $p_{\text{Holm}} = 1.0000$), as the 50-epoch neural network exhibits intrinsic robustness to additive high-frequency noise.
6. **Descriptive Checkpoint Comparison**: Compared to the 3-epoch baseline (severe contrast unmitigated: **$0.0405 \pm 0.0370$** [corrected ground-truth reference]), the 50-epoch model demonstrates superior unmitigated baseline robustness ($0.8158 \pm 0.0677$), and both models converge to $>99\%$ of their respective clean baselines under CLAHE.
7. **Protocol Fidelity & Independent Audit**: All 19 pre-registered hypotheses across the 4 pre-specified families were audited independently with **zero discrepancies**. The original frozen baseline `checkpoints/best_model_frozen.pt` remains unmodified.

---

## 2. Pre-Registered Locked Protocol Specifications

All hyperparameter values were frozen prior to test execution based on validation set tuning (`split="val"`, 15 patients, 142 slices, zero test leakage) and preserved without deviation:

| Intervention Component | Locked Parameter | Frozen Value | Implementation Source |
| :--- | :--- | :--- | :--- |
| **CLAHE Preprocessing** | `clip_limit` | `0.01` | `src/preprocessing/restoration.py` |
| | `tile_grid_size` | `(16, 16)` | `src/preprocessing/restoration.py` |
| | Normalization | `to_unit_interval` | `src/preprocessing/restoration.py` |
| | Re-standardization | `re_standardize = True` | `src/preprocessing/restoration.py` |
| **Spatial Denoising** | Filter Type | Median Filter | `src/preprocessing/restoration.py` |
| | Kernel Size | `size = 3` ($3 \times 3$) | `src/preprocessing/restoration.py` |
| | Re-standardization | `re_standardize = False` | `src/preprocessing/restoration.py` |
| **Morphological Post-Proc** | Method | Per-Class LCC | `src/postprocessing/morphology.py` |
| | Minimum Area: RV | $30\text{ pixels}$ | `src/postprocessing/morphology.py` |
| | Minimum Area: MYO | $50\text{ pixels}$ | `src/postprocessing/morphology.py` |
| | Minimum Area: LV | $30\text{ pixels}$ | `src/postprocessing/morphology.py` |
| | Myocardium Hole Safeguard | `max_hole_size = 16` | `src/postprocessing/morphology.py` |

### Anatomical Hole-Filling Safeguard Verification
The myocardium forms a thick annular ring enclosing the left ventricular blood pool. Standard binary hole filling on the myocardium label fills the central LV chamber (~706 pixels on mid-ventricular slices), creating anatomical invalidity. The locked safeguard enforces `max_hole_size = 16` pixels. Small intramyocardial voids (mean 9 pixels) are closed while the large central LV cavity remains completely preserved.

---

## 3. Mathematical Analysis of CLAHE Contrast Invariance

In both the 3-epoch and 50-epoch mitigation evaluations, CLAHE yields identical slice-level and patient-level predictions across all contrast conditions ($\alpha \in \{0.75, 0.50, 0.25\}$). The exact mathematical derivation is as follows:

1. **Affine Contrast Corruption**:
   $$I_{\text{corrupt}} = \mu + \alpha (I - \mu) = \alpha I + (1 - \alpha)\mu$$
   where $\mu = \text{mean}(I)$ and $\alpha > 0$.
2. **Min-Max Mapping (`to_unit_interval`)**:
   $$\min(I_{\text{corrupt}}) = \alpha \min(I) + (1 - \alpha)\mu$$
   $$\max(I_{\text{corrupt}}) = \alpha \max(I) + (1 - \alpha)\mu$$
3. **Normalized Input to CLAHE**:
   $$I_{\text{unit}} = \frac{I_{\text{corrupt}} - \min(I_{\text{corrupt}})}{\max(I_{\text{corrupt}}) - \min(I_{\text{corrupt}})} = \frac{[\alpha I + (1 - \alpha)\mu] - [\alpha \min(I) + (1 - \alpha)\mu]}{[\alpha \max(I) + (1 - \alpha)\mu] - [\alpha \min(I) + (1 - \alpha)\mu]} = \frac{\alpha (I - \min(I))}{\alpha (\max(I) - \min(I))} = \frac{I - \min(I)}{\max(I) - \min(I)}$$
   Because the positive scalar $\alpha$ cancels out exactly in numerator and denominator, $I_{\text{unit}}(\text{corrupt}) \equiv I_{\text{unit}}(\text{clean})$.
4. **Empirical Numerical Audit**:
   Across all 15 test patients, the maximum absolute difference between `clean_baseline` + CLAHE and `contrast_reduction_severe` + CLAHE is:
   $$\max_{p \in \text{Patients}} |\text{Dice}_{\text{Clean+CLAHE}}(p) - \text{Dice}_{\text{Severe+CLAHE}}(p)| = \mathbf{0.00 \times 10^0}$$

This invariance is a direct mathematical consequence of min-max scaling prior to histogram equalization on linear contrast transforms.

---

## 4. Primary Hypotheses Results Across 4 Pre-Specified Families

The Phase 5 protocol pre-specifies **exactly 19 primary hypotheses** partitioned into 4 distinct families. Step-down Holm-Bonferroni correction ($\alpha = 0.05$) is applied independently within each family. Two-tailed paired Student's $t$-test serves as the primary confirmatory test, with the two-tailed Wilcoxon signed-rank test reported as non-parametric sensitivity analysis.

### Family 1: Morphological Post-Processing Alone Across All Conditions ($m=10$)

| Condition ID | Baseline Dice | Post Dice | Delta ($\Delta$) | $95\%\text{ CI}$ of $\Delta$ | Cohen's $d$ | Paired $t$ $p_{\text{raw}}$ | Paired $t$ $p_{\text{Holm}}$ | Wilcoxon $p_{\text{raw}}$ | Wilcoxon $p_{\text{Holm}}$ | Significance Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `clean_baseline` | 0.8958 | 0.8980 | +0.0022 | [-0.0031, +0.0076] | 0.2294 | 3.8933e-01 | 1.0000e+00 | 1.1585e-01 | 8.1096e-01 | Not Significant |
| `gaussian_noise_mild` | 0.8957 | 0.8978 | +0.0021 | [-0.0033, +0.0075] | 0.2181 | 4.1245e-01 | 1.0000e+00 | 1.7295e-01 | 9.4764e-01 | Not Significant |
| `gaussian_noise_mod` | 0.8948 | 0.8970 | +0.0022 | [-0.0033, +0.0076] | 0.2196 | 4.0932e-01 | 1.0000e+00 | 1.8234e-01 | 9.4764e-01 | Not Significant |
| `gaussian_noise_sev` | 0.8913 | 0.8942 | +0.0029 | [-0.0026, +0.0083] | 0.2905 | 2.7947e-01 | 1.0000e+00 | 2.4549e-01 | 9.4764e-01 | Not Significant |
| `gaussian_blur_mild` | 0.8860 | 0.8894 | +0.0034 | [-0.0037, +0.0105] | 0.2631 | 3.2542e-01 | 1.0000e+00 | 5.9739e-02 | 5.3765e-01 | Not Significant |
| `gaussian_blur_mod` | 0.8540 | 0.8617 | +0.0076 | [-0.0034, +0.0187] | 0.3824 | 1.6076e-01 | 1.0000e+00 | 7.5368e-02 | 6.0295e-01 | Not Significant |
| `gaussian_blur_sev` | 0.7775 | 0.7915 | +0.0140 | [-0.0006, +0.0286] | 0.5297 | 5.9421e-02 | 5.9421e-01 | 2.3130e-02 | 2.3130e-01 | Nominal Only (Unadjusted) |
| `contrast_reduction_mild` | 0.8956 | 0.8982 | +0.0026 | [-0.0033, +0.0085] | 0.2401 | 3.6811e-01 | 1.0000e+00 | 1.5794e-01 | 9.4764e-01 | Not Significant |
| `contrast_reduction_mod` | 0.8935 | 0.8949 | +0.0014 | [-0.0033, +0.0061] | 0.1661 | 5.3048e-01 | 1.0000e+00 | 9.5277e-01 | 9.5277e-01 | Not Significant |
| `contrast_reduction_sev` | 0.8158 | 0.8156 | -0.0002 | [-0.0019, +0.0016] | -0.0492 | 8.5174e-01 | 1.0000e+00 | 2.7195e-01 | 9.4764e-01 | Not Significant |

### Family 2: CLAHE Preprocessing Alone on Contrast Conditions ($m=3$)

| Condition ID | Baseline Dice | CLAHE Dice | Delta ($\Delta$) | $95\%\text{ CI}$ of $\Delta$ | Cohen's $d$ | Paired $t$ $p_{\text{raw}}$ | Paired $t$ $p_{\text{Holm}}$ | Wilcoxon $p_{\text{raw}}$ | Wilcoxon $p_{\text{Holm}}$ | Significance Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `contrast_reduction_sev` | 0.8158 | 0.8896 | **+0.0738** | [+0.0354, +0.1121] | 1.0647 | 1.0336e-03 | **3.1008e-03** | 3.0518e-04 | **9.1553e-04** | **Significant** |
| `contrast_reduction_mod` | 0.8935 | 0.8896 | -0.0039 | [-0.0121, +0.0044] | -0.2616 | 3.2812e-01 | 3.2812e-01 | 3.3026e-01 | 3.3026e-01 | Not Significant |
| `contrast_reduction_mild` | 0.8956 | 0.8896 | -0.0060 | [-0.0124, +0.0004] | -0.5232 | 6.2235e-02 | 1.2447e-01 | 1.3538e-01 | 2.7075e-01 | Not Significant |

### Family 3: Combined Pipeline (CLAHE + Post-Processing) on Contrast ($m=3$)

| Condition ID | Baseline Dice | Comb Dice | Delta ($\Delta$) | $95\%\text{ CI}$ of $\Delta$ | Cohen's $d$ | Paired $t$ $p_{\text{raw}}$ | Paired $t$ $p_{\text{Holm}}$ | Wilcoxon $p_{\text{raw}}$ | Wilcoxon $p_{\text{Holm}}$ | Significance Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `contrast_reduction_sev` | 0.8158 | 0.8921 | **+0.0763** | [+0.0375, +0.1150] | 1.0907 | 8.4967e-04 | **2.5490e-03** | 1.8311e-04 | **5.4932e-04** | **Significant** |
| `contrast_reduction_mod` | 0.8935 | 0.8921 | -0.0014 | [-0.0074, +0.0046] | -0.1297 | 6.2332e-01 | 6.2332e-01 | 5.6140e-01 | 9.7742e-01 | Not Significant |
| `contrast_reduction_mild` | 0.8956 | 0.8921 | -0.0035 | [-0.0093, +0.0023] | -0.3383 | 2.1122e-01 | 4.2243e-01 | 4.8871e-01 | 9.7742e-01 | Not Significant |

### Family 4: Combined Pipeline (Median + Post-Processing) on Noise ($m=3$)

| Condition ID | Baseline Dice | Comb Dice | Delta ($\Delta$) | $95\%\text{ CI}$ of $\Delta$ | Cohen's $d$ | Paired $t$ $p_{\text{raw}}$ | Paired $t$ $p_{\text{Holm}}$ | Wilcoxon $p_{\text{raw}}$ | Wilcoxon $p_{\text{Holm}}$ | Significance Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `gaussian_noise_mild` | 0.8957 | 0.8928 | -0.0029 | [-0.0112, +0.0055] | -0.1896 | 4.7482e-01 | 1.0000e+00 | 8.3252e-02 | 2.1899e-01 | Not Significant |
| `gaussian_noise_mod` | 0.8948 | 0.8922 | -0.0026 | [-0.0116, +0.0064] | -0.1607 | 5.4378e-01 | 1.0000e+00 | 7.2998e-02 | 2.1899e-01 | Not Significant |
| `gaussian_noise_sev` | 0.8913 | 0.8910 | -0.0003 | [-0.0111, +0.0106] | -0.0133 | 9.5959e-01 | 1.0000e+00 | 4.8871e-01 | 4.8871e-01 | Not Significant |

---

## 5. Negative Controls Evaluation

To guard against false-positive intervention efficacy and evaluate off-target side effects, negative control experiments were evaluated across unaffected and non-target conditions:

| Negative Control Arm | Tested Condition | Unmitigated Dice | Treated Dice | Delta ($\Delta$) | Paired $t$ $p_{\text{raw}}$ | Wilcoxon $p_{\text{raw}}$ | Classification | Clinical Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **CLAHE Alone** | `clean_baseline` | 0.8958 | 0.8896 | -0.0062 | 2.2405e-02 | 2.5574e-02 | Off-Target Clean | Minor nominal drop; histogram equalization introduces subtle intensity distortion on clean scans. |
| **CLAHE Alone** | `gaussian_noise_mild` | 0.8957 | 0.8880 | -0.0076 | 7.2285e-02 | 9.4604e-02 | Off-Target Noise | Noise amplification; CLAHE boosts background noise variance. |
| **CLAHE Alone** | `gaussian_noise_mod` | 0.8948 | 0.8839 | -0.0109 | 6.6276e-02 | 7.2998e-02 | Off-Target Noise | Progressive noise amplification with increasing noise severity. |
| **CLAHE Alone** | `gaussian_noise_sev` | 0.8913 | 0.8744 | -0.0169 | 2.6467e-02 | 2.5574e-02 | Off-Target Noise | Significant nominal degradation; CLAHE should NOT be applied to noisy scans without prior denoising. |
| **CLAHE Alone** | `gaussian_blur_mild` | 0.8860 | 0.8829 | -0.0031 | 4.6065e-01 | 5.9949e-01 | Off-Target Blur | Negligible effect on mild blur. |
| **CLAHE Alone** | `gaussian_blur_mod` | 0.8540 | 0.8598 | +0.0058 | 3.8646e-01 | 7.1973e-01 | Off-Target Blur | Slight edge enhancement, not statistically significant. |
| **CLAHE Alone** | `gaussian_blur_sev` | 0.7775 | 0.8051 | +0.0276 | 3.9682e-02 | 5.5359e-02 | Off-Target Blur | Nominal edge restoration; sharpens attenuated blurred boundaries. |
| **Combined Contrast**| `clean_baseline` | 0.8958 | 0.8921 | -0.0037 | 1.1760e-01 | 2.0776e-01 | Clean Pipeline | Mild nominal attenuation; post-processing partially compensates for CLAHE distortion. |
| **Combined Noise** | `clean_baseline` | 0.8958 | 0.8929 | -0.0029 | 4.6000e-01 | 8.3252e-02 | Clean Pipeline | Median filter blurs sharp myocardial edges slightly on pristine inputs. |

---

## 6. Checkpoint Performance Comparison: 50-Epoch vs 3-Epoch

> [!IMPORTANT]
> **Methodological Framing**: The following comparison is strictly a **descriptive comparison of checkpoint performance** between two distinct training artifacts: `checkpoints/best_model_frozen.pt` (3-epoch baseline, random initialization) and `checkpoints/50epochs/best_model_50epochs.pt` (50-epoch run, Epoch 44, separate random initialization). Because training duration, optimizer states, and initialization seeds differed, this comparison does **not** assert a causal claim regarding epoch count alone.

Historical 3-epoch values below use the verified ground-truth test evaluation: for severe contrast unmitigated, the ground-truth reference value is **$0.0405 \pm 0.0370$** (correcting the historical documentation transcription error of $0.5794$).

| Evaluation Condition / Mitigation Arm | 3-Epoch Checkpoint (`best_model_frozen.pt`) | 50-Epoch Checkpoint (`best_model_50epochs.pt`) | Absolute Difference ($\Delta_{\text{50e - 3e}}$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline (Raw Input)** | $0.7355 \pm 0.0911$ | $0.8958 \pm 0.0372$ | **+0.1603** | +21.8% |
| Clean + Post-Processing Alone | $0.7415 \pm 0.0898$ | $0.8980 \pm 0.0357$ | +0.1565 | +21.1% |
| Clean + CLAHE Alone | $0.7304 \pm 0.0934$ | $0.8896 \pm 0.0429$ | +0.1592 | +21.8% |
| Clean + Combined (CLAHE + Post) | $0.7395 \pm 0.0898$ | $0.8921 \pm 0.0371$ | +0.1526 | +20.6% |
| **Severe Contrast Unmitigated ($\alpha=0.25$)** | **$0.0405 \pm 0.0370$** | **$0.8158 \pm 0.0677$** | **+0.7753** | **+1914.3%** |
| Severe Contrast + Post-Processing Alone | $0.0394 \pm 0.0360$ | $0.8156 \pm 0.0682$ | +0.7762 | +1970.1% |
| Severe Contrast + CLAHE Alone | $0.7304 \pm 0.0934$ | $0.8896 \pm 0.0429$ | +0.1592 | +21.8% |
| Severe Contrast + Combined (CLAHE + Post) | $0.7395 \pm 0.0898$ | $0.8921 \pm 0.0371$ | +0.1526 | +20.6% |
| **Severe Blur Unmitigated ($\sigma=2.50$)** | $0.6718 \pm 0.1065$ | $0.7775 \pm 0.0800$ | **+0.1057** | +15.7% |
| Severe Blur + Post-Processing Alone | $0.6747 \pm 0.1066$ | $0.7915 \pm 0.0754$ | +0.1168 | +17.3% |
| **Severe Noise Unmitigated ($\sigma=0.10$)** | $0.7313 \pm 0.0926$ | $0.8913 \pm 0.0393$ | **+0.1600** | +21.9% |
| Severe Noise + Combined (Median + Post) | $0.7315 \pm 0.0925$ | $0.8910 \pm 0.0396$ | +0.1595 | +21.8% |

### Key Observations from Checkpoint Comparison:
1. **Dramatic Unmitigated Contrast Robustness**: While the 3-epoch model collapsed completely under severe contrast ($0.0405$ Dice), the 50-epoch checkpoint retains substantial anatomical recognition ($0.8158$ Dice) even without any preprocessing.
2. **Universal CLAHE Rescue Proportion**: In both models, CLAHE restores performance to **$99.3\%$** of their respective clean baselines ($0.7304 / 0.7355 = 99.3\%$ for 3-epoch; $0.8896 / 0.8958 = 99.3\%$ for 50-epoch).
3. **Severe Blur Gap**: The 50-epoch model is markedly more resilient to boundary degradation (+0.1057 Dice), and post-processing provides an additional $+0.0140$ boost.

---

## 7. Deliverables & Artifact Inventory

All deliverables have been generated in `experiments/phase5_validation_locked_mitigation/` and mirrored to `reports/phase5_50epochs/`:

| Artifact File | Format | Record Count / Dimensions | Description |
| :--- | :---: | :---: | :--- |
| `mitigation_results.json` | JSON | 504 KB | Full nested dictionary with cryptographic cache provenance, all condition evaluations, slice metrics, and per-patient breakdowns. |
| `mitigation_results.csv` | CSV | 38 rows | Master tabular comparison with audited raw and Holm-adjusted $p$-values and FWER significance. |
| `primary_hypotheses_results.csv` | CSV | 19 rows | Pre-registered primary hypothesis matrix across 4 families. |
| `negative_control_results.csv` | CSV | 9 rows | Separated negative control evaluations (off-target preprocessing). |
| `patient_level_results.csv` | CSV | 630 rows | Patient-level 3D Dice (RV, MYO, LV, Macro FG) for all 15 patients across conditions and arms. |
| `slice_level_results.csv` | CSV | 5,502 rows | 2D slice-level Dice scores for all 131 test slices across evaluated configurations. |
| `statistical_tests.csv` | CSV | 19 rows | Detailed test statistics: degrees of freedom ($df=14$), mean difference, standard error, $95\%\text{ CI}$, Cohen's $d$, $t$-stat, $W$-stat. |
| `holm_adjusted_results.csv` | CSV | 38 rows | Detailed step-down audit log showing rank $i$, multiplier $(m-i+1)$, raw $p$, running max, and capped Holm $p$. |
| `reproducibility_manifest.json` | JSON | 2.1 KB | Machine environment, library versions, seed, checkpoint SHA-256, config SHA-256, manifest SHA-256. |
| `checkpoint_sha256.txt` | TXT | 64 chars | Verified SHA-256 hash string (`a4c320f1...`). |
| `cohort_manifest.json` | JSON | 413 bytes | Held-out test cohort patient list and spatial resolution specifications. |
| `experiment_config_snapshot.json` | JSON | 3.5 KB | Snapshot of model YAML parameters and locked mitigation settings. |
| `mitigation_comparison.png` | PNG | 1800 x 500 px | Publication-quality 3-panel summary plot (Severe contrast rescue, contrast curve, clean interventions). |
| `visualizations/class_wise_mitigation_effect.png` | PNG | 1800 x 500 px | Per-class breakdown (RV, MYO, LV) comparing Unmitigated, CLAHE Alone, and Combined pipelines. |
| `representative_overlays/overlay_patient001_slice04.png` | PNG | 1600 x 1500 px | 4x4 panel overlay: Input MRI, Ground Truth, Unmitigated Prediction, Mitigated Prediction. |
| `representative_overlays/overlay_patient004_slice05.png` | PNG | 1600 x 1500 px | 4x4 panel overlay for mid-ventricular slice. |
| `representative_overlays/overlay_patient038_slice05.png` | PNG | 1600 x 1500 px | 4x4 panel overlay for challenging apical anatomy. |
| `phase5_independent_audit.json` | JSON | 448 bytes | Output of automated verification audit (0 discrepancies, PASS). |
| `phase5_independent_audit_report.md` | MD | 641 bytes | Markdown audit verification summary. |

---

## 8. Methodological Limitations

1. **Synthetic vs Real Physical Corruptions**: Corruptions were generated mathematically via Gaussian noise, Gaussian blur, and affine contrast scaling. Although standard benchmarks in robust medical vision, real clinical scanner artifacts (e.g., patient respiratory ghosting, cardiac arrhythmia mis-triggering, Gibbs ringing, dielectric shimming shading) exhibit complex k-space phase distortions not fully captured by spatial domain transforms.
2. **2D Slice-Wise Processing Without 3D Inter-Slice Context**: Segmentation inference and post-processing operate slice-by-slice in the 2D short-axis plane. 3D through-plane spatial consistency (longitudinal continuity from base to apex) is not enforced, which can lead to slice-to-slice volume jitter.
3. **Single-Center Cohort Constraints**: The ACDC dataset originates from a single clinical center (Dijon University Hospital) with standardized Siemens 1.5T and 3.0T scanners. Cross-vendor generalization to GE or Philips scanners with different raw reconstruction algorithms was not evaluated.
4. **End-Diastolic Frame Restriction**: Experiments focused strictly on end-diastolic frames (`frame01.nii.gz`), representing maximum cardiac ventricular volume. End-systolic frames, characterized by contracted myocardium and altered chamber blood pool contrast, were not evaluated in this protocol.
5. **Affine Contrast Invariance Boundary Conditions**: The strict mathematical invariance of CLAHE demonstrated in Section 3 applies to affine transformations $I_{\text{corrupt}} = \alpha I + \beta$. Non-linear intensity distortions (e.g., $B_1$ radiofrequency transmit field inhomogeneity bias fields, non-linear gamma mappings) would violate this affine property.
6. **Validation Set Tuning Size**: Hyperparameters were tuned on a 15-patient validation split. While zero test data leaked into hyperparameter selection, tuning on larger cohorts may discover finer-grained optimal post-processing thresholds.

---

## 9. Independent Audit & Protocol Verification Conclusion

- **Protocol Audit**: 100% compliant with pre-registered specifications.
- **Hypothesis Matrix**: Exactly 19 primary hypothesis tests across 4 pre-registered families verified.
- **Negative Controls**: Evaluated and reported separately from primary hypotheses.
- **Baseline Integrity**: Original frozen 3-epoch baseline checkpoint `checkpoints/best_model_frozen.pt` (SHA-256: `e7bfa884...`) and historical reports remain 100% preserved and untouched.
- **Test Suite**: 103 of 103 unit tests pass cleanly.

<!-- GOAL_COMPLETE -->
