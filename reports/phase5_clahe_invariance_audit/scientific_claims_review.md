# Phase 5 Scientific Claims Review: Forensic Invariance Audit

**Investigation**: Causal Attribution and Invariance Audit of CLAHE Contrast Mitigation  
**Experiment ID**: `phase5_clahe_invariance_audit`  
**Date**: November 5, 2026  
**Auditor**: Senior Medical Imaging Research Scientist & Experimental Auditor  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  

---

## 1. Context of the Audit

In Phase 5, the evaluation report attributed a $+0.0738$ increase in patient 3D macro foreground Dice under severe contrast reduction ($\alpha = 0.25$) to the restorative capabilities of **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**.

However, forensic inspection revealed that the `apply_clahe` function calls `to_unit_interval` as its first step. `to_unit_interval` applies deterministic min-max normalization:
$$I_{\text{unit}} = \frac{I - \min(I)}{\max(I) - \min(I)}$$

For any positive affine transformation $I_{\text{corrupt}} = \alpha I + (1 - \alpha)\mu$ with $\alpha > 0$:
$$\frac{I_{\text{corrupt}} - \min(I_{\text{corrupt}})}{\max(I_{\text{corrupt}}) - \min(I_{\text{corrupt}})} = \frac{\alpha (I - \min(I))}{\alpha (\max(I) - \min(I))} = \frac{I - \min(I)}{\max(I) - \min(I)}$$

This forensic review re-evaluates all empirical claims made in Phase 5 to determine whether the reported recovery reflects genuine adaptive computer vision robustness or a methodological artifact of input re-normalization.

---

## 2. Item-by-Item Review of Original Phase 5 Scientific Claims

### Claim 1: "CLAHE restores cardiac MRI segmentation performance under severe contrast reduction (+0.0738 Dice, p < 0.005)."
- **Audited Status**: **MISATTRIBUTED (Methodological Artifact / Confounded)**.
- **Empirical Evidence**:
  - Max absolute difference between Clean after `to_unit_interval` ($C$) and Severe Contrast after `to_unit_interval` ($D$): **$2.38 \times 10^{-7}$** (floating-point machine epsilon).
  - Max absolute difference between Clean after CLAHE ($E$) and Severe Contrast after CLAHE ($F$): **$0.00 \times 10^{0}$** (strictly identical arrays).
  - Pure Z-Score Standardization alone ($z(I_{\text{corrupt}})$) achieves **$0.8958 \pm 0.0372$** (a $+0.0800$ rescue, matching the clean baseline $100.0\%$).
  - Severe Contrast + CLAHE achieves **$0.8896 \pm 0.0429$**.
- **Scientific Reality**: The recovery from $0.8158$ to $0.8896$ was **100% produced by linear normalization** canceling out $\alpha = 0.25$. CLAHE itself was not only unnecessary; it actually **reduced** performance relative to pure normalization ($0.8896$ vs $0.8958$, a net loss of $-0.0062$). The claim that CLAHE's adaptive histogram redistributions restored the segmentation is scientifically false.

---

### Claim 2: "Combined CLAHE + Post-Processing provides the optimal rescue (+0.0763 Dice)."
- **Audited Status**: **MISATTRIBUTED**.
- **Empirical Evidence**:
  - Combined pipeline achieves $0.8921 \pm 0.0371$.
  - Pure Z-score normalization alone achieves $0.8958 \pm 0.0372$.
  - Pure Z-score normalization + Post-processing achieves $0.8980 \pm 0.0357$.
- **Scientific Reality**: The $+0.0763$ delta over unmitigated ($0.8158$) is entirely driven by normalization canceling the corruption, followed by a $+0.0025$ boost from morphological cleanup. CLAHE contributed negatively ($-0.0059$) to this pipeline.

---

### Claim 3: "CLAHE achieves 99.3% recovery back to the clean baseline."
- **Audited Status**: **MATHEMATICALLY TRIVIAL (Spurious Metric)**.
- **Empirical Evidence**:
  - Clean Baseline: $0.8958$.
  - Clean + CLAHE: $0.8896$.
  - Severe Contrast + CLAHE: $0.8896$.
  - Ratio: $0.8896 / 0.8958 = 99.31\%$.
- **Scientific Reality**: Because `to_unit_interval` makes the clean and contrast-corrupted inputs mathematically identical, Severe Contrast + CLAHE produces the exact same score as Clean + CLAHE ($0.8896$). The $99.3\%$ ratio does not measure the fraction of corruption overcome; it measures the degree to which CLAHE perturbed clean images ($0.8896 / 0.8958$).

---

### Claim 4: "Statistically significant paired t-test and Wilcoxon tests prove CLAHE's efficacy."
- **Audited Status**: **INVALID CAUSAL INFERENCE**.
- **Empirical Evidence**:
  - The paired test compared `Unmitigated Corrupted` (which had $\sigma \approx 0.25$, non-standard dynamic range) against `Mitigated Corrupted` (which underwent min-max scaling + CLAHE + z-score standardization).
- **Scientific Reality**: The statistical test confirms a difference between unstandardized and standardized pipelines ($p = 0.0031$). However, the statistical hypothesis test failed to isolate CLAHE as the causal variable because the experimental arm simultaneously changed normalization and applied CLAHE. This is a classic confounding error.

---

### Claim 5: "Morphological post-processing provides general, corruption-agnostic improvement across all conditions."
- **Audited Status**: **VALID (With Pre-Registered FWER Qualification)**.
- **Empirical Evidence**:
  - Post-processing yields positive deltas across all 10 conditions ($\Delta \in [+0.0014, +0.0140]$).
  - Maximum effect on Severe Blur: $0.7775 \to 0.7915$ ($\Delta = +0.0140$, $+1.40\%$, paired $t = 2.0514$, nominal $p_{\text{raw}} = 0.0231$ on Wilcoxon).
- **Scientific Reality**: Morphological cleanup operates solely on the discrete categorical segmentation mask, removing spurious islands $<30$ px and preserving the anatomical LV cavity with `max_hole_size = 16`. Under step-down Holm-Bonferroni correction ($m=10$), these effects are properly classified as *Nominal Only (Unadjusted)*, but the mechanism is genuine and corruption-independent.

---

### Claim 6: "Median filtering fails to improve Gaussian noise robustness on the 50-epoch model."
- **Audited Status**: **VALID & SCIENTIFICALLY SOUND**.
- **Empirical Evidence**:
  - Combined Median + Post on noise: $\Delta = -0.0003$ to $-0.0029$, all $p_{\text{Holm}} = 1.0000$.
- **Scientific Reality**: Deep feature representations in `LightweightUNet2D` trained for 50 epochs already possess spatial pooling invariance to high-frequency additive noise ($0.8958 \to 0.8913$ under severe noise). Spatial median filtering ($3 \times 3$) blurs true myocardial borders, degrading rather than improving accuracy.

---

### Claim 7: "Negative controls demonstrate off-target CLAHE degradation on clean scans and noise."
- **Audited Status**: **VALID & CRITICAL**.
- **Empirical Evidence**:
  - CLAHE on Clean Baseline: $0.8958 \to 0.8896$ ($\Delta = -0.0062$, $p = 0.0224$).
  - CLAHE on Severe Noise: $0.8913 \to 0.8744$ ($\Delta = -0.0169$, $p = 0.0265$).
- **Scientific Reality**: Histogram equalization assumes poor contrast distribution. On well-calibrated MRI scans, it amplifies background thermal noise and artificially shifts blood-pool-to-myocardium gradients.

---

## 3. Summary of Status Matrix

| Original Claim | Audited Status | Primary Causal Factor |
| :--- | :---: | :--- |
| CLAHE rescues severe contrast segmentation | **MISATTRIBUTED** | Linear Min-Max Normalization in `to_unit_interval` |
| Combined pipeline achieves optimal contrast rescue | **MISATTRIBUTED** | Normalization + Morphological Post-Processing |
| CLAHE achieves 99.3% recovery | **TRIVIAL ARTIFACT** | Invariance of min-max scaling under affine scaling |
| Statistical tests confirm CLAHE efficacy | **CONFOUNDED** | Confounds normalization with CLAHE |
| Morphological post-processing cleans masks | **VALID** | Per-class LCC with myocardial cavity safeguard |
| Median filtering ineffective on noise | **VALID** | Network intrinsic noise robustness; edge blurring |
| Negative controls show off-target degradation | **VALID** | Gradient distortion on clean; noise amplification |
