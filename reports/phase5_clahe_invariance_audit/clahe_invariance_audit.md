# Forensic Audit Report: CLAHE Contrast Invariance & Causal Attribution

**Investigation Title**: Forensic Investigation of Min-Max Normalization Invariance and Causal Attribution of CLAHE Mitigation in Cardiac MRI Segmentation  
**Experiment Identifier**: `phase5_clahe_invariance_audit`  
**Date**: November 5, 2026  
**Auditor / Investigator**: Senior Medical Imaging Research Scientist, Computer Vision Engineer & Experimental Auditor  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Evaluated Architecture**: `LightweightUNet2D` (1,963,860 parameters)  
**Evaluated Model Checkpoint**: `checkpoints/50epochs/best_model_50epochs.pt` (Epoch 44, SHA-256: `a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38`)  
**Dataset & Cohort**: ACDC Held-Out Test Cohort (15 patients, 131 end-diastolic 2D slices)  

---

## 1. Executive Summary & Objective

The objective of this forensic investigation is to resolve a fundamental scientific question arising from the Phase 5 experimental results:

> **Core Research Question**:  
> In Phase 5, CLAHE was reported to rescue segmentation accuracy under severe contrast reduction from $0.8158$ to $0.8896$ (+0.0738 Dice, $p < 0.005$).  
> Did CLAHE genuinely restore segmentation robustness through adaptive histogram redistributions, or was the synthetic contrast corruption already canceled out by the min-max normalization (`to_unit_interval`) preceding CLAHE?

### Definitive Conclusion:
1. **Mathematical Invariance**: Min-max normalization ($I_{\text{norm}} = (I - \min)/( \max - \min)$) is analytically invariant to any positive affine contrast transformation $I_{\text{corrupt}} = \alpha I + (1 - \alpha)\mu$ with $\alpha > 0$. The factor $\alpha$ divides out completely in both numerator and denominator.
2. **Tensor-Level Evidence**: Across all 131 held-out test slices, the clean image after `to_unit_interval` ($C$) and the severe contrast image after `to_unit_interval` ($D$) are **strictly identical down to single-precision machine epsilon** ($\max |C - D| = \mathbf{2.38 \times 10^{-7}}$, mean $|C - D| = \mathbf{6.09 \times 10^{-9}}$).
3. **CLAHE Output Identity**: Because CLAHE receives identical inputs for clean and contrast-corrupted images, the enhanced arrays ($E$ and $F$) are **literally zero-difference identical** ($\max |E - F| = \mathbf{0.00 \times 10^0}$). Model predictions on clean+CLAHE and severe contrast+CLAHE exhibit **$100.00\%$ pixel agreement** across all 15 patients.
4. **Causal Attribution**: The $+0.0738$ recovery was **100% caused by input re-normalization**. In fact, applying pure standard z-score normalization alone without CLAHE achieves **$0.8958 \pm 0.0372$** (a $+0.0800$ recovery, restoring $100.0\%$ of the clean baseline). CLAHE itself added no positive value; rather, it imposed a net **$-0.0062$ performance penalty** due to artificial intensity distortion.

---

## 2. Exact Preprocessing Pipelines

The exact sequence of operations used in each inference regime was traced through the codebase:

```
[ACDC Raw NIfTI] 
       │
       ▼ (1. Center crop/pad to 256x256)
       ▼ (2. Percentile clipping [0.5, 99.5])
       ▼ (3. Per-slice z-score standardization: mu=0, sigma=1)
[DataLoader Output: I_clean] (Shape: 1x256x256, float32, mean ~ 0.0, std ~ 1.0)
```

### Regime A: Clean Baseline (Unmitigated)
- **Source**: `scripts/evaluate_mitigation_test.py` lines 88-120
- **Path**: `I_clean` $\to$ `model(I_clean)`
- **Input to Model**: $I_{\text{clean}}$ with $\mu \approx 0.0, \sigma \approx 1.0$.

### Regime B: Severe Contrast Reduction (Unmitigated)
- **Source**: `src/corruptions/contrast.py` lines 42-46
- **Operation**:
  ```python
  mean = x.mean(dim=(-2, -1), keepdim=True)
  corrupted = mean + factor * (x - mean)  # factor = 0.25
  ```
- **Path**: `I_clean` $\to$ `corrupt_img` $\to$ `model(corrupt_img)`
- **Input to Model**: $I_{\text{corrupt}}$ with $\mu \approx 0.0, \sigma \approx 0.25$.
- **Mechanism of Failure**: The input dynamic range is compressed by $4\times$. Because `LightweightUNet2D` has batch normalization layers calibrated to unit variance, attenuated input dynamic range dampens activations across all convolutional layers, causing foreground Dice to drop from $0.8958$ to $0.8158$.

### Regime C: Severe Contrast + CLAHE Mitigation
- **Source**: `src/preprocessing/restoration.py` lines 46-115
- **Sequence**:
  1. `img_01, min_val, max_val = to_unit_interval(corrupt_img)`:
     Min-max scaling to $[0.0, 1.0]$.
  2. `exposure.equalize_adapthist(img_01, kernel_size=(16, 16), clip_limit=0.01, nbins=256)`:
     Tile-based contextual histogram equalization.
  3. `z_score_standardize(enhanced)`:
     Re-standardization to $\mu = 0.0, \sigma = 1.0$.
  4. `model(enhanced)`: Forward pass.

---

## 3. Mathematical Verification of Invariance

### Theorem: Invariance of Min-Max Scaling under Linear Contrast Corruption
Let $I \in \mathbb{R}^{H \times W}$ denote a 2D image slice with spatial mean $\mu = \frac{1}{HW}\sum_{i,j} I_{i,j}$, minimum $I_{\min} = \min_{i,j} I_{i,j}$, and maximum $I_{\max} = \max_{i,j} I_{i,j}$.

The synthetic contrast corruption applied in the project is defined as:
$$I'(\alpha) = \mu + \alpha (I - \mu) = \alpha I + (1 - \alpha)\mu, \quad \alpha \in (0, 1]$$

Because $\alpha > 0$ is a strictly positive scalar and $(1 - \alpha)\mu$ is a spatially constant offset, the ordering of pixel intensities is strictly preserved. Consequently:
$$I'_{\min} = \min_{i,j} I'_{i,j} = \alpha I_{\min} + (1 - \alpha)\mu$$
$$I'_{\max} = \max_{i,j} I'_{i,j} = \alpha I_{\max} + (1 - \alpha)\mu$$

The intensity range of the corrupted image is:
$$R(I') = I'_{\max} - I'_{\min} = [\alpha I_{\max} + (1 - \alpha)\mu] - [\alpha I_{\min} + (1 - \alpha)\mu] = \alpha (I_{\max} - I_{\min}) = \alpha R(I)$$

Now consider the min-max normalization operator $T(X) = \frac{X - X_{\min}}{R(X)}$ implemented in `to_unit_interval`:
$$T(I') = \frac{I' - I'_{\min}}{R(I')} = \frac{[\alpha I + (1 - \alpha)\mu] - [\alpha I_{\min} + (1 - \alpha)\mu]}{\alpha R(I)} = \frac{\alpha (I - I_{\min})}{\alpha R(I)} = \frac{I - I_{\min}}{R(I)} \equiv T(I)$$

**Conclusion**: For every pixel $(i, j)$, $T(I')_{i,j} = T(I)_{i,j}$. The normalized tensor is mathematically invariant to $\alpha$ for all $\alpha > 0$.

### Scope & Boundary Conditions of the Proof:
1. **Affine Requirement**: The invariance holds because $I' = \alpha I + \beta$ is an affine map with positive slope $\alpha > 0$.
2. **Absence of Clipping**: In `src/corruptions/contrast.py`, no intensity clamping or integer rounding is performed on the normalized float32 tensor. If threshold saturation (clipping) were applied, pixel ordering would be destroyed and $T(I')$ would no longer equal $T(I)$.
3. **Per-Slice Normalization**: The invariance holds because both `contrast.py` and `to_unit_interval` compute spatial statistics per 2D slice.

---

## 4. Tensor-Level Forensic Audit

The forensic audit evaluated all 131 test slices across the 15 held-out test patients at each stage:
- **Stage A**: Clean image from DataLoader
- **Stage B**: Severe contrast-corrupted image ($\alpha = 0.25$)
- **Stage C**: Clean image after `to_unit_interval`
- **Stage D**: Severe contrast image after `to_unit_interval`
- **Stage E**: Clean image after CLAHE
- **Stage F**: Severe contrast image after CLAHE

### Empirical Comparison Matrix

| Stage Comparison | Measured Quantity | Mean Across Cohort | Min Across Cohort | Max Across Cohort | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A vs B** (Raw Clean vs Corrupted) | Max Abs Diff | $3.5821$ | $1.2140$ | $8.4520$ | Dynamic range compressed by $4\times$ |
| | Mean Abs Diff | $0.5642$ | $0.2105$ | $0.9412$ | Substantial intensity deviation |
| | Pearson Correlation | $1.000000$ | $1.000000$ | $1.000000$ | Perfect collinearity |
| **C vs D** (Clean Unit vs Corrupt Unit) | **Max Abs Diff** | **$6.09 \times 10^{-9}$** | **$0.00 \times 10^0$** | **$2.38 \times 10^{-7}$** | **Identical to float32 machine epsilon** |
| | **Mean Abs Diff** | **$3.12 \times 10^{-10}$** | **$0.00 \times 10^0$** | **$1.85 \times 10^{-8}$** | **Zero meaningful difference** |
| | **% Identical Pixels (tol 1e-5)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **$100\%$ pixel-level identity** |
| | Pearson Correlation | $1.000000$ | $1.000000$ | $1.000000$ | Collinear and identical |
| **E vs F** (Clean CLAHE vs Corrupt CLAHE) | **Max Abs Diff** | **$0.00 \times 10^0$** | **$0.00 \times 10^0$** | **$0.00 \times 10^0$** | **Literally identical arrays** |
| | **Mean Abs Diff** | **$0.00 \times 10^0$** | **$0.00 \times 10^0$** | **$0.00 \times 10^0$** | Zero deviation |
| | **% Identical Pixels (exact)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | Zero pixel divergence |
| **Pred E vs Pred F** (Model Output) | **Pixel Agreement** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **All 8,585,216 prediction pixels match** |
| | Foreground Dice | **$1.000000$** | **$1.000000$** | **$1.000000$** | Perfect segmentation equivalence |

---

## 5. Patient-Level Prediction Evidence

Across all 15 held-out test patients, patient 3D macro foreground Dice scores were computed under:
1. `Clean Baseline`
2. `Severe Contrast Unmitigated`
3. `Clean + CLAHE`
4. `Severe Contrast + CLAHE`
5. `Severe Contrast + Z-Score Alone` (simple re-standardization without CLAHE)

### Patient-Level Comparison Table

| Patient ID | Clean Baseline | Severe Unmitigated | Clean + CLAHE | Severe + CLAHE | Abs Diff (Clean vs Sev CLAHE) | Severe + Z-Score Alone | Abs Diff (Clean vs Sev Z-Score) |
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

---

## 6. Causal Decomposition Analysis

To disentangle the effect of **linear normalization** from the effect of **CLAHE**, we decompose the total recovery:

$$\Delta_{\text{total}} = \text{Dice}(\text{Corrupt + CLAHE}) - \text{Dice}(\text{Corrupt Unmitigated}) = 0.8896 - 0.8158 = \mathbf{+0.0738}$$

Decomposing into two sequential operations:
1. **Component 1: Linear Normalization Alone**:
   $$\Delta_{\text{norm}} = \text{Dice}(\text{Corrupt + Z-Score}) - \text{Dice}(\text{Corrupt Unmitigated}) = 0.8958 - 0.8158 = \mathbf{+0.0800}$$
2. **Component 2: CLAHE on Top of Normalization**:
   $$\Delta_{\text{CLAHE}} = \text{Dice}(\text{Corrupt + CLAHE}) - \text{Dice}(\text{Corrupt + Z-Score}) = 0.8896 - 0.8958 = \mathbf{-0.0062}$$

$$\Delta_{\text{total}} = \Delta_{\text{norm}} + \Delta_{\text{CLAHE}} = (+0.0800) + (-0.0062) = +0.0738$$

### Scientific Meaning:
- **$108.4\%$ of the recovery** ($+0.0800 / +0.0738$) is attributable to simple dynamic range re-scaling.
- **$-8.4\%$ of the recovery** ($-0.0062 / +0.0738$) is an adverse penalty introduced by CLAHE.
- CLAHE did **not** contribute to overcoming contrast corruption. Normalization eliminated the corruption, and CLAHE subsequently distorted the clean tissue boundaries.

---

## 7. What Can Legitimate Research Papers State?

### Statements That Must Be Withdrawn or Corrected:
1. *Do not claim*: "CLAHE provides effective mitigation against cardiac MRI contrast degradation."
2. *Do not claim*: "Adaptive histogram equalization restores anatomical features lost during contrast reduction."
3. *Do not claim*: "Statistical significance ($p < 0.005$) validates CLAHE's clinical utility for low-contrast scans."

### Statements That Are Scientifically Legitimate:
1. *Valid finding*: "In standard 2D medical imaging inference, neural networks calibrated on zero-mean, unit-variance data exhibit sensitivity to unnormalized input dynamic range compression ($\Delta = -0.0800$ Dice under $\alpha=0.25$)."
2. *Valid finding*: "Global linear contrast reduction $I' = \mu + \alpha(I - \mu)$ is an affine transformation that is trivially and completely eliminated by standard min-max or z-score re-normalization at zero model cost."
3. *Valid finding*: "When histogram equalization (CLAHE) is evaluated as a candidate mitigation, forensic ablation reveals that its perceived efficacy is a confounding artifact of its internal min-max scaling step. On already-normalized images, CLAHE induces minor negative distortion ($\Delta = -0.0062$ on clean scans; $\Delta = -0.0169$ on noisy scans)."
4. *Valid finding*: "Morphological post-processing (connected component analysis with myocardial cavity preservation) provides consistent nominal artifact cleanup across discrete prediction masks without altering continuous input radiometry."

---

## 8. Recommended Next Experiments for Genuine Contrast Robustness

To evaluate contrast robustness where normalization cannot simply invert the degradation, future studies must employ **non-affine, spatially varying, or non-linear corruptions**:

### Experiment A: Non-Linear Power-Law (Gamma) Dynamic Compression
- **Transform**: $I_{\text{gamma}} = I_{\text{unit}}^\gamma, \quad \gamma \in \{0.5, 0.7, 1.5, 2.0\}$
- **Mechanism**: Modifies the probability distribution and non-linearly skews contrast. Linear scaling cannot cancel $(I^\gamma - \min)/( \max - \min) \neq (I - \min)/( \max - \min)$.

### Experiment B: Spatially Varying Radiofrequency Receive/Transmit Coil Bias Fields
- **Transform**: $I_{\text{bias}}(x, y) = I(x, y) \cdot B(x, y)$, where $B(x, y)$ is a low-frequency 2D polynomial or Gaussian spatial field (N4ITK model).
- **Mechanism**: Modifies contrast locally across the field of view. Global per-slice normalization cannot homogenize spatially varying signal loss.

### Experiment C: Contrast Windowing with Intensity Clamping (Saturation)
- **Transform**: $I_{\text{clamped}} = \text{clip}(I, p_{\text{low}}, p_{\text{high}})$.
- **Mechanism**: Causes irreversible information loss in the blood pool or myocardial wall.
