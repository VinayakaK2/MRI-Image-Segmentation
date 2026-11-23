# CLAHE Forensic Audit: Mathematical Invariance & Causal Attribution Analysis

**Project**: ACDC Cardiac MRI Segmentation — LightweightUNet2D  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Audit Date**: November 6, 2026  
**Auditor**: Senior Medical Imaging Research Scientist, Computer Vision Engineer & Experimental Auditor  
**Investigation**: Forensic Investigation of Min-Max Normalization Invariance and Causal Attribution of CLAHE Mitigation  

---

## 1. Executive Summary & Forensic Verdict

In the initial Phase 5 evaluation of the cardiac MRI segmentation model, Contrast-Limited Adaptive Histogram Equalization (CLAHE) was reported to restore segmentation performance under severe contrast reduction from **$0.8158$** to **$0.8896$** ($\Delta = +0.0738$ Dice, paired $t = 4.1235$, $p < 0.005$).

This forensic investigation proves, both analytically and empirically, that:

> **The Forensic Verdict**:
> 1. **CLAHE did NOT rescue the model from contrast degradation.** The performance restoration was **100% produced by standard linear dynamic range re-normalization** occurring inside the helper function `to_unit_interval`.
> 2. Because the synthetic contrast corruption is a positive affine transform ($I' = \alpha I + (1-\alpha)\mu$), min-max normalization analytically cancels $\alpha$.
> 3. Across all 131 test slices and 15 patients, the input tensors to CLAHE on clean vs corrupted scans are **identical down to float32 machine epsilon ($2.38 \times 10^{-7}$)**, and the outputs of CLAHE are **strictly identical ($0.00 \times 10^0$)**.
> 4. Pure linear Z-score standardization alone (without CLAHE) restores the full clean baseline performance of **$0.8958 \pm 0.0372$** ($\Delta = +0.0800$).
> 5. Applying CLAHE on top of normalized inputs actually inflicts a **$-0.0062$ performance penalty** ($0.8896$ vs $0.8958$).
> 6. The claim that "CLAHE restores contrast robustness" is a confounding artifact of input re-scaling.

---

## 2. Mathematical Proof of Invariance

### Theorem: Invariance of Min-Max Scaling Under Positive Linear Contrast Attenuation
Let $I \in \mathbb{R}^{H \times W}$ denote a 2D image slice with spatial mean $\mu = \frac{1}{HW}\sum_{i,j} I_{i,j}$, minimum $I_{\min} = \min_{i,j} I_{i,j}$, and maximum $I_{\max} = \max_{i,j} I_{i,j}$.

The synthetic contrast reduction corruption implemented in `src/corruptions/contrast.py` (`apply_contrast_reduction`) is defined as:
$$I'(\alpha) = \mu + \alpha (I - \mu) = \alpha I + (1 - \alpha)\mu, \quad \alpha \in (0, 1]$$

Because $\alpha > 0$ is a positive scalar and $(1 - \alpha)\mu$ is a spatial constant added identically to every pixel, the spatial ordering of pixel intensities is strictly preserved. Therefore:
$$I'_{\min} = \min_{i,j} [\alpha I_{i,j} + (1 - \alpha)\mu] = \alpha I_{\min} + (1 - \alpha)\mu$$
$$I'_{\max} = \max_{i,j} [\alpha I_{i,j} + (1 - \alpha)\mu] = \alpha I_{\max} + (1 - \alpha)\mu$$

The dynamic range (span) of the corrupted slice is:
$$\text{Range}(I') = I'_{\max} - I'_{\min} = [\alpha I_{\max} + (1 - \alpha)\mu] - [\alpha I_{\min} + (1 - \alpha)\mu] = \alpha (I_{\max} - I_{\min}) = \alpha \cdot \text{Range}(I)$$

Now consider the min-max normalization operator $T(X) = \frac{X - X_{\min}}{\text{Range}(X)}$ implemented in `src/preprocessing/restoration.py` (`to_unit_interval`):
$$T(I') = \frac{I' - I'_{\min}}{\text{Range}(I')} = \frac{[\alpha I + (1 - \alpha)\mu] - [\alpha I_{\min} + (1 - \alpha)\mu]}{\alpha \cdot \text{Range}(I)} = \frac{\alpha (I - I_{\min})}{\alpha \cdot \text{Range}(I)} \equiv \frac{I - I_{\min}}{\text{Range}(I)} = T(I)$$

$$\therefore T(I') \equiv T(I) \quad \forall \alpha > 0 \quad \blacksquare$$

### Critical Code Assumptions Verified:
1. **Affine Preservation**: In `src/corruptions/contrast.py`, no threshold clamping, sigmoid non-linearity, or dynamic range clipping is applied. The operation is purely affine float32 arithmetic.
2. **Per-Slice Statistics**: Both `contrast.py` and `to_unit_interval` compute spatial statistics independently per 2D slice, matching coordinate frames exactly.
3. **Absence of Integer Quantization**: Tensors remain in IEEE 754 32-bit floating-point format throughout corruption and normalization.

---

## 3. End-to-End Pipeline Execution Trace

The actual execution flow through the codebase is traced below:

```
                      ACDC NIfTI 3D Volume
                               │
                1. Center crop/pad to 256x256
                2. Percentile clipping [0.5, 99.5]
                3. Per-slice z-score standardization
                               ▼
                   DataLoader Output (I_clean)
                    (mu ~ 0.0, sigma ~ 1.0)
                               │
       ┌───────────────────────┼────────────────────────┐
       ▼                       ▼                        ▼
[Arm 1: Clean]       [Arm 2: Corrupted]       [Arm 3: CLAHE Mitigated]
I_clean              I_corrupt = mu + a(I-mu) I_corrupt = mu + a(I-mu)
(mu=0.0, sigma=1.0)  (mu=0.0, sigma=0.25)     (mu=0.0, sigma=0.25)
       │                       │                        │
       │                       │                        ▼ to_unit_interval(I)
       │                       │               I_unit = (I - min) / (max - min)
       │                       │               (alpha cancels completely!)
       │                       │                        │
       │                       │                        ▼ skimage equalize_adapthist
       │                       │               Tile-based histogram equalization
       │                       │                        │
       │                       │                        ▼ z_score_standardize
       │                       │               Re-standardized to mu=0, sigma=1
       │                       │                        │
       ▼                       ▼                        ▼
LightweightUNet2D       LightweightUNet2D        LightweightUNet2D
Dice = 0.8958           Dice = 0.8158            Dice = 0.8896
```

### Why Unmitigated Corrupted Fails:
`LightweightUNet2D` contains standard `BatchNorm2d` layers calibrated during training on inputs with variance $\sigma^2 \approx 1.0$. In Arm 2 (Unmitigated Corrupted), the input tensor dynamic range is compressed $4\times$ ($\sigma \approx 0.25$), which scales down activation levels across initial convolutional layers and degrades feature representation, dropping Dice from $0.8958$ to $0.8158$.

---

## 4. Empirical Tensor Measurements Across Test Cohort

Source: `experiments/phase5_clahe_invariance_audit/tensor_comparison_results.csv` (131 test slices).

- **Stage A**: Clean input from DataLoader
- **Stage B**: Severe contrast corrupted ($\alpha = 0.25$)
- **Stage C**: Clean input after `to_unit_interval` ($T(I_{\text{clean}})$)
- **Stage D**: Severe contrast corrupted after `to_unit_interval` ($T(I_{\text{corrupt}})$)
- **Stage E**: Clean input after CLAHE + re-standardization ($\text{CLAHE}(C)$)
- **Stage F**: Severe contrast corrupted after CLAHE + re-standardization ($\text{CLAHE}(D)$)

| Comparison Pair | Evaluated Metric | Mean Value | Minimum | Maximum | Verification Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A vs B** (Clean vs Corrupt) | Max Absolute Difference | 3.5821 | 1.2140 | 8.4520 | Large dynamic range compression ($4\times$) |
| | Pearson Correlation ($r$) | 1.000000 | 1.000000 | 1.000000 | Perfect linear collinearity |
| **C vs D** (Clean Unit vs Corrupt Unit) | **Max Absolute Difference** | **$6.09 \times 10^{-9}$** | **$0.00 \times 10^{0}$** | **$2.38 \times 10^{-7}$** | **Strictly bounded by float32 machine epsilon** |
| | **Mean Absolute Difference** | **$3.12 \times 10^{-10}$** | **$0.00 \times 10^{0}$** | **$1.85 \times 10^{-8}$** | **Zero practical divergence** |
| | **Matching Pixels ($|\Delta| < 10^{-5}$)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **Exact pixel identity** |
| **E vs F** (Clean CLAHE vs Corrupt CLAHE) | **Max Absolute Difference** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **Exact numerical array identity** |
| | **Mean Absolute Difference** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **Identical float32 representations** |
| **Pred E vs Pred F** (Model Output) | **Pixel Agreement** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **All 8,585,216 test voxels match** |
| | **Foreground Dice** | **$1.000000$** | **$1.000000$** | **$1.000000$** | **Identical segmentation masks** |

---

## 5. Model Inference Under Isolated Experimental Conditions

To separate the effect of input normalization from CLAHE, model inference was evaluated across all 15 test patients under five conditions:
1. `Clean Baseline`: Clean input directly to model
2. `Severe Unmitigated`: $\alpha = 0.25$ input directly to model
3. `Clean + CLAHE`: Clean input with CLAHE applied
4. `Severe + CLAHE`: $\alpha = 0.25$ input with CLAHE applied
5. `Severe + Z-Score Alone`: $\alpha = 0.25$ input with per-slice z-score re-standardization only (NO CLAHE)

### Patient 3D Macro Foreground Dice Results:

| Patient ID | Clean Baseline | Severe Unmitigated | Clean + CLAHE | Severe + CLAHE | Severe + Z-Score Alone |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `patient001` | 0.8932 | 0.8727 | 0.8998 | 0.8998 | 0.8932 |
| `patient004` | 0.9178 | 0.8745 | 0.9157 | 0.9157 | 0.9178 |
| `patient009` | 0.8827 | 0.8926 | 0.8847 | 0.8847 | 0.8827 |
| `patient027` | 0.9088 | 0.8836 | 0.8916 | 0.8916 | 0.9088 |
| `patient038` | 0.8026 | 0.7058 | 0.7766 | 0.7766 | 0.8026 |
| `patient040` | 0.9291 | 0.6647 | 0.9268 | 0.9268 | 0.9291 |
| `patient044` | 0.9131 | 0.8230 | 0.9062 | 0.9062 | 0.9131 |
| `patient052` | 0.8857 | 0.8763 | 0.8826 | 0.8826 | 0.8857 |
| `patient059` | 0.9361 | 0.7798 | 0.9260 | 0.9260 | 0.9361 |
| `patient062` | 0.9383 | 0.8797 | 0.9318 | 0.9318 | 0.9383 |
| `patient063` | 0.8830 | 0.7785 | 0.8851 | 0.8851 | 0.8830 |
| `patient068` | 0.9207 | 0.8086 | 0.9257 | 0.9257 | 0.9207 |
| `patient086` | 0.8345 | 0.8047 | 0.8183 | 0.8183 | 0.8345 |
| `patient098` | 0.8810 | 0.7777 | 0.8647 | 0.8647 | 0.8810 |
| `patient100` | 0.9096 | 0.8148 | 0.9079 | 0.9079 | 0.9096 |
| **Mean** | **0.8958** | **0.8158** | **0.8896** | **0.8896** | **0.8958** |
| **Std** | **0.0372** | **0.0677** | **0.0429** | **0.0429** | **0.0372** |

---

## 6. Causal Decomposition & Confounding Analysis

We decompose the observed $+0.0738$ recovery into its independent causal components:

$$\Delta_{\text{total}} = \text{Dice}(\text{Severe + CLAHE}) - \text{Dice}(\text{Severe Unmitigated}) = 0.8896 - 0.8158 = \mathbf{+0.0738}$$

1. **Causal Component 1: Linear Dynamic Range Normalization Alone**:
   $$\Delta_{\text{norm}} = \text{Dice}(\text{Severe + Z-Score}) - \text{Dice}(\text{Severe Unmitigated}) = 0.8958 - 0.8158 = \mathbf{+0.0800}$$
2. **Causal Component 2: Adaptive Histogram Redistribution (CLAHE)**:
   $$\Delta_{\text{CLAHE}} = \text{Dice}(\text{Severe + CLAHE}) - \text{Dice}(\text{Severe + Z-Score}) = 0.8896 - 0.8958 = \mathbf{-0.0062}$$

$$\Delta_{\text{total}} = \Delta_{\text{norm}} + \Delta_{\text{CLAHE}} = (+0.0800) + (-0.0062) = +0.0738$$

### Attribution Breakdown:
- **Linear Normalization Share**: $\frac{+0.0800}{+0.0738} = \mathbf{108.4\%}$
- **CLAHE Share**: $\frac{-0.0062}{+0.0738} = \mathbf{-8.4\%}$ (a net penalty)
- **Mechanism of Confounding**: The Phase 5 experimental arm simultaneously introduced two modifications: it re-scaled input variance back to $1.0$ (via `to_unit_interval` and `z_score_standardize`) and applied local histogram redistribution. The statistical hypothesis test captured the large benefit of variance normalization and falsely credited CLAHE.

---

## 7. Retraction & Correction Protocol

| Claim in Previous Reports | Forensic Reality | Required Scientific Correction |
| :--- | :--- | :--- |
| "CLAHE restores contrast robustness (+0.0738 Dice, $p < 0.005$)." | Recovery produced entirely by linear min-max scaling inside `to_unit_interval`. CLAHE itself added a $-0.0062$ penalty. | **Retracted.** Attribute recovery to linear dynamic range re-standardization. |
| "CLAHE achieves 99.3% recovery back to clean baseline." | Clean+CLAHE and Corrupt+CLAHE arrays are identical ($0.8896$). The ratio $0.8896 / 0.8958 = 99.31\%$ measures CLAHE distortion on clean data. | **Retracted.** Disclose that metric is a mathematical artifact of input array identity. |
| "Combined CLAHE + Post-Processing is the optimal contrast mitigation." | Optimal contrast mitigation is pure Z-Score re-standardization + morphological post-processing ($0.8980 \pm 0.0357$). | **Corrected.** Update recommended clinical pipeline to standard normalization + LCC. |

---

## 8. Recommendations for Valid Non-Linear Contrast Robustness Testing

To evaluate true contrast robustness where linear normalization cannot trivially cancel the degradation, future studies should implement:
1. **Power-Law (Gamma) Contrast Compression**: $I_{\text{gamma}} = \text{sign}(I) \cdot |I|^\gamma$ ($\gamma \in [0.5, 2.0]$). Non-linear warping does not commute with min-max scaling ($T(I^\gamma) \neq T(I)$).
2. **Smooth RF Receive/Transmit Coil Bias Fields ($B_1$ Shading)**: $I_{\text{bias}}(x,y) = I(x,y) \cdot B(x,y)$ with low-frequency spatial variation. Cannot be eliminated by global slice normalization.
3. **Dynamic Range Saturation & Clamping**: $I_{\text{clipped}} = \text{clamp}(I, p_{10}, p_{90})$. Irreversibly eliminates contrast in extreme intensity percentiles, testing true spatial context robustness.
