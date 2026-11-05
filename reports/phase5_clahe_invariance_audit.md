# Phase 5 Forensic Investigation: CLAHE Contrast Invariance and Causal Attribution Audit

**Investigation**: Scientific Forensic Audit of Min-Max Normalization Invariance and Causal Attribution of CLAHE Mitigation  
**Experiment ID**: `phase5_clahe_invariance_audit`  
**Date**: November 5, 2026  
**Auditor**: Senior Medical Imaging Research Scientist, Computer Vision Engineer & Experimental Auditor  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Model Architecture**: `LightweightUNet2D` (1,963,860 parameters)  
**Evaluated Checkpoint**: `checkpoints/50epochs/best_model_50epochs.pt` (Epoch 44, SHA-256: `a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38`)  
**Evaluation Cohort**: ACDC Held-Out Test Cohort (15 patients, 131 end-diastolic 2D slices)  

---

## Executive Summary & Core Verdict

This forensic audit investigates a critical methodological discrepancy discovered in the Phase 5 evaluation of the cardiac MRI segmentation model:

> **The Scientific Mystery**:
> In Phase 5, Contrast-Limited Adaptive Histogram Equalization (CLAHE) was reported to restore segmentation performance under severe contrast reduction from $0.8158$ to $0.8896$ (+0.0738 Dice, $p < 0.005$). However, the restoration utility `to_unit_interval` applies min-max scaling to $[0, 1]$ before histogram equalization. Because linear contrast scaling is an affine transformation $I' = \alpha I + (1 - \alpha)\mu$, min-max normalization analytically cancels the attenuation factor $\alpha$.
>
> **The Audit Verdict**:
> **CLAHE did NOT rescue the model from contrast degradation.** The recovery from $0.8158$ to $0.8896$ was **100% produced by standard linear dynamic range re-normalization** inside `to_unit_interval`. Across all 131 test slices and 15 patients, the input to CLAHE is identical down to float32 machine epsilon ($2.38 \times 10^{-7}$), and CLAHE outputs on clean vs corrupted images are numerically indistinguishable ($\Delta = 0.00 \times 10^0$). Pure z-score normalization alone recovers the full clean baseline of **$0.8958 \pm 0.0372$**, whereas adding CLAHE actually inflicts a **$-0.0062$ penalty**. The claim of "CLAHE contrast robustness" is a confounding artifact.

---

## 1. Exact Preprocessing Pipelines

Traced directly from `src/data/dataset.py`, `src/corruptions/contrast.py`, `src/preprocessing/restoration.py`, and `scripts/evaluate_mitigation_test.py`:

```
                    Raw ACDC NIfTI Volume
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
[Regime 1: Clean]    [Regime 2: Corrupted]    [Regime 3: CLAHE Mitigated]
I_clean              I_corrupt = mu + a(I-mu) I_corrupt = mu + a(I-mu)
(mu=0.0, sigma=1.0)  (mu=0.0, sigma=0.25)     (mu=0.0, sigma=0.25)
      │                       │                        │
      │                       │                        ▼ to_unit_interval(I)
      │                       │               I_01 = (I - min) / (max - min)
      │                       │               (alpha cancels completely!)
      │                       │                        │
      │                       │                        ▼ skimage.exposure.equalize_adapthist
      │                       │               Tile-based histogram equalization
      │                       │                        │
      │                       │                        ▼ z_score_standardize
      │                       │               Re-standardized to mu=0, sigma=1
      │                       │                        │
      ▼                       ▼                        ▼
Model Forward           Model Forward            Model Forward
Dice = 0.8958           Dice = 0.8158            Dice = 0.8896
```

### Key Architectural Vulnerability:
The `LightweightUNet2D` uses standard BatchNorm2d layers calibrated during training on zero-mean, unit-variance tensors ($\sigma \approx 1.0$). In Regime 2 (Unmitigated Contrast), the input tensor dynamic range is compressed $4\times$ ($\sigma \approx 0.25$), muting activation levels across network layers and causing a $-0.0800$ Dice collapse.

---

## 2. Mathematical Verification: Analytical Proof of Invariance

### Theorem (Invariance of Min-Max Scaling Under Linear Contrast Compression):
Let $I \in \mathbb{R}^{H \times W}$ represent a 2D image slice with spatial mean $\mu = \frac{1}{HW}\sum_{i,j} I_{i,j}$, spatial minimum $I_{\min} = \min_{i,j} I_{i,j}$, and spatial maximum $I_{\max} = \max_{i,j} I_{i,j}$.

The synthetic contrast corruption implemented in `src/corruptions/contrast.py` is defined as:
$$I'(\alpha) = \mu + \alpha (I - \mu) = \alpha I + (1 - \alpha)\mu, \quad \alpha \in (0, 1]$$

Because $\alpha > 0$ is a positive scalar and $(1 - \alpha)\mu$ is a spatial constant added across all pixels, rank order is strictly preserved:
$$I'_{\min} = \min_{i,j} [\alpha I_{i,j} + (1 - \alpha)\mu] = \alpha I_{\min} + (1 - \alpha)\mu$$
$$I'_{\max} = \max_{i,j} [\alpha I_{i,j} + (1 - \alpha)\mu] = \alpha I_{\max} + (1 - \alpha)\mu$$

The dynamic range of the corrupted slice is:
$$R(I') = I'_{\max} - I'_{\min} = [\alpha I_{\max} + (1 - \alpha)\mu] - [\alpha I_{\min} + (1 - \alpha)\mu] = \alpha (I_{\max} - I_{\min}) = \alpha R(I)$$

Now consider the min-max normalization operator $T(X) = \frac{X - X_{\min}}{R(X)}$ implemented in `to_unit_interval`:
$$T(I') = \frac{I' - I'_{\min}}{R(I')} = \frac{[\alpha I + (1 - \alpha)\mu] - [\alpha I_{\min} + (1 - \alpha)\mu]}{\alpha R(I)} = \frac{\alpha (I - I_{\min})}{\alpha R(I)} \equiv \frac{I - I_{\min}}{R(I)} = T(I)$$

$$\therefore T(I') \equiv T(I) \quad \forall \alpha > 0$$

### Critical Boundary Conditions:
1. **Affine Preservation**: The proof holds strictly because the corruption is an affine transform with slope $\alpha > 0$ and constant intercept $(1-\alpha)\mu$.
2. **Absence of Clamping**: No threshold saturation or integer quantization was applied during `contrast.py`. If pixel intensities had been clipped at dynamic boundaries, monotonicity would be destroyed and $T(I') \neq T(I)$.
3. **Per-Slice Computing**: Both `contrast.py` and `to_unit_interval` compute spatial statistics independently per 2D slice, matching boundary domains exactly.

---

## 3. Tensor-Level Forensic Evidence

To test this analytical proof empirically, a full forensic audit was executed across all 15 test patients and 131 slices (`experiments/phase5_clahe_invariance_audit/tensor_comparison_results.csv`):
- **Stage A**: Clean image from DataLoader ($\mu \approx 0, \sigma \approx 1$)
- **Stage B**: Severe contrast corrupted image ($\alpha = 0.25, \sigma \approx 0.25$)
- **Stage C**: Clean image after `to_unit_interval` ($T(I_{\text{clean}})$)
- **Stage D**: Severe contrast image after `to_unit_interval` ($T(I_{\text{corrupt}})$)
- **Stage E**: Clean image after CLAHE + re-standardization ($\text{CLAHE}(C)$)
- **Stage F**: Severe contrast image after CLAHE + re-standardization ($\text{CLAHE}(D)$)

### Measured Tensor Divergence:

| Stage Comparison | Evaluated Metric | Cohort Mean | Cohort Min | Cohort Max | Definitive Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A vs B** (Clean vs Corrupted) | Max Absolute Diff | $3.5821$ | $1.2140$ | $8.4520$ | Dynamic range compressed $4\times$ |
| | Mean Absolute Diff | $0.5642$ | $0.2105$ | $0.9412$ | Substantial pixel intensity shift |
| | Pearson Correlation ($r$) | $1.000000$ | $1.000000$ | $1.000000$ | Linear collinearity confirmed |
| **C vs D** (Clean Unit vs Corrupt Unit) | **Max Absolute Diff** | **$6.09 \times 10^{-9}$** | **$0.00 \times 10^{0}$** | **$2.38 \times 10^{-7}$** | **Strictly bounded by float32 epsilon** |
| | **Mean Absolute Diff** | **$3.12 \times 10^{-10}$** | **$0.00 \times 10^{0}$** | **$1.85 \times 10^{-8}$** | **Zero practical difference** |
| | **% Matching Pixels (tol $10^{-5}$)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **Every pixel matches within tolerance** |
| **E vs F** (Clean CLAHE vs Corrupt CLAHE) | **Max Absolute Diff** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **Literally identical tensors** |
| | **Mean Absolute Diff** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **$0.00 \times 10^{0}$** | **Exact numerical identity** |
| | **% Matching Pixels (exact)** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **Zero numerical deviation** |
| **Pred E vs Pred F** (Model Output) | **Pixel Agreement** | **$100.00\%$** | **$100.00\%$** | **$100.00\%$** | **All 8,585,216 prediction pixels match** |
| | Foreground Dice | **$1.000000$** | **$1.000000$** | **$1.000000$** | **Identical segmentation masks** |

---

## 4. Patient-Level Prediction Evidence

Model inference was executed across the 15 test patients under five conditions:
1. `Clean Baseline`: Clean input directly to model
2. `Severe Unmitigated`: $\alpha=0.25$ input directly to model
3. `Clean + CLAHE`: Clean input with CLAHE applied
4. `Severe + CLAHE`: $\alpha=0.25$ input with CLAHE applied
5. `Severe + Z-Score Alone`: $\alpha=0.25$ input with z-score re-standardization only (NO CLAHE)

### Patient 3D Macro Foreground Dice Matrix:

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
| **Mean** | **0.8958** | **0.8158** | **0.8896** | **0.8896** | **0.0000** | **0.8958** | **0.0000** |
| **Std** | **0.0372** | **0.0677** | **0.0429** | **0.0429** | **0.0000** | **0.0372** | **0.0000** |

---

## 5. Statistical Interpretation & Causal Decomposition

### Mathematical Recovery Decomposition:
The observed performance change between Unmitigated Corrupted and Mitigated Corrupted is:
$$\Delta_{\text{total}} = \text{Dice}(\text{Corrupt + CLAHE}) - \text{Dice}(\text{Corrupt Unmitigated}) = 0.8896 - 0.8158 = \mathbf{+0.0738}$$

We isolate the causal components:
1. **Component 1 (Linear Normalization Alone)**:
   $$\Delta_{\text{norm}} = \text{Dice}(\text{Corrupt + Z-Score}) - \text{Dice}(\text{Corrupt Unmitigated}) = 0.8958 - 0.8158 = \mathbf{+0.0800}$$
2. **Component 2 (CLAHE on Normalized Input)**:
   $$\Delta_{\text{CLAHE}} = \text{Dice}(\text{Corrupt + CLAHE}) - \text{Dice}(\text{Corrupt + Z-Score}) = 0.8896 - 0.8958 = \mathbf{-0.0062}$$

$$\Delta_{\text{total}} = \Delta_{\text{norm}} + \Delta_{\text{CLAHE}} = (+0.0800) + (-0.0062) = +0.0738$$

### Causal Attribution Summary:
- **Normalization Share**: $\frac{+0.0800}{+0.0738} = \mathbf{108.4\%}$ of the observed recovery.
- **CLAHE Share**: $\frac{-0.0062}{+0.0738} = \mathbf{-8.4\%}$ (a net degradation).
- **The Confounding Error**: The Phase 5 experimental arm simultaneously altered two variables: it re-scaled input variance back to $1.0$ and applied adaptive histogram redistribution. The statistical hypothesis test captured the benefit of variance restoration and falsely credited CLAHE.

---

## 6. Scientific Claims That Remain Valid

1. **Morphological Post-Processing (LCC with Cavity Safeguard)**:
   - Preserved anatomical integrity across all tested conditions.
   - Removed detached spurious islands ($<30$ px) without filling legitimate ventricular cavities (`max_hole_size = 16`).
   - Achieved consistent positive deltas across all 10 conditions ($\Delta \in [+0.0014, +0.0140]$), with largest benefit on severe blur ($0.7775 \to 0.7915$, $+1.40\%$).
2. **Median Filter Inefficacy on 50-Epoch Model**:
   - Accurately observed that $3 \times 3$ median filtering does not enhance noise robustness ($\Delta = -0.0003$ to $-0.0029$).
   - The 50-epoch baseline model already possesses deep spatial pooling invariance to high-frequency noise ($0.8958 \to 0.8913$ under severe noise); spatial filtering degrades fine myocardial borders.
3. **Negative Control Diagnostics**:
   - Correctly captured that applying CLAHE to clean images degrades performance ($0.8958 \to 0.8896$, $-0.0062$) by distorting natural intensity gradients.
   - Correctly captured that applying CLAHE to noisy images severely degrades performance ($0.8913 \to 0.8744$, $-0.0169$) by amplifying high-frequency noise spikes.

---

## 7. Scientific Claims That Require Correction

| Original Claim | Revised Scientific Reality | Required Action |
| :--- | :--- | :--- |
| "CLAHE restores performance under severe contrast reduction (+0.0738 Dice, $p < 0.005$)." | **Misattributed.** The recovery was $100\%$ generated by min-max scaling inside `to_unit_interval`. CLAHE itself imposed a $-0.0062$ penalty. | **Retract claim.** State that linear normalization fully restores affine contrast degradation. |
| "Combined CLAHE + Post-Processing is the optimal contrast mitigation (+0.0763 Dice)." | **Misattributed.** Optimal contrast mitigation is pure Z-Score normalization + post-processing ($0.8980 \pm 0.0357$). | **Update recommended pipeline.** Remove CLAHE; use standard normalization + LCC. |
| "CLAHE achieves 99.3% recovery back to clean baseline." | **Mathematical artifact.** Clean+CLAHE and Corrupt+CLAHE are identical arrays ($0.8896 / 0.8958 = 99.31\%$). | **Retract metric.** The metric measures CLAHE distortion on clean data, not recovery. |
| "Statistically significant paired tests confirm CLAHE efficacy." | **Confounded causal inference.** Tests evaluated unstandardized vs standardized data, not CLAHE vs no-CLAHE. | **Re-frame statistical analysis.** Re-state as sensitivity to unnormalized input dynamic range. |

---

## 8. Recommended Next Experiments for Genuine Contrast Robustness

To rigorously evaluate contrast robustness where simple linear re-normalization cannot trivially cancel the degradation, future investigations should implement:

### Experiment 1: Non-Linear Power-Law (Gamma) Contrast Compression
- **Formulation**: $I_{\text{gamma}} = \text{sign}(I_{\text{unit}}) \cdot |I_{\text{unit}}|^\gamma, \quad \gamma \in \{0.5, 0.7, 1.4, 2.0\}$
- **Scientific Rationale**: Power-law curves warp the intensity histogram non-linearly. Because $(I^\gamma - \min) / (\max - \min) \neq (I - \min) / (\max - \min)$, min-max scaling cannot cancel $\gamma$, allowing genuine assessment of adaptive histogram methods.

### Experiment 2: Spatially Varying RF Receive/Transmit Coil Bias Fields
- **Formulation**: $I_{\text{bias}}(x, y) = I(x, y) \cdot B(x, y)$, where $B(x, y)$ is a low-frequency smooth polynomial or 2D Gaussian spatial profile (simulating physical B1 coil inhomogeneity).
- **Scientific Rationale**: Global slice normalization cannot correct spatially heterogeneous signal attenuation. Evaluating N4ITK bias-field correction vs unsharp masking under localized contrast dropout tests true spatial image restoration.

### Experiment 3: Dynamic Windowing with Non-Linear Intensity Clamping (Saturation)
- **Formulation**: $I_{\text{clipped}} = \text{clamp}(I, p_{10}, p_{90})$ where intensities below the 10th percentile and above the 90th percentile are saturated.
- **Scientific Rationale**: Introduces irreversible information loss in the blood pool and myocardium, forcing the segmentation model to rely on structural context rather than gray-level re-scaling.
