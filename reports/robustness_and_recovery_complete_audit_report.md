# Comprehensive Audit Report: Baseline, Degradation & Recovery Lifecycle
## Epoch 44 Lightweight 2D U-Net on ACDC Cardiac MRI Benchmark

---

### Executive Summary

This report provides the complete, mathematically audited, and empirical transition ledger for the **Epoch 44 Lightweight 2D U-Net** (`best_model_50epochs.pt`) evaluated on the **15 held-out test patients** (131 End-Diastolic short-axis slices) of the ACDC benchmark.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    THE FULL PERFORMANCE LIFECYCLE                               │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│   STAGE 1: CLEAN BASELINE           STAGE 2: CORRUPTION INJECTION       STAGE 3: MITIGATION RECOVERY     │
│   [Pristine 131 Test Slices]        [Unmitigated Degradation]           [Preprocessing & Post-Proc]      │
│                                                                                                  │
│   Patient 3D Dice: 0.8958  ───────► Contrast Sev (α=0.25): 0.8158  ───► Normalization Alone: 0.8958    │
│   Slice 2D Dice:   0.8297           (-8.85% Drop)                       (+100% Full Recovery!)           │
│                                                                                                  │
│   • LV:  0.9420                     Spatial Blur Sev (σ=2.5): 0.7775 ─► Morph LCC Cleanup: 0.7915       │
│   • RV:  0.9019                     (-13.15% Drop)                      (+11.8% Partial Recovery)        │
│   • MYO: 0.8434                                                                                  │
│                                     Gaussian Noise Sev (σ=0.1): 0.8913► Denoising Filter: 0.8910        │
│                                     (-0.51% Drop - Intrinsic Robust)    (No Gain / Slight Penalty)       │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Stage 1: Clean Baseline Benchmark

Evaluated on 15 held-out test patients (131 short-axis slices, strictly End-Diastolic `frame01.nii.gz`) without any artificial noise, blur, or contrast alteration.

### 1.1 Master Clean Baseline Metrics

| Metric Operational Level | Macro Foreground Dice | Macro Foreground IoU | RV Dice | MYO Dice | LV Dice |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2D Slice-Level Macro Average (131 slices)** | **$0.8297$** ($82.97\%$) | **$0.7617$** ($76.17\%$) | $0.8136$ | $0.7914$ | $0.8840$ |
| **3D Patient Voxel-Pooled ($\mu \pm \sigma$, 15 patients)** | **$0.8958 \pm 0.0372$** | **$0.8161 \pm 0.0577$** | **$0.9019 \pm 0.0455$** | **$0.8434 \pm 0.0541$** | **$0.9420 \pm 0.0301$** |

### 1.2 Anatomical Class Hierarchy in Clean Scans
- **Left Ventricle (LV)**: **`0.9420` (94.20%)** — Best recognized structure due to high contrast blood pool and consistent roughly circular cross-section.
- **Right Ventricle (RV)**: **`0.9019` (90.19%)** — High volume-pooled accuracy across central slices.
- **Myocardium Wall (MYO)**: **`0.8434` (84.34%)** — Lower baseline because it is a thin annular ring ($8–12\text{ mm}$) bounded by two separate complex borders (epicardium and endocardium).

---

## 2. Stage 2: Degradation & Corruption Drop

In Phase 4, corruptions were directly applied to normalized image tensors **without re-normalization or dynamic range clipping**, directly measuring the raw sensitivity of the frozen Epoch 44 neural network.

### 2.1 Master Degradation Ledger (Clean $\rightarrow$ Corrupted)

| Condition Name | Perturbation Type | Severity / Param Value | Clean Score (Patient 3D) | Corrupted Score (Patient 3D) | Absolute Drop ($\Delta$) | Relative Drop (%) | Paired $t$ $p_{\text{Holm}}$ | FWER Decision ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `clean_baseline` | None | $0.0$ | **$0.8958$** | **$0.8958$** | $0.0000$ | $0.00\%$ | — | Benchmark Baseline |
| `gaussian_noise_mild` | Additive Noise | $\sigma=0.02$ | $0.8958$ | $0.8957$ | $-0.0001$ | $-0.01\%$ | $1.0000$ | Not Significant |
| `gaussian_noise_mod` | Additive Noise | $\sigma=0.05$ | $0.8958$ | $0.8948$ | $-0.0010$ | $-0.11\%$ | $0.3651$ | Not Significant |
| `gaussian_noise_sev` | Additive Noise | $\sigma=0.10$ | $0.8958$ | $0.8913$ | $-0.0044$ | $-0.51\%$ | $0.0741$ | Nominal Only ($p_{\text{raw}}=0.014$) |
| `gaussian_blur_mild` | Spatial Smoothing | $\sigma=0.75$ | $0.8958$ | $0.8860$ | $-0.0098$ | $-1.07\%$ | $0.0741$ | Nominal Only ($p_{\text{raw}}=0.012$) |
| `gaussian_blur_mod` | Spatial Smoothing | $\sigma=1.50$ | $0.8958$ | $0.8540$ | **$-0.0418$** | **$-4.62\%$** | **$0.0029$** | **Statistically Significant** |
| `gaussian_blur_sev` | Spatial Smoothing | $\sigma=2.50$ | $0.8958$ | **$0.7775$** | **$-0.1182$** | **$-13.15\%$** | **$0.0004$** | **Catastrophic Crash ($p < 0.001$)** |
| `contrast_reduction_mild` | Dynamic Squeeze | $\alpha=0.75$ | $0.8958$ | $0.8956$ | $-0.0002$ | $-0.01\%$ | $1.0000$ | Not Significant |
| `contrast_reduction_mod` | Dynamic Squeeze | $\alpha=0.50$ | $0.8958$ | $0.8935$ | $-0.0023$ | $-0.24\%$ | $0.8571$ | Not Significant |
| `contrast_reduction_sev` | Dynamic Squeeze | $\alpha=0.25$ | $0.8958$ | **$0.8158$** | **$-0.0800$** | **$-8.85\%$** | **$0.0035$** | **Statistically Significant ($p < 0.005$)** |

---

### 2.2 Anatomical Structure Breakdown: Who Suffered the Most?

```
Condition               RV Dice (Δ)          MYO Dice (Δ)         LV Dice (Δ)
----------------------------------------------------------------------------------
Clean Baseline          0.8136 (Reference)   0.7914 (Reference)   0.8840 (Reference)
Gaussian Noise Severe   0.8030 (-0.0106)     0.7799 (-0.0115)     0.8811 (-0.0029)
Gaussian Blur Moderate  0.7791 (-0.0345)     0.7423 (-0.0491)     0.8660 (-0.0180)
Gaussian Blur Severe    0.6709 (-0.1427)     0.6571 (-0.1343)     0.8120 (-0.0720)
Contrast Severe         0.6462 (-0.1674)     0.7051 (-0.0863)     0.8304 (-0.0536)
----------------------------------------------------------------------------------
```

1. **Right Ventricle (RV)**: Suffered the highest vulnerability across all corruptions (collapsed by **$-16.74\%$** in contrast and **$-14.27\%$** in blur) because the crescent wall is paper-thin and easily swallowed by blurred tissue.
2. **Myocardium (MYO)**: Heavily damaged by blur (**$-13.43\%$**) because both inner and outer boundaries get smeared.
3. **Left Ventricle (LV)**: Retained strong recognition even under worst-case blur ($0.8120$) due to large anatomical cavity area.

---

## 3. Stage 3: Mitigation & Recovery Ledger ("Konsa Kya Karne Se Kitna Wapas Aaya?")

In Phase 5, pre-registered, validation-locked mitigation methods were applied to rescue degraded performance without retraining model weights.

### 3.1 Master Recovery Ledger (Corrupted $\rightarrow$ Mitigated)

| Target Condition | Unmitigated Corrupted Score | Mitigation Pipeline Applied | Mitigated Score Reached | Absolute Recovery ($\Delta_{\text{rec}}$) | % of Drop Recovered | % of Clean Baseline Restored | Statistical Decision |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Severe Contrast** ($\alpha=0.25$) | $0.8158$ | **CLAHE Preprocessing Alone** | **$0.8896$** | **$+0.0738$** | **$92.25\%$** | **$99.31\%$** | **Significant** ($p_{\text{Holm}}=0.0031$) |
| **Severe Contrast** ($\alpha=0.25$) | $0.8158$ | **Combined (CLAHE + Morph LCC)** | **$0.8921$** | **$+0.0763$** | **$95.38\%$** | **$99.59\%$** | **Significant** ($p_{\text{Holm}}=0.0025$) |
| **Severe Contrast** ($\alpha=0.25$) | $0.8158$ | **Pure Linear Z-Score Normalization** | **$0.8958$** | **$+0.0800$** | **$100.00\%$** | **$100.00\%$** | **Full Ground-Truth Restoration** |
| **Severe Blur** ($\sigma=2.50$) | $0.7775$ | **Morphological LCC Post-Processing**| **$0.7915$** | **$+0.0140$** | **$11.84\%$** | **$88.36\%$** | Nominal Only ($p_{\text{raw}}=0.023$) |
| **Severe Blur** ($\sigma=2.50$) | $0.7775$ | **CLAHE Preprocessing Alone** | **$0.8051$** | **$+0.0276$** | **$23.35\%$** | **$89.88\%$** | Nominal Edge Boost ($p_{\text{raw}}=0.039$) |
| **Severe Noise** ($\sigma=0.10$)| $0.8913$ | **Median Filter ($3\times 3$) + LCC**| **$0.8910$** | **$-0.0003$** | **$0.00\%$** | **$99.46\%$** | No Recovery (Slight Penalty) |
| **Clean Baseline** ($0.0$) | $0.8958$ | **Morphological LCC Post-Processing**| **$0.8980$** | **$+0.0022$** | — | **$100.25\%$** | Clean Baseline Enhanced |

---

## 4. Root Cause Mechanics: Why It Dropped & How It Recovered ("Kyun Gira Aur Kaise Aaya?")

### 4.1 Why Did Spatial Blur Crash the Network (-13.15%)?
- **Spatial Resolution Mismatch**: Adult myocardium ring thickness is only $8–12\text{ mm}$ ($6–12\text{ pixels}$). A 2D Gaussian blur with $\sigma=2.50$ extends over a kernel window of $6\sigma+1 = 16\text{ pixels}$. 
- The filter mathematically averages high-intensity blood cavity pixels across the myocardium into the dark chest cavity. Gradient step edges $(\nabla I)$ vanish, making it mathematically impossible for shallow convolutional layers to localize the tissue interface.
- **Why Recovery Was Limited to +1.4%**: Morphological LCC removes false-positive islands generated out in the chest wall, but cannot reconstruct structural high-frequency phase information that was physically destroyed.

### 4.2 Why Did Severe Contrast Squeeze Drop Performance (-8.85%)?
- At $\alpha=0.25$, pixel variance compressed by $75\%$ ($\sigma_{\text{tensor}} \approx 0.25$).
- The network's `BatchNorm2d` layers had fixed running statistics calibrated on $\mathcal{N}(0, 1)$. 
- Squeezed activations failed to cross the threshold of ReLU activation layers ($\max(0, x)$), starving deeper bottleneck layers of feature representations.

---

### 4.3 The "Magic" Explained: How Contrast Recovered 100%

In early testing, it was assumed that **CLAHE** produced the $+0.0738$ recovery. Our rigorous mathematical and code audit proved this was an artifact of the normalization step inside the pipeline:

1. **Affine Contrast Corruption**:
   $$I_{\text{corrupt}} = \alpha I + (1 - \alpha)\mu, \quad \alpha \in (0, 1]$$
2. **Min-Max Scaling in Preprocessing (`to_unit_interval`)**:
   $$T(X) = \frac{X - \min(X)}{\max(X) - \min(X)}$$
3. **Substituting $I_{\text{corrupt}}$ into $T(X)$**:
   $$T(I_{\text{corrupt}}) = \frac{[\alpha I + (1-\alpha)\mu] - [\alpha \min(I) + (1-\alpha)\mu]}{[\alpha \max(I) + (1-\alpha)\mu] - [\alpha \min(I) + (1-\alpha)\mu]} = \frac{\alpha [I - \min(I)]}{\alpha [\max(I) - \min(I)]} \equiv \frac{I - \min(I)}{\max(I) - \min(I)} = T(I)$$
4. **Mathematical Cancellation**:
   Because $\alpha > 0$ cancels identically in numerator and denominator, **$T(I_{\text{corrupt}}) \equiv T(I_{\text{clean}})$**!

```
Causal Decomposition of the +0.0738 Recovery:
─────────────────────────────────────────────────────────────────────────────
Linear Z-Score / Min-Max Normalization Alone:    +0.0800  (108.4% of recovery)
CLAHE (Histogram Warping) Addition:              -0.0062  (-8.4% Net Penalty)
─────────────────────────────────────────────────────────────────────────────
Net Measured Recovery:                           +0.0738  (100.0%)
```

**Key Takeaway**: Linear Dynamic Range Normalization completely solves positive linear contrast attenuation. CLAHE is not only unnecessary, it introduces a $-0.0062$ penalty on clean scans and catastrophically amplifies noise on noisy scans ($-0.0169$).

---

### 4.4 The Post-Processing Cavity Safeguard
Standard morphological hole-filling (`scipy.ndimage.binary_fill_holes`) applied to an annular myocardium ring fills the entire enclosed central Left Ventricle chamber (~706 pixels), creating an anatomically invalid solid disk. 

The validation-locked post-processing pipeline implemented **`max_hole_size = 16 pixels`**:
- Small intramyocardial voids (mean 9 pixels) are closed.
- The central $706\text{ px}$ blood lumen remains $100\%$ untouched and anatomically accurate.

---

## 5. Negative Control Audit: Risks of Applying Filters Blindly

| Filter Applied | Tested Condition | Untreated Score | Treated Score | Net Impact ($\Delta$) | Risk Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CLAHE** | Clean Baseline | $0.8958$ | $0.8896$ | **$-0.0062$** | Alters calibrated tissue contrast ratios. |
| **CLAHE** | Severe Noise ($\sigma=0.10$) | $0.8913$ | $0.8744$ | **$-0.0169$** | **Severe Noise Amplification**: High-frequency spikes get boosted. |
| **Median Filter** | Clean Baseline | $0.8958$ | $0.8929$ | **$-0.0029$** | Blurs sharp, authentic epicardial/endocardial boundaries. |

---

## 6. Final Recommendations for Production Clinical Pipelines

```
[Raw Patient DICOM/NIfTI Input]
              │
              ▼
[1. Strict Per-Slice Z-Score Normalization: μ=0, σ=1]  ◄── (Sufficient to 100% cure contrast issues)
              │
              ▼
[2. DO NOT Apply CLAHE or Median Filter Blindly]       ◄── (Prevents noise amplification & boundary blur)
              │
              ▼
[3. Epoch 44 LightweightUNet2D Inference]
              │
              ▼
[4. Morphological Post-Processing with Cavity Safeguard]
    • RV Min Area: 30 px  |  MYO Min Area: 50 px  |  LV Min Area: 30 px
    • Cavity Safeguard: max_hole_size = 16 px         ◄── (Guarantees central LV lumen preservation)
              │
              ▼
[Final Validated 3D Volume Output: 0.8980 Patient Dice]
```
