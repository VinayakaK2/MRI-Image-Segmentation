# Comprehensive Robustness & Mitigation Evaluation Report: Epoch 44 Lightweight 2D U-Net Checkpoint
## ACDC Cardiac MRI Semantic Segmentation Study (Clean Baseline, Degradation Profiles, Failure Causality & Mitigation Recovery)

---

### Executive Overview & Metadata

| Attribute | Experimental Specification |
| :--- | :--- |
| **Model Architecture** | `LightweightUNet2D` (4-level encoder-decoder, 1,963,860 trainable parameters) |
| **Training Run** | 50 Epochs total (fresh initialization, seed `42`, AdamW, Cosine Annealing, DiceCELoss) |
| **Primary Checkpoint Evaluated** | **Epoch 44** (`checkpoints/50epochs/best_model_50epochs.pt`) |
| **Optimal Validation Dice** | **$0.8871$** (Best validation checkpoint across 50 epochs) |
| **Cryptographic SHA-256** | `a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38` |
| **Benchmark Dataset** | ACDC (Automated Cardiac Diagnosis Challenge) — 100 Patients |
| **Dataset Splits** | **Train**: 70 patients (664 slices) \| **Val**: 15 patients (142 slices) \| **Test**: 15 patients (131 slices) |
| **Evaluation Scope** | 15 strictly held-out test patients; End-Diastolic (ED) phase volumes (`frame01.nii.gz`) |
| **Target Anatomical Structures** | Class 1: Right Ventricle (RV) \| Class 2: Myocardium (MYO) \| Class 3: Left Ventricle (LV) |

---

## 1. Clean Baseline Performance ("Clean Mai Kitna Tha?")

Evaluated on the held-out test cohort of 15 patients (131 short-axis slices) without any artificial image corruptions or perturbations.

### 1.1 Summary Benchmark Metrics

| Metric Operational Level | Macro Foreground Dice | Macro Foreground IoU | RV Dice | MYO Dice | LV Dice |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2D Slice-Level (Macro Average across 131 slices)** | **$0.8297$** ($82.97\%$) | **$0.7617$** ($76.17\%$) | $0.8136$ | $0.7914$ | $0.8840$ |
| **3D Patient-Level (Voxel-Pooled across 15 patients)** | **$0.8958 \pm 0.0372$** | **$0.8161 \pm 0.0577$** | **$0.9019 \pm 0.0455$** | **$0.8434 \pm 0.0541$** | **$0.9420 \pm 0.0301$** |

### 1.2 Mathematical Insight: Why is Patient-Level (0.8958) Higher than Slice-Level (0.8297)?
In 2D slice-level evaluation, every single slice has an equal weight of $1/131 \approx 0.76\%$. Extreme apical and basal slices have very small cardiac structures (sometimes only a few pixels). A boundary error of just 5–10 pixels on an apical slice drops that slice's Dice to $0.50$ or $0.00$ (especially when the RV crescent disappears near the apex). 
In **Patient-Level 3D Voxel Pooling**, true positives ($TP$), false positives ($FP$), and false negatives ($FN$) are summed across all slices for that patient:
$$\text{Dice}_{p, c} = \frac{2 \sum_{s} TP_{s, c}}{2 \sum_{s} TP_{s, c} + \sum_{s} FP_{s, c} + \sum_{s} FN_{s, c}}$$
This correctly weights slices proportionally to anatomical volume mass, matching actual clinical cardiac MRI volume estimation protocol.

### 1.3 Patient-by-Patient Clean Baseline Breakdown

```
Patient       Mean FG Dice     RV Dice     MYO Dice     LV Dice
----------------------------------------------------------------
patient001       0.8932        0.8831       0.8487       0.9479
patient004       0.9178        0.9172       0.8687       0.9674
patient009       0.8827        0.8572       0.8619       0.9290
patient027       0.9088        0.9157       0.8811       0.9295
patient038       0.8026        0.7681       0.7419       0.8979
patient040       0.9291        0.9388       0.8920       0.9566
patient044       0.9131        0.9258       0.8631       0.9504
patient052       0.8857        0.8997       0.8105       0.9470
patient059       0.9361        0.9554       0.8893       0.9635
patient062       0.9383        0.9587       0.8950       0.9613
patient063       0.8830        0.8814       0.8228       0.9449
patient068       0.9207        0.9255       0.8821       0.9545
patient086       0.8345        0.8066       0.7699       0.9270
patient098       0.8810        0.8887       0.8344       0.9200
patient100       0.9096        0.9142       0.8546       0.9600
----------------------------------------------------------------
Cohort Mean      0.8958        0.9019       0.8434       0.9420
Std Dev (±)      0.0372        0.0455       0.0541       0.0301
```

---

## 2. Robustness Evaluation: What was Added, What Dropped & How Much? ("Kya Add Karne Se Kya Gira, Kitna Gira?")

Under Phase 4, the frozen Epoch 44 model was evaluated under 3 distinct corruption families across 3 severity tiers each (total 9 corrupted conditions + 1 clean baseline). Slices were corrupted directly in the tensor space without re-standardization.

### 2.1 Master Degradation Matrix (All 10 Experimental Conditions)

| Condition Name | Perturbation Type | Severity / Param | Patient 3D Dice | Slice 2D Dice | Absolute Delta ($\Delta$) | Patient-Wise % Drop | Cohort-Mean % Drop | Paired $t$ $p_{\text{Holm}}$ | Wilcoxon $p_{\text{Holm}}$ | Statistical Verdict ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`clean_baseline`** | None (Clean) | $0.0$ | **$0.8958$** | **$0.8297$** | $0.0000$ | $0.00\%$ | $0.00\%$ | — | — | **Gold Standard Baseline** |
| `gaussian_noise_mild` | Additive Noise | $\sigma=0.02$ | $0.8957$ | $0.8300$ | $-0.0001$ | $-0.01\%$ | $-0.01\%$ | $1.0000$ | $1.0000$ | Not Significant |
| `gaussian_noise_mod` | Additive Noise | $\sigma=0.05$ | $0.8948$ | $0.8263$ | $-0.0010$ | $-0.11\%$ | $-0.11\%$ | $0.3651$ | $0.4280$ | Not Significant |
| `gaussian_noise_sev` | Additive Noise | $\sigma=0.10$ | $0.8913$ | $0.8213$ | $-0.0044$ | $-0.51\%$ | $-0.50\%$ | $0.0741$ | $0.0256$ | Nominal Only ($p_{\text{raw}}=0.014$) |
| `gaussian_blur_mild` | Spatial Blur | $\sigma=0.75$ | $0.8860$ | $0.8259$ | $-0.0098$ | $-1.07\%$ | $-1.09\%$ | $0.0741$ | $0.0418$ | Nominal Only ($p_{\text{raw}}=0.012$) |
| `gaussian_blur_mod` | Spatial Blur | $\sigma=1.50$ | $0.8540$ | $0.7958$ | **$-0.0418$** | **$-4.62\%$** | **$-4.66\%$** | **$0.0029$** | **$0.0010$** | **Statistically Significant** |
| `gaussian_blur_sev` | Spatial Blur | $\sigma=2.50$ | **$0.7775$** | **$0.7133$** | **$-0.1182$** | **$-13.15\%$** | **$-13.20\%$** | **$0.0004$** | **$0.0005$** | **Catastrophic Failure ($p < 0.001$)** |
| `contrast_reduction_mild` | Dynamic Range | $\alpha=0.75$ | $0.8956$ | $0.8350$ | $-0.0002$ | $-0.01\%$ | $-0.02\%$ | $1.0000$ | $1.0000$ | Not Significant |
| `contrast_reduction_mod` | Dynamic Range | $\alpha=0.50$ | $0.8935$ | $0.8337$ | $-0.0023$ | $-0.24\%$ | $-0.26\%$ | $0.8571$ | $0.6879$ | Not Significant |
| `contrast_reduction_sev` | Dynamic Range | $\alpha=0.25$ | **$0.8158$** | **$0.7273$** | **$-0.0800$** | **$-8.85\%$** | **$-8.93\%$** | **$0.0035$** | **$0.0013$** | **Statistically Significant ($p < 0.005$)** |

### 2.2 Class-Wise Breakdown: Which Anatomy Suffered the Worst?

```
Condition               RV Dice (Δ)          MYO Dice (Δ)         LV Dice (Δ)
----------------------------------------------------------------------------------
clean_baseline          0.8136 (ref)         0.7914 (ref)         0.8840 (ref)
gaussian_noise_sev      0.8030 (-0.0106)     0.7799 (-0.0115)     0.8811 (-0.0029)
gaussian_blur_mod       0.7791 (-0.0345)     0.7423 (-0.0491)     0.8660 (-0.0180)
gaussian_blur_sev       0.6709 (-0.1427)     0.6571 (-0.1343)     0.8120 (-0.0720)
contrast_reduct_sev     0.6462 (-0.1674)     0.7051 (-0.0863)     0.8304 (-0.0536)
----------------------------------------------------------------------------------
```
- **Most Fragile Structure**: **Right Ventricle (RV)** collapsed by $-0.1674$ under severe contrast and $-0.1427$ under severe blur.
- **Second Fragile Structure**: **Myocardium (MYO)** collapsed by $-0.1343$ under severe blur.
- **Most Resilient Structure**: **Left Ventricle (LV)** blood pool retained $>0.81$ Dice under all extreme degradations.

---

## 3. Root Cause Analysis: Why Did Each Perturbation Drop Performance? ("Kyun Gira?")

### 3.1 Why Did Spatial Gaussian Blur Inflict the Largest Collapse (-13.15%)?
1. **Physical Scale vs Anatomical Dimensions**:
   - The adult myocardium wall thickness in end-diastole is typically $8–12\text{ mm}$ (approx. $6–12\text{ pixels}$ at $1.25\text{ mm/pixel}$ resolution).
   - An isotropic Gaussian blur with kernel parameter $\sigma = 2.50$ extends its spatial smoothing footprint over $6\sigma + 1 = 16\text{ pixels}$.
   - The smoothing filter physically averages voxel intensities from the bright LV blood cavity directly across the dark myocardium into the surrounding lung/chest wall tissue.
2. **Frequency Attenuation & Edge Annihilation**:
   - High spatial frequency components (which encode the sharp endocardial and epicardial border contours) are completely suppressed: $\hat{G}(f) = e^{-2\pi^2\sigma^2 f^2}$.
   - The shallow convolution filters of the U-Net rely on gradient step edges $(\nabla I)$. When gradients flatten into broad transitions, the network fails to classify boundary pixels, shrinking the myocardium ring or breaking it into disjoint segments.
3. **Right Ventricle Crescent Thinning**:
   - The right ventricular free wall is paper-thin ($3–5\text{ mm}$). Gaussian blur smears this thin structure into the surrounding low-intensity epicardial fat, dropping RV Dice to $0.6709$.

### 3.2 Why Did Severe Contrast Reduction ($\alpha=0.25$) Cause Non-Linear Failure (-8.85%)?
1. **Variance Compression & BatchNorm Mismatch**:
   - Contrast reduction was modeled as $I' = \alpha I + (1-\alpha)\mu$. At $\alpha=0.25$, the standard deviation of input slice tensors drops from $\sigma \approx 1.0$ down to $\sigma \approx 0.25$.
   - In Phase 4, corruptions were injected without re-standardizing the inputs.
   - The U-Net's `BatchNorm2d` layers maintain running mean $\hat{\mu} \approx 0$ and running variance $\hat{\sigma}^2 \approx 1.0$ learned during training on clean, standardized images.
   - When inputs with variance $0.0625$ ($0.25^2$) pass into these fixed batch norm layers, intermediate feature map activations are shifted into the near-zero non-responsive regime of the ReLU activation functions ($\max(0, x)$). Weak feature signals fail to cross the activation thresholds, extinguishing feature detection in deeper layers.
2. **Non-Linear Cliff Effect**:
   - Mild ($\alpha=0.75$, $-0.01\%$) and Moderate ($\alpha=0.50$, $-0.24\%$) contrast reductions show virtually zero impact. Only when contrast falls below a critical dynamic threshold ($\alpha=0.25$) do activations get zeroed out, creating an abrupt performance cliff.

### 3.3 Why Was the Model Remarkably Invariant to Gaussian Noise (-0.51%)?
1. **Hierarchical Downsampling as Low-Pass Filtering**:
   - Additive Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is spatially uncorrelated (independent and identically distributed white noise).
   - The U-Net architecture contains four $2\times 2$ MaxPool operations in its encoder. Max-pooling and successive $3\times 3$ convolutional kernel summations act as natural low-pass spatial integrators:
     $$\text{Var}\left(\frac{1}{K}\sum_{i=1}^K \epsilon_i\right) = \frac{\sigma^2}{K}$$
   - By the time activations reach bottleneck layer channels (resolution $16\times 16$), the high-frequency pixel noise has been almost entirely filtered out.
2. **50-Epoch Learned Semantic Representations**:
   - At Epoch 44, deep convolution kernels have converged to robust global anatomical shapes (circular LV, crescent RV) rather than relying on brittle local pixel intensity values.

---

## 4. Mitigation & Preprocessing Interventions: What Recovered & How Much? ("Preprocessing Karne Se Kitna Recover Hua?")

In Phase 5, pre-registered, validation-locked mitigation strategies were tested to recover lost performance without retraining the neural network.

### 4.1 Specification of Locked Mitigation Components

| Pipeline Component | Method / Parameters | Frozen Setting | Anatomical Purpose |
| :--- | :--- | :---: | :--- |
| **Preprocessing: CLAHE** | Contrast-Limited Adaptive Histogram Equalization | `clip_limit=0.01`, `grid=(16, 16)`, unit-scaling, re-standardization | Local contrast redistribution |
| **Preprocessing: Denoising**| Non-linear 2D Median Filter | $3 \times 3$ window (`size=3`) | Impulse/speckle noise suppression |
| **Post-Processing: LCC** | Largest Connected Component analysis | Min Area: RV $30\text{ px}$, MYO $50\text{ px}$, LV $30\text{ px}$ | Removes extracardiac spurious false positives |
| **Post-Processing: Cavity Safeguard** | Bounded morphological hole filling | **`max_hole_size = 16 pixels`** | **Preserves the central LV cavity (~706 px) from getting filled!** |

> [!IMPORTANT] The Myocardial Cavity Safeguard
> In cardiac segmentation, standard morphological `binary_fill_holes` fills the inner cavity of any closed ring. If applied naively to the myocardium, the entire central Left Ventricular blood cavity (~706 pixels on mid-ventricular slices) is converted into myocardium, catastrophically destroying segmentation validity! The locked safeguard enforces `max_hole_size = 16`, filling small intramyocardial voids while leaving the large ventricular lumen 100% intact.

---

### 4.2 Mitigation Results & Recovery Quantities

```
====================================================================================================
Condition               Unmitigated Dice    Mitigated Pipeline            Mitigated Dice     Recovery (Δ)
====================================================================================================
contrast_reduct_sev     0.8158              CLAHE Preprocessing Alone     0.8896             +0.0738 (+7.38%)
contrast_reduct_sev     0.8158              Combined (CLAHE + Morph LCC)  0.8921             +0.0763 (+7.63%)
gaussian_blur_sev       0.7775              Morphological LCC Alone       0.7915             +0.0140 (+1.40%)
gaussian_noise_sev      0.8913              Combined (Median + Morph LCC) 0.8910             -0.0003 (-0.03%)
clean_baseline          0.8958              Morphological LCC Alone       0.8980             +0.0022 (+0.22%)
====================================================================================================
```

#### Detailed Recovery Breakdown:
1. **Severe Contrast Reduction Recovery**:
   - Unmitigated: **$0.8158 \pm 0.0677$**
   - After Preprocessing (CLAHE): **$0.8896 \pm 0.0429$**
   - **Absolute Recovery**: **$+0.0738$ Dice points** ($+7.38\%$, $p_{\text{Holm}} = 0.0031$, Cohen's $d = 1.0647$).
   - **Recovery Percentage**: Rescued **$92.3\%$** of the lost performance ($0.0738 / 0.0800$), returning the model to **$99.3\%$** of its clean baseline ($0.8896 / 0.8958$).
   - With Morphological Post-Processing added: Reached **$0.8921$** ($\Delta = \mathbf{+0.0763}$, $95.4\%$ of lost performance recovered).

2. **Severe Gaussian Blur Recovery**:
   - Unmitigated: **$0.7775 \pm 0.0800$**
   - After Morphological Post-Processing: **$0.7915 \pm 0.0754$**
   - **Absolute Recovery**: **$+0.0140$ Dice points** ($+1.40\%$, unadjusted $p = 0.0231$, Cohen's $d = 0.5297$).
   - Note: Post-processing eliminates stray false-positive islands generated in the lungs, but cannot restore blurred internal boundaries.

3. **Gaussian Noise Mitigation (Median Filter)**:
   - Unmitigated: **$0.8913$** $\rightarrow$ Mitigated: **$0.8910$** ($\Delta = -0.0003$).
   - Result: Median filtering offered **zero recovery**. The model is already noise-invariant; spatial median filtering slightly blurs delicate myocardial edges.

---

## 5. The Critical Scientific Finding: Forensic Causal Audit of CLAHE

While initial Phase 5 testing attributed the $+0.0738$ contrast recovery to **CLAHE**, our forensic audit revealed a profound mathematical and algorithmic reality.

### 5.1 Mathematical Proof of Affine Invariance
The contrast corruption applied to normalized slice $I$ was:
$$I_{\text{corrupt}} = \alpha I + (1 - \alpha)\mu, \quad \alpha \in (0, 1]$$
Before applying CLAHE, the preprocessing pipeline applies `to_unit_interval(X)`:
$$T(X) = \frac{X - \min(X)}{\max(X) - \min(X)}$$
Substituting $I_{\text{corrupt}}$ into $T(X)$:
$$T(I_{\text{corrupt}}) = \frac{[\alpha I + (1 - \alpha)\mu] - [\alpha \min(I) + (1 - \alpha)\mu]}{[\alpha \max(I) + (1 - \alpha)\mu] - [\alpha \min(I) + (1 - \alpha)\mu]}$$
$$= \frac{\alpha [I - \min(I)]}{\alpha [\max(I) - \min(I)]} \equiv \frac{I - \min(I)}{\max(I) - \min(I)} = T(I)$$

Because $\alpha > 0$ cancels out identically in numerator and denominator, **$T(I_{\text{corrupt}}) \equiv T(I)$**.

### 5.2 Empirical Verification Across 131 Test Slices
Across all 131 test slices (8,585,216 voxels):
- The maximum tensor difference between clean and corrupted images after `to_unit_interval` was **$2.38 \times 10^{-7}$** (strictly float32 precision rounding error).
- After CLAHE, the output tensors were **$100.00\%$ bit-for-bit identical**.
- The model produced identical predictions across all 15 patients:
  $$\max_{p} |\text{Dice}_{\text{Clean+CLAHE}}(p) - \text{Dice}_{\text{Severe+CLAHE}}(p)| = \mathbf{0.00 \times 10^0}$$

### 5.3 Causal Factor Decomposition: Normalization vs CLAHE
To determine what actually recovered the performance, we ran an ablation test applying **pure linear Z-score standardization alone (without CLAHE)**:

$$\begin{aligned}
\Delta_{\text{total}} &= \text{Dice}(\text{Severe + CLAHE}) - \text{Dice}(\text{Severe Unmitigated}) = 0.8896 - 0.8158 = \mathbf{+0.0738} \\
\Delta_{\text{normalization}} &= \text{Dice}(\text{Severe + Z-Score Alone}) - \text{Dice}(\text{Severe Unmitigated}) = 0.8958 - 0.8158 = \mathbf{+0.0800} \\
\Delta_{\text{CLAHE}} &= \text{Dice}(\text{Severe + CLAHE}) - \text{Dice}(\text{Severe + Z-Score Alone}) = 0.8896 - 0.8958 = \mathbf{-0.0062}
\end{aligned}$$

$$\Delta_{\text{total}} = \Delta_{\text{normalization}} + \Delta_{\text{CLAHE}} = (+0.0800) + (-0.0062) = +0.0738$$

```
Component                       Contribution (Dice)    Percentage of Total Recovery
----------------------------------------------------------------------------------
Linear Dynamic Normalization    +0.0800                +108.4%
CLAHE (Histogram Equalization)  -0.0062                 -8.4% (Net Penalty)
----------------------------------------------------------------------------------
Total Observed Recovery         +0.0738                100.0%
```

> [!CAUTION] Crucial Clinical Conclusion
> **CLAHE did not rescue the model — standard linear normalization did!**  
> In fact, applying CLAHE introduced a $-0.0062$ penalty because histogram equalization artificially warps calibrated MR signal intensities. Pure linear Z-score standardization restores $100\%$ of the baseline accuracy ($0.8958$).

---

## 6. Negative Controls: Side Effects of Blindly Applying Preprocessing

Applying preprocessing filters without knowing the corruption state can actively damage clean or noisy MRI scans:

| Preprocessing Applied | Target Condition | Baseline Score | Treated Score | Side Effect Impact ($\Delta$) | Clinical Risk / Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CLAHE** | Clean Scans | $0.8958$ | $0.8896$ | **$-0.0062$** ($p=0.022$) | Alters calibrated tissue intensity ratios. |
| **CLAHE** | Severe Noise ($\sigma=0.10$) | $0.8913$ | $0.8744$ | **$-0.0169$** ($p=0.026$) | **Catastrophic noise amplification**; boosts high-frequency background noise. |
| **Median Filter** | Clean Scans | $0.8958$ | $0.8929$ | **$-0.0029$** | Edge attenuation; blurs sharp cardiac boundaries. |

---

## 7. Comprehensive Summary & Practical Recommendations

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ROBUSTNESS WORKFLOW SUMMARY                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│ 1. CLEAN SCAN BENCHMARK                                                               │
│    • 15 Test Patients (131 ED Slices)                                                 │
│    • 2D Slice Macro Dice: 0.8297 | 3D Patient Macro Dice: 0.8958 ± 0.0372             │
│    • LV: 0.9420 | RV: 0.9019 | MYO: 0.8434                                            │
│                                                                                        │
│ 2. VULNERABILITY RANKING (HOW MUCH DROPPED & WHY)                                      │
│    ① Spatial Blur (σ=2.50): Drops by -0.1182 (-13.15%) [Dice: 0.7775]                 │
│       -> Cause: Filter footprint (16 px) obliterates thin myocardium wall (8-12 px)    │
│    ② Contrast Reduction (α=0.25): Drops by -0.0800 (-8.85%) [Dice: 0.8158]            │
│       -> Cause: Variance collapse (σ≈0.25) pushes signals below BatchNorm/ReLU cutoff  │
│    ③ Gaussian Noise (σ=0.10): Drops by only -0.0044 (-0.51%) [Dice: 0.8913]           │
│       -> Cause: Inherent low-pass filtering in U-Net pooling & convolutions            │
│                                                                                        │
│ 3. RECOVERY & MITIGATION (WHAT RECOVERED & HOW)                                        │
│    • Contrast Recovery: +0.0738 to +0.0800 recovered (Back to 0.8896 - 0.8958)        │
│       -> 108.4% of recovery comes from Linear Normalization, NOT CLAHE!                │
│    • Blur Recovery: +0.0140 recovered via Morphological LCC (0.7775 -> 0.7915)         │
│    • Noise Recovery: Median filter does NOT help (decreases score by -0.0029 on clean)│
│                                                                                        │
│ 4. RECOMMENDED CLINICAL PIPELINE                                                       │
│    [Raw MRI Input] ─────────────────────────────────────────────────────────┐          │
│          │                                                                  │          │
│          ▼                                                                  ▼          │
│    [Per-Slice Linear Z-Score Normalization (μ=0, σ=1)]               [NO CLAHE]        │
│          │                                                                             │
│          ▼                                                                             │
│    [Epoch 44 LightweightUNet2D Inference]                                              │
│          │                                                                             │
│          ▼                                                                             │
│    [Per-Class Largest Connected Component (LCC)]                                       │
│    • RV Min Area: 30 px | MYO Min Area: 50 px | LV Min Area: 30 px                    │
│    • Myocardial Cavity Safeguard: max_hole_size = 16 px                                │
│          │                                                                             │
│          ▼                                                                             │
│    [Final Clinically Validated Segmentation: 0.8980 Patient 3D Dice]                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Verification Artifacts & References in Workspace
- **Complete Canonical Report**: [`reports/canonical_final_research_report.md`](file:///H:/Mini%20project/reports/canonical_final_research_report.md)
- **Phase 4 Summary & CSV**: [`reports/phase4_50epochs/robustness_summary.md`](file:///H:/Mini%20project/reports/phase4_50epochs/robustness_summary.md) & [`reports/phase4_50epochs/robustness_results.csv`](file:///H:/Mini%20project/reports/phase4_50epochs/robustness_results.csv)
- **Phase 5 Mitigation Report**: [`reports/phase5_validation_locked_mitigation_report.md`](file:///H:/Mini%20project/reports/phase5_validation_locked_mitigation_report.md)
- **CLAHE Forensic Findings**: [`reports/audit/clahe_forensic_findings.md`](file:///H:/Mini%20project/reports/audit/clahe_forensic_findings.md)
- **Visual Plots**:
  - Degradation Curve: [`experiments/phase4_robustness_50epoch_best_epoch44/severity_vs_dice.png`](file:///H:/Mini%20project/experiments/phase4_robustness_50epoch_best_epoch44/severity_vs_dice.png)
  - Class-wise Degradation: [`experiments/phase4_robustness_50epoch_best_epoch44/class_wise_degradation.png`](file:///H:/Mini%20project/experiments/phase4_robustness_50epoch_best_epoch44/class_wise_degradation.png)
  - Mitigation Comparison: [`experiments/phase5_validation_locked_mitigation/mitigation_comparison.png`](file:///H:/Mini%20project/experiments/phase5_validation_locked_mitigation/mitigation_comparison.png)
