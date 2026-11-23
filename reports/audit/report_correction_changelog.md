# Report Correction & Scientific Change Log

**Project**: ACDC Cardiac MRI Segmentation — LightweightUNet2D  
**Repository**: `VinayakaK2/MRI-Image-Segmentation`  
**Audit Date**: November 6, 2026  
**Auditor**: Senior Medical Imaging Research Scientist, ML Engineer & Statistical Auditor  
**Purpose**: Formal audit log documenting every historical documentation discrepancy, empirical correction, claim retraction, and document supersession across the repository.

---

## 1. Document Inventory & Authoritative Hierarchy

| Document Path | Original Creation Date | Original Scope | Evaluated Model Checkpoint | Current Status | Superseded By |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `reports/phase3_1_baseline_audit.md` | 2026-10-21 | Phase 3 baseline architecture audit | `best_model_frozen.pt` (3 ep) | Historical Reference | `canonical_final_research_report.md` |
| `reports/phase3_50epochs_evaluation_report.md` | 2026-10-29 | 50-epoch GPU training completion | `best_model_50epochs.pt` (44 ep) | Verified Historical | `canonical_final_research_report.md` |
| `reports/robustness_summary.md` | 2026-10-21 | Original 3-epoch Phase 4 robustness | `best_model_frozen.pt` (3 ep) | Historical Baseline | `canonical_final_research_report.md` |
| `reports/phase4_50epochs/robustness_summary.md` | 2026-11-04 | 50-epoch Phase 4 robustness | `best_model_50epochs.pt` (44 ep) | Verified Phase 4 | `canonical_final_research_report.md` |
| `reports/phase4_pre_phase5_verification_audit.md` | 2026-11-04 | Phase 4 verification gate | Both checkpoints | Verified Audit | `canonical_final_research_report.md` |
| `reports/phase5_mitigation_report.md` | 2026-10-21 | 3-epoch Phase 5 mitigation | `best_model_frozen.pt` (3 ep) | Historical Reference | `canonical_final_research_report.md` |
| `reports/phase5_scientific_audit_report.md` | 2026-10-22 | 3-epoch Phase 5 scientific audit | `best_model_frozen.pt` (3 ep) | Historical Reference | `canonical_final_research_report.md` |
| `reports/historical/phase5_validation_locked_mitigation_report_v1.md` | 2026-11-05 | 50-epoch Phase 5 mitigation (pre-audit) | `best_model_50epochs.pt` (44 ep) | Historical Archive | `canonical_final_research_report.md` |
| `reports/phase5_validation_locked_mitigation_report.md` | 2026-11-05 | 50-epoch Phase 5 mitigation (updated) | `best_model_50epochs.pt` (44 ep) | Updated / Synchronized | `canonical_final_research_report.md` |
| `reports/phase5_clahe_invariance_audit.md` | 2026-11-05 | CLAHE forensic invariance audit | `best_model_50epochs.pt` (44 ep) | Verified Forensic | `canonical_final_research_report.md` |
| `reports/canonical_final_research_report.md` | 2026-11-06 | Unified, audited, authoritative research report | Both checkpoints (3 ep & 44 ep) | **CANONICAL AUTHORITATIVE** | **Current Gold Standard** |

---

## 2. Detailed Item-by-Item Correction Log

### Item 1: Historical 3-Epoch Baseline Values (Transcription Error Correction)
- **Affected Documents**: Informal user prompts, early drafts of Phase 4 pre-audit discussion.
- **Original Claim**:
  - 3-epoch clean standard deviation: cited as $0.0886$.
  - 3-epoch severe contrast: cited as $0.5794 \pm 0.1017$.
  - 3-epoch severe blur: cited as $0.6121 \pm 0.0825$.
- **Actual Evidence Ground Truth** (`reports/robustness_results.json`):
  - 3-epoch clean Dice: **$0.7355 \pm 0.1074$** (std $= 0.1074$).
  - 3-epoch severe contrast: **$0.0405 \pm 0.0446$** (near-total model collapse).
  - 3-epoch severe blur: **$0.6718 \pm 0.1147$**.
- **Root Cause**:
  1. $0.0886$ was the severe contrast score of an individual patient (`patient009` in `reports/robustness_summary.md`), which was mistakenly copied into the summary column for clean standard deviation.
  2. $0.5794$ and $0.6121$ were transcription errors that never existed in the raw JSON output.
- **Correction Applied**: All comparisons in `reports/canonical_final_research_report.md` and `reports/audit/numerical_results_verification.md` use the verified ground-truth values ($0.0405$ severe contrast, $0.6718$ severe blur, $0.1074$ clean std).

---

### Item 2: Causal Attribution of CLAHE Contrast Mitigation (Retraction & Correction)
- **Affected Documents**: `reports/phase5_validation_locked_mitigation_report.md` (Sections 1, 4, 6).
- **Original Claim**: "CLAHE rescues patient Dice under severe contrast reduction from $0.8158$ to $0.8896$ (+0.0738 Dice, $p < 0.005$), achieving $99.3\%$ recovery back to clean baseline."
- **Actual Evidence Ground Truth** (`experiments/phase5_clahe_invariance_audit/`):
  - In `src/preprocessing/restoration.py`, `apply_clahe` calls `to_unit_interval` as its first step.
  - Linear contrast reduction is an affine transform: $I' = \alpha I + (1 - \alpha)\mu$.
  - Min-max normalization analytically cancels $\alpha$: $\frac{\alpha(I - \min I)}{\alpha(\max I - \min I)} = \frac{I - \min I}{\max I - \min I}$.
  - The input tensors to CLAHE on clean vs corrupted images are identical down to float32 machine epsilon ($2.38 \times 10^{-7}$).
  - Pure linear Z-score re-standardization alone achieves **$0.8958 \pm 0.0372$** ($100.0\%$ recovery).
  - Applying CLAHE on top of normalized inputs results in **$0.8896 \pm 0.0429$** (a net penalty of $-0.0062$).
  - The $99.3\%$ ratio ($0.8896 / 0.8958$) measures CLAHE distortion on clean data, not recovery from corruption.
- **Correction Applied**:
  - Retracted claim that CLAHE's histogram equalization provided the recovery.
  - Corrected finding: Linear normalization cancels affine contrast scaling; CLAHE itself imposes a $-0.0062$ penalty.
  - Recommended pipeline updated to linear standardization + morphological post-processing ($0.8980 \pm 0.0357$).

---

### Item 3: Multiple Comparison Correction Scope Clarification
- **Affected Documents**: `reports/phase5_validation_locked_mitigation_report.md` (Sections 1, 9).
- **Original Phrasing**: Implied global FWER control across all 19 hypotheses.
- **Actual Evidence Ground Truth**:
  - Pre-registered protocol in `scripts/evaluate_mitigation_test.py` and `scripts/audit_phase5_metrics.py` defines **4 separate families** ($m=10, m=3, m=3, m=3$).
  - Step-down Holm-Bonferroni correction was applied independently within each family.
- **Correction Applied**: Explicitly documented that FWER control is family-local at $\alpha = 0.05$ across the 4 pre-specified families, not global across all 19 hypotheses.

---

### Item 4: Checkpoint Comparison Framing (Causal vs Descriptive)
- **Affected Documents**: Phase 4 and Phase 5 comparison sections.
- **Original Phrasing**: Risk of implying that 50 epochs of training was the sole causal factor for all robustness gains.
- **Actual Evidence Ground Truth**:
  - 3-epoch model: CPU trained, initial exploratory run, seed 42.
  - 50-epoch model: GPU trained, fresh random initialization, Epoch 44 checkpoint selected by validation Dice.
- **Correction Applied**: Comparison is explicitly framed as a **descriptive comparison of two distinct checkpoint artifacts**, not a controlled causal attribution to epoch count alone.

---

### Item 5: Clinical Generalization Scope Boundary
- **Affected Documents**: General conclusions in earlier reports.
- **Original Phrasing**: Mentioned general robustness without explicit scanner physics boundaries.
- **Correction Applied**: Added explicit limitations: synthetic mathematical corruptions (Gaussian noise, Gaussian blur, global linear contrast) do not capture real clinical MRI scanner physics (spatially heterogeneous $B_1$ transmit field inhomogeneities, patient respiratory/cardiac motion ghosting, flow voids, slice misregistration, or multi-vendor reconstruction differences).

---

## 3. Supersession Declaration

The unified document:
[`reports/canonical_final_research_report.md`](file:///H:/Mini%20project/reports/canonical_final_research_report.md)

is hereby designated as the **Sole Canonical Authoritative Final Research Report** for the ACDC Cardiac MRI Segmentation project.

All prior reports remain intact in their respective directories (`reports/`, `reports/historical/`, `experiments/`) to preserve the historical audit trail, but are scientifically superseded by the canonical report.
