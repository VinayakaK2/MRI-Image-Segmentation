# Phase 3 Baseline Training Report — 50 Completed Epochs

**Project**: `VinayakaK2/MRI-Image-Segmentation`  
**Dataset**: ACDC Cardiac Cine-MRI (100 patients: 70 train, 15 val, 15 test)  
**Model Architecture**: `LightweightUNet2D` (1,963,860 parameters)  
**Execution Environment**: Google Colab (NVIDIA Tesla T4 GPU, 14.56 GB VRAM, CUDA 12.2 / PyTorch)  
**Execution Status**: **COMPLETED (50/50 Epochs)**  
**Laptop CPU Usage During Training**: **0% (Offloaded 100% to Google Colab GPU)**  
**Frozen Model Integrity**: `checkpoints/best_model_frozen.pt` **UNTOUCHED** (SHA-256: `e7bfa884bef0f4eafce30cd6948c8d60fc4e7c8c8ca2d0bb0e83f2977fc29c02`)

---

## 1. Executive Summary

Phase 3 baseline training has been successfully executed to full completion (50 epochs) on an isolated cloud GPU (Google Colab Tesla T4). The 3-epoch partial CPU checkpoint has been preserved as a historical reference, and all new outputs have been generated under the isolated `50epochs` namespace.

### Key Milestones Achieved:
1. **Convergence**: Reached a best validation foreground Dice of **0.8871** at Epoch 44 (up from 0.6884 at Epoch 3).
2. **Held-Out Test Performance**: Achieved a Macro Foreground Dice of **0.8297** across the 15 held-out test patients (an absolute improvement of **+0.094 to +0.125** over the 3-epoch baseline).
3. **Safety & Non-Regression**: 
   - `checkpoints/best_model_frozen.pt` remains unmodified.
   - All 95 tests across 15 test suites pass (`pytest tests/`).
   - Zero files staged, zero Git commits created, zero pushes.

---

## 2. Quantitative Performance Comparison

| Metric | 3-Epoch Partial Baseline | 50-Epoch Completed Baseline | Absolute Gain ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Best Validation FG Dice** | 0.6884 (Epoch 3) | **0.8871 (Epoch 44)** | **+0.1987** | **+28.9%** |
| **Test Macro FG Dice** | 0.7043–0.7355 | **0.8297** | **+0.0942 to +0.1254** | **+12.8% to +17.8%** |
| **Test Macro FG IoU** | ~0.5828 | **0.7617** | **+0.1789** | **+30.7%** |
| **Test RV Dice** | ~0.6812 | **0.8136** | **+0.1324** | **+19.4%** |
| **Test MYO Dice** | ~0.5793 | **0.7914** | **+0.2121** | **+36.6%** |
| **Test LV Dice** | ~0.8046 | **0.8840** | **+0.0794** | **+9.9%** |
| **Test Background Dice** | ~0.9950 | **0.9975** | **+0.0025** | **+0.3%** |

---

## 3. Training Trajectory & Convergence Analysis

The model was trained deterministically with seed `42` using `DiceCELoss` and `AdamW` ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) scheduled via `CosineAnnealingLR` ($\text{min\_lr} = 10^{-5}$) over 50 epochs.

### Milestone Epoch Progression:
- **Epoch 01**: Train Loss: `1.3751` | Val Loss: `1.1591` | Val FG Dice: `0.1878`
- **Epoch 03**: Train Loss: `0.6514` | Val Loss: `0.5191` | Val FG Dice: `0.6462`
- **Epoch 05**: Train Loss: `0.3991` | Val Loss: `0.3046` | Val FG Dice: `0.7704`
- **Epoch 10**: Train Loss: `0.2073` | Val Loss: `0.1714` | Val FG Dice: `0.8351`
- **Epoch 15**: Train Loss: `0.1593` | Val Loss: `0.1454` | Val FG Dice: `0.8564`
- **Epoch 22**: Train Loss: `0.1322` | Val Loss: `0.1229` | Val FG Dice: `0.8699`
- **Epoch 28**: Train Loss: `0.1180` | Val Loss: `0.1183` | Val FG Dice: `0.8823`
- **Epoch 36**: Train Loss: `0.1097` | Val Loss: `0.1149` | Val FG Dice: `0.8854`
- **Epoch 44 (*BEST*)**: Train Loss: `0.1052` | Val Loss: `0.1135` | Val FG Dice: **`0.8871`**
  - **RV Dice**: `0.8601`
  - **MYO Dice**: `0.8624`
  - **LV Dice**: `0.9382`
- **Epoch 50 (Final)**: Train Loss: `0.1046` | Val Loss: `0.1143` | Val FG Dice: `0.8861`

Total GPU Training Wall Time: **819.9 seconds (~13.6 minutes)** on NVIDIA Tesla T4.

---

## 4. Held-Out 15-Patient Test Cohort Evaluation

Evaluated deterministically on the 131 2D slices across the 15 held-out test patients using the optimal checkpoint (`best_model_50epochs.pt`, Epoch 44):

```json
{
  "model_architecture": "LightweightUNet2D",
  "parameters": {
    "total": 1963860,
    "trainable": 1963860
  },
  "best_epoch": 44,
  "best_val_foreground_dice": 0.8871069795435059,
  "total_completed_epochs": 50,
  "test_metrics": {
    "dice_background": 0.9975410481689598,
    "iou_background": 0.9951261494175428,
    "dice_rv": 0.8135713139212731,
    "iou_rv": 0.7464506025346899,
    "dice_myo": 0.7914215379755563,
    "iou_myo": 0.6942091937393524,
    "dice_lv": 0.8840218776765637,
    "iou_lv": 0.8443821903401774,
    "mean_dice_foreground": 0.8296715765244638,
    "mean_iou_foreground": 0.76168066220474,
    "mean_dice_all": 0.8716389444355882,
    "mean_iou_all": 0.820042034007941
  }
}
```

---

## 5. Artifact Verification & Checksums

### Local Workspace Artifacts:
| File | Status | Description |
| :--- | :---: | :--- |
| `reports/training_history_50epochs.json` | **VERIFIED** | Full 50-epoch loss, learning rate, per-class Dice, and IoU trajectory |
| `reports/baseline_test_evaluation_50epochs.json` | **VERIFIED** | Complete held-out test cohort metrics for the 50-epoch checkpoint |
| `checkpoints/best_model_frozen.pt` | **VERIFIED** | Unmodified original frozen checkpoint (`e7bfa884bef0f4eafce30cd6948c8d60fc4e7c8c8ca2d0bb0e83f2977fc29c02`) |
| `tests/test_trainer_resume.py` | **VERIFIED** | 3 unit tests verifying checkpoint state loading, optimizer recovery, and suffix isolation |

### Google Colab GPU Checkpoints:
| Checkpoint File | Size | SHA-256 Checksum | Location |
| :--- | :---: | :---: | :--- |
| `best_model_50epochs.pt` | 23,680,419 B | `a4c320f19bf3f74750099badb9fda41d0c3b1abfbd68e4163620fd349d5b4e38` | `/content/MRI-Image-Segmentation/checkpoints/` |
| `last_checkpoint_50epochs.pt` | 23,682,863 B | `5565e6c1c5264d57d3054ef293db5339e8fd94a3e1fc5e735e77e3fd73115081` | `/content/MRI-Image-Segmentation/checkpoints/` |

---

## 6. Software & Pipeline Enhancements

To make the training pipeline fully reproducible, resumable, and safe:
1. **`src/training/trainer.py`**:
   - Implemented `resume_from_checkpoint(checkpoint_path)`: Restores `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, completed epoch count, history records, and early stopping patience.
   - Added `checkpoint_suffix` parameter: Directs checkpoints and reports to isolated filenames (e.g. `*_50epochs.pt`, `*_50epochs.json`).
   - Added `early_stopping_enabled` toggle: Allows unconstrained 50-epoch execution when early stopping is disabled.
   - Bundled full history into saved checkpoint dictionary for self-contained portability.
2. **`scripts/train_baseline.py`**:
   - Added CLI arguments: `--resume`, `--checkpoint-suffix`, `--no-early-stopping`, and `--patience`.
   - Wired suffix isolation into checkpoint loading and test report export.
3. **Automated Test Suite**:
   - Added `tests/test_trainer_resume.py` covering resumption, error handling on missing checkpoints, and suffix isolation.
   - Entire regression suite: **95 passed, 0 failed, 1 warning** in 25.53s.

---

## 7. Compliance Checklist

- [x] Baseline training executed to full 50 epochs
- [x] Zero training workload on user laptop CPU (executed on Google Colab Tesla T4 GPU)
- [x] `checkpoints/best_model_frozen.pt` strictly preserved and untouched
- [x] Isolated suffixes used (`50epochs`) to protect existing Phase 4/5 outputs
- [x] All 50 epochs recorded in `training_history_50epochs.json`
- [x] 15-patient held-out test cohort evaluated and recorded in `baseline_test_evaluation_50epochs.json`
- [x] Full regression test suite passing (95/95 tests)
- [x] No unauthorized Git staging, commits, or pushes
