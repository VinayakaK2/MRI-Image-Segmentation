"""Independent audit of patient-level metrics, relative drops, and Wilcoxon adjustments."""

import json
import numpy as np
import scipy.stats as stats
from pathlib import Path

def step_down_holm(pvals):
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    res = []
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        mult = m - rank
        raw_adj = p * mult
        running_max = max(running_max, raw_adj)
        capped = min(1.0, float(running_max))
        res.append((orig_idx, p, mult, raw_adj, capped))
    
    final_adj = [0.0] * m
    for orig_idx, p, mult, raw_adj, capped in res:
        final_adj[orig_idx] = capped
    return final_adj, res

def main():
    json_path = Path("reports/robustness_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Audit Patient-Level Dice and Relative Drops
    clean_patients = data["conditions"]["clean_baseline"]["patient_level"]
    sev_contrast_patients = data["conditions"]["contrast_reduction_severe"]["patient_level"]

    print("=== 1. AUDIT OF PATIENT-LEVEL DICE (Severe Contrast Reduction) ===")
    clean_dices = []
    corrupt_dices = []
    patient_ids = sorted(list(clean_patients.keys()))

    print(f"{'Patient ID':12s} | {'Clean Dice':10s} | {'Corrupt Dice':12s} | {'Delta Abs':12s} | {'Rel Drop (%)':12s}")
    print("-" * 70)

    patient_rel_drops = []
    for p in patient_ids:
        c = clean_patients[p]["mean_dice_foreground"]
        k = sev_contrast_patients[p]["mean_dice_foreground"]
        delta = k - c
        rel_pct = (delta / c) * 100.0 if c > 0 else 0.0
        clean_dices.append(c)
        corrupt_dices.append(k)
        patient_rel_drops.append(rel_pct)
        print(f"{p:12s} | {c:10.4f} | {k:12.4f} | {delta:12.4f} | {rel_pct:11.2f}%")

    clean_dices = np.array(clean_dices)
    corrupt_dices = np.array(corrupt_dices)
    patient_rel_drops = np.array(patient_rel_drops)

    print("-" * 70)
    print(f"Number of test patients: {len(patient_ids)}")
    print(f"Min clean patient Dice: {clean_dices.min():.4f}")
    print(f"Max clean patient Dice: {clean_dices.max():.4f}")
    print(f"Any clean patient Dice == 0: {(clean_dices == 0).any()}")
    print(f"Mean clean patient Dice: {clean_dices.mean():.4f}")
    print(f"Mean corrupt patient Dice: {corrupt_dices.mean():.4f}")
    print(f"Mean of 15 patient-wise relative drops: {patient_rel_drops.mean():.4f}%")
    print(f"Relative drop of cohort means: {((corrupt_dices.mean() - clean_dices.mean()) / clean_dices.mean() * 100.0):.4f}%")

    # 2. Audit Wilcoxon and Holm Adjustments
    print("\n=== 2. AUDIT OF WILCOXON P-VALUES ACROSS 9 CORRUPTIONS ===")
    paired = data["paired_comparisons"]
    cond_names = list(paired.keys())

    raw_t = [paired[c]["paired_t_pvalue"] for c in cond_names]
    raw_w = [paired[c]["wilcoxon_pvalue"] for c in cond_names]

    adj_w, audit_w = step_down_holm(raw_w)
    adj_t, audit_t = step_down_holm(raw_t)

    print("\nWilcoxon Step-Down Audit Trail (sorted by raw p-value):")
    print(f"{'Rank':4s} | {'Orig Index':10s} | {'Condition':30s} | {'Raw p':10s} | {'Mult (m-k)':10s} | {'Raw * Mult':12s} | {'Holm Adjusted':12s}")
    print("-" * 105)
    for rank, (orig_idx, p, mult, adj, capped) in enumerate(audit_w):
        c_name = cond_names[orig_idx]
        print(f"{rank:4d} | {orig_idx:10d} | {c_name:30s} | {p:10.4e} | {mult:10d} | {adj:12.4e} | {capped:12.4e}")

    print("\nComparison between raw_w and adj_w in original condition order:")
    print(f"{'Condition':30s} | {'Raw Wilcoxon':14s} | {'Holm Wilcoxon':14s} | {'Sig (alpha=0.05)':18s}")
    print("-" * 80)
    for c_name, r_w, a_w in zip(cond_names, raw_w, adj_w):
        sig = "Significant" if a_w < 0.05 else "Not Significant"
        print(f"{c_name:30s} | {r_w:14.4e} | {a_w:14.4e} | {sig:18s}")

if __name__ == "__main__":
    main()
