"""Independent Phase 5 verification and statistical audit script.

Audits per-patient arrays, verifies affine contrast invariance, recomputes
paired differences, and applies Holm-Bonferroni correction across clearly defined
comparison families, separating raw from adjusted p-values.
"""

import json
import numpy as np
import scipy.stats as stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def step_down_holm(pvals):
    """Applies Holm-Bonferroni step-down procedure with monotonicity enforcement."""
    m = len(pvals)
    if m == 0:
        return []
    # 1-based rank sorting
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    running_max = 0.0
    adjusted_indexed = []
    for rank_idx, (orig_idx, p) in enumerate(indexed):
        mult = m - rank_idx  # Equivalent to m - i + 1 for 1-based rank i
        raw_adj = p * mult
        running_max = max(running_max, raw_adj)
        capped = min(1.0, float(running_max))
        adjusted_indexed.append((orig_idx, p, mult, capped))

    final_adj = [0.0] * m
    for orig_idx, p, mult, capped in adjusted_indexed:
        final_adj[orig_idx] = capped
    return final_adj, indexed


def main():
    json_path = PROJECT_ROOT / "reports" / "mitigation_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print(" AUDIT 1: INVESTIGATION OF IDENTICAL CLAHE SCORES ACROSS CONTRAST CONDITIONS")
    print("=" * 80)

    clahe = data["clahe_alone"]
    combined = data["combined_contrast"]
    contrast_keys = [
        "clean_baseline",
        "contrast_reduction_mild",
        "contrast_reduction_moderate",
        "contrast_reduction_severe",
    ]

    print("\nPer-Patient CLAHE Scores Across Contrast Conditions:")
    p_ids = sorted(list(clahe["clean_baseline"]["per_patient"].keys()))
    print(f"{'Patient ID':12s} | {'Clean+CLAHE':12s} | {'Mild+CLAHE':12s} | {'Mod+CLAHE':12s} | {'Sev+CLAHE':12s}")
    print("-" * 68)
    for p in p_ids:
        sc_c = clahe["clean_baseline"]["per_patient"][p]["mean_dice_foreground"]
        sc_mild = clahe["contrast_reduction_mild"]["per_patient"][p]["mean_dice_foreground"]
        sc_mod = clahe["contrast_reduction_moderate"]["per_patient"][p]["mean_dice_foreground"]
        sc_sev = clahe["contrast_reduction_severe"]["per_patient"][p]["mean_dice_foreground"]
        print(f"{p:12s} | {sc_c:12.6f} | {sc_mild:12.6f} | {sc_mod:12.6f} | {sc_sev:12.6f}")

    diff_sev_clean = np.max([
        abs(clahe["contrast_reduction_severe"]["per_patient"][p]["mean_dice_foreground"] -
            clahe["clean_baseline"]["per_patient"][p]["mean_dice_foreground"])
        for p in p_ids
    ])
    print(f"\nMax difference between Clean+CLAHE and Severe+CLAHE across all 15 patients: {diff_sev_clean:.2e}")

    print("\n--- Mathematical Mechanism Verification ---")
    print("Contrast corruption: I_corrupt = mu + alpha * (I - mu) = alpha * I + (1 - alpha) * mu")
    print("Min-max normalization in to_unit_interval: (I_corrupt - min) / (max - min)")
    print("Since alpha > 0 is scalar, [alpha*I - alpha*min] / [alpha*max - alpha*min] = (I - min) / (max - min).")
    print("Conclusion: Min-max scaling in to_unit_interval is analytically invariant to any positive linear")
    print("contrast scaling factor alpha. CLAHE receives mathematically identical [0, 1] input arrays.")

    print("\n" + "=" * 80)
    print(" AUDIT 2: RECOMPUTATION OF PAIRED DIFFERENCES & P-VALUES (POST-PROCESSING ALONE)")
    print("=" * 80)

    postproc = data["postproc_alone"]
    unmit = data["unmitigated"]
    all_cond_keys = list(postproc.keys())

    # Family: Post-Processing Alone across all 10 conditions (m=10)
    post_diffs = []
    post_t_p = []
    post_w_p = []

    print(f"{'Condition ID':28s} | {'Unmit Dice':10s} | {'Post Dice':10s} | {'Delta':10s} | {'t-test p_raw':12s} | {'Wilcoxon p_raw':12s}")
    print("-" * 92)

    for cid in all_cond_keys:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        p_vals = [postproc[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(p_vals) - np.array(u_vals)
        mean_diff = float(np.mean(diff))
        post_diffs.append(mean_diff)

        # Statistical tests
        if np.allclose(diff, 0.0):
            t_pval, w_pval = 1.0, 1.0
        else:
            t_res = stats.ttest_rel(p_vals, u_vals)
            t_pval = float(t_res.pvalue)
            w_res = stats.wilcoxon(p_vals, u_vals)
            w_pval = float(w_res.pvalue)
        post_t_p.append(t_pval)
        post_w_p.append(w_pval)

        u_mean = unmit[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        p_mean = postproc[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        print(f"{cid:28s} | {u_mean:10.4f} | {p_mean:10.4f} | {mean_diff:+10.4f} | {t_pval:12.4e} | {w_pval:12.4e}")

    # Holm-Bonferroni correction across m=10 post-processing comparisons
    t_adj_post, _ = step_down_holm(post_t_p)
    w_adj_post, _ = step_down_holm(post_w_p)

    print("\nHolm-Adjusted P-values for Post-Processing Family (m=10):")
    print(f"{'Condition ID':28s} | {'t-test p_raw':12s} | {'t-test p_Holm':12s} | {'Wilcoxon p_raw':12s} | {'Wilcoxon p_Holm':12s} | {'Decision (alpha=0.05)':22s}")
    print("-" * 115)
    for i, cid in enumerate(all_cond_keys):
        t_raw = post_t_p[i]
        t_adj = t_adj_post[i]
        w_raw = post_w_p[i]
        w_adj = w_adj_post[i]
        sig = "Significant" if w_adj < 0.05 or t_adj < 0.05 else "Not Significant"
        if (w_raw < 0.05 or t_raw < 0.05) and sig == "Not Significant":
            sig = "Nominal only (unadj)"
        print(f"{cid:28s} | {t_raw:12.4e} | {t_adj:12.4e} | {w_raw:12.4e} | {w_adj:12.4e} | {sig:22s}")

    print("\n" + "=" * 80)
    print(" AUDIT 3: CONTRAST RESTORATION PAIRED COMPARISONS (CLAHE & COMBINED)")
    print("=" * 80)

    clahe_keys = ["contrast_reduction_severe", "contrast_reduction_moderate", "contrast_reduction_mild"]
    clahe_t_p = []
    clahe_w_p = []
    comb_t_p = []
    comb_w_p = []

    print("\nCLAHE Alone on Contrast (m=3 family):")
    print(f"{'Condition ID':28s} | {'Unmit Dice':10s} | {'CLAHE Dice':10s} | {'Delta':10s} | {'t-test p_raw':12s} | {'Wilcoxon p_raw':12s}")
    print("-" * 92)
    for cid in clahe_keys:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        c_vals = [clahe[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(c_vals) - np.array(u_vals)
        mean_diff = float(np.mean(diff))
        t_res = stats.ttest_rel(c_vals, u_vals)
        w_res = stats.wilcoxon(c_vals, u_vals)
        t_pval = float(t_res.pvalue)
        w_pval = float(w_res.pvalue)
        clahe_t_p.append(t_pval)
        clahe_w_p.append(w_pval)
        u_mean = unmit[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        c_mean = clahe[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        print(f"{cid:28s} | {u_mean:10.4f} | {c_mean:10.4f} | {mean_diff:+10.4f} | {t_pval:12.4e} | {w_pval:12.4e}")

    t_adj_clahe, _ = step_down_holm(clahe_t_p)
    w_adj_clahe, _ = step_down_holm(clahe_w_p)

    print("\nHolm-Adjusted P-values for CLAHE Contrast Family (m=3):")
    print(f"{'Condition ID':28s} | {'t-test p_raw':12s} | {'t-test p_Holm':12s} | {'Wilcoxon p_raw':12s} | {'Wilcoxon p_Holm':12s} | {'Decision (alpha=0.05)':22s}")
    print("-" * 115)
    for i, cid in enumerate(clahe_keys):
        t_raw = clahe_t_p[i]
        t_adj = t_adj_clahe[i]
        w_raw = clahe_w_p[i]
        w_adj = w_adj_clahe[i]
        sig = "Significant" if w_adj < 0.05 or t_adj < 0.05 else "Not Significant"
        print(f"{cid:28s} | {t_raw:12.4e} | {t_adj:12.4e} | {w_raw:12.4e} | {w_adj:12.4e} | {sig:22s}")

    print("\nCombined (CLAHE + Post-Processing) on Contrast (m=3 family):")
    comb = data["combined_contrast"]
    for cid in clahe_keys:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        cb_vals = [comb[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(cb_vals) - np.array(u_vals)
        mean_diff = float(np.mean(diff))
        t_res = stats.ttest_rel(cb_vals, u_vals)
        w_res = stats.wilcoxon(cb_vals, u_vals)
        t_pval = float(t_res.pvalue)
        w_pval = float(w_res.pvalue)
        comb_t_p.append(t_pval)
        comb_w_p.append(w_pval)
        u_mean = unmit[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        cb_mean = comb[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        print(f"{cid:28s} | {u_mean:10.4f} | {cb_mean:10.4f} | {mean_diff:+10.4f} | {t_pval:12.4e} | {w_pval:12.4e}")

    t_adj_comb, _ = step_down_holm(comb_t_p)
    w_adj_comb, _ = step_down_holm(comb_w_p)

    print("\nHolm-Adjusted P-values for Combined Contrast Family (m=3):")
    print(f"{'Condition ID':28s} | {'t-test p_raw':12s} | {'t-test p_Holm':12s} | {'Wilcoxon p_raw':12s} | {'Wilcoxon p_Holm':12s} | {'Decision (alpha=0.05)':22s}")
    print("-" * 115)
    for i, cid in enumerate(clahe_keys):
        t_raw = comb_t_p[i]
        t_adj = t_adj_comb[i]
        w_raw = comb_w_p[i]
        w_adj = w_adj_comb[i]
        sig = "Significant" if w_adj < 0.05 or t_adj < 0.05 else "Not Significant"
        print(f"{cid:28s} | {t_raw:12.4e} | {t_adj:12.4e} | {w_raw:12.4e} | {w_adj:12.4e} | {sig:22s}")


if __name__ == "__main__":
    main()
