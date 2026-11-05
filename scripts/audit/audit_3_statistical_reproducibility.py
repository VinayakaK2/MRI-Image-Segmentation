"""Audit 3: Independent Reproduction of Patient-Level Statistics and Holm Correction."""

import json
from pathlib import Path
import sys
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def independent_holm_stepdown(raw_pvals):
    """Transparent independent manual implementation of Holm-Bonferroni step-down correction.
    
    Given m unadjusted p-values:
    1. Sort ascending: p_(1) <= p_(2) <= ... <= p_(m)
    2. Multiplier for rank k (1-indexed) is (m - k + 1)
    3. Running maximum monotonicity: p_adj,(k) = max(p_adj,(k-1), (m - k + 1) * p_(k))
    4. Capping at 1.0: min(1.0, p_adj)
    5. Restore original index order.
    """
    m = len(raw_pvals)
    if m == 0:
        return []
    
    indexed = sorted(enumerate(raw_pvals), key=lambda x: x[1])
    audit_trail = []
    running_max = 0.0
    
    adjusted_sorted = [0.0] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        mult = m - rank
        raw_adj = p * mult
        running_max = max(running_max, raw_adj)
        capped = min(1.0, float(running_max))
        adjusted_sorted[rank] = capped
        audit_trail.append({
            "rank": rank + 1,
            "orig_idx": orig_idx,
            "raw_p": float(p),
            "multiplier": mult,
            "raw_times_mult": float(raw_adj),
            "running_max": float(running_max),
            "holm_adjusted": capped,
        })
        
    final_adj = [0.0] * m
    for (orig_idx, _), adj_p in zip(indexed, adjusted_sorted):
        final_adj[orig_idx] = adj_p
        
    return final_adj, audit_trail


def run_audit_3():
    print("======================================================================")
    print("   AUDIT 3: PATIENT-LEVEL STATISTICS & HOLM CORRECTION REPRODUCIBILITY")
    print("======================================================================")

    json_path = PROJECT_ROOT / "experiments" / "phase4_robustness_50epoch_best_epoch44" / "robustness_results.json"
    if not json_path.exists():
        json_path = PROJECT_ROOT / "reports" / "phase4_50epochs" / "robustness_results.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_patient_dict = data["conditions"]["clean_baseline"]["patient_level"]
    reported_paired = data.get("paired_comparisons", {})

    patient_ids = sorted(list(clean_patient_dict.keys()))
    clean_scores = np.array([clean_patient_dict[p]["mean_dice_foreground"] for p in patient_ids], dtype=np.float64)

    print(f"Number of test patients: {len(patient_ids)}")
    print(f"Test Patient IDs: {patient_ids}")
    print(f"Clean Patient Scores:\n{clean_scores}\n")

    corruption_keys = [
        "gaussian_noise_mild",
        "gaussian_noise_moderate",
        "gaussian_noise_severe",
        "gaussian_blur_mild",
        "gaussian_blur_moderate",
        "gaussian_blur_severe",
        "contrast_reduction_mild",
        "contrast_reduction_moderate",
        "contrast_reduction_severe",
    ]

    reproduced_stats = {}
    raw_t_pvals = []
    raw_w_pvals = []

    print("--- INDEPENDENT PAIRED STATISTICAL RECALCULATION ---")
    for cond in corruption_keys:
        corrupt_patient_dict = data["conditions"][cond]["patient_level"]
        # Explicit alignment check
        corrupt_patients = sorted(list(corrupt_patient_dict.keys()))
        assert corrupt_patients == patient_ids, f"Patient mismatch in {cond}!"

        corrupt_scores = np.array([corrupt_patient_dict[p]["mean_dice_foreground"] for p in patient_ids], dtype=np.float64)

        # Differences: corrupt - clean
        diffs = corrupt_scores - clean_scores
        rel_diffs = (diffs / clean_scores) * 100.0

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        mean_rel = float(np.mean(rel_diffs))
        cohort_rel = float((np.mean(corrupt_scores) - np.mean(clean_scores)) / np.mean(clean_scores) * 100.0)

        # Independent paired t-test (two-sided)
        t_stat, t_pval = stats.ttest_rel(corrupt_scores, clean_scores)

        # Independent Wilcoxon signed-rank test (two-sided, zero_method='wilcox')
        w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        w_stat = float(w_res.statistic)
        w_pval = float(w_res.pvalue)

        raw_t_pvals.append(t_pval)
        raw_w_pvals.append(w_pval)

        reproduced_stats[cond] = {
            "n_patients": len(patient_ids),
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "mean_rel_pct": mean_rel,
            "cohort_rel_pct": cohort_rel,
            "t_stat": float(t_stat),
            "t_pval_raw": float(t_pval),
            "w_stat": w_stat,
            "w_pval_raw": w_pval,
        }

    # Apply independent Holm step-down correction across m=9
    adj_t_pvals, audit_trail_t = independent_holm_stepdown(raw_t_pvals)
    adj_w_pvals, audit_trail_w = independent_holm_stepdown(raw_w_pvals)

    for idx, cond in enumerate(corruption_keys):
        reproduced_stats[cond]["t_pval_holm"] = adj_t_pvals[idx]
        reproduced_stats[cond]["w_pval_holm"] = adj_w_pvals[idx]
        reproduced_stats[cond]["significant_t"] = bool(adj_t_pvals[idx] < 0.05)
        reproduced_stats[cond]["significant_w"] = bool(adj_w_pvals[idx] < 0.05)

    # Compare against reported numbers
    print("\n--- COMPARISON TABLE: REPRODUCED VS REPORTED ---")
    print(f"{'Condition':28s} | {'Raw t-test':12s} | {'Holm t-test':12s} | {'Raw Wilcox':12s} | {'Holm Wilcox':12s} | {'Sig (alpha=0.05)':18s}")
    print("-" * 105)

    all_matched = True
    for idx, cond in enumerate(corruption_keys):
        rep = reproduced_stats[cond]
        rep_p = reported_paired.get(cond, {})

        t_raw_match = np.isclose(rep["t_pval_raw"], rep_p.get("paired_t_pvalue", -1), atol=1e-5)
        t_holm_match = np.isclose(rep["t_pval_holm"], rep_p.get("paired_t_pvalue_holm", -1), atol=1e-5)
        w_raw_match = np.isclose(rep["w_pval_raw"], rep_p.get("wilcoxon_pvalue", -1), atol=1e-5)
        w_holm_match = np.isclose(rep["w_pval_holm"], rep_p.get("wilcoxon_pvalue_holm", -1), atol=1e-5)

        if not (t_raw_match and t_holm_match and w_raw_match and w_holm_match):
            all_matched = False

        sig_str = "Significant" if rep["significant_w"] and rep["significant_t"] else (
            "Sig (Wilcoxon only)" if rep["significant_w"] else "Not Significant"
        )
        print(f"{cond:28s} | {rep['t_pval_raw']:12.4e} | {rep['t_pval_holm']:12.4e} | {rep['w_pval_raw']:12.4e} | {rep['w_pval_holm']:12.4e} | {sig_str:18s}")

    print(f"\nAll Reported Statistical P-Values Match Independent Recalculation: {all_matched}")

    # Inspect specific claims in the prompt:
    print("\n--- SPECIFIC USER PROMPT CLAIMS VERIFICATION ---")
    claims = [
        ("Severe noise: Wilcoxon Holm p = 0.0256", "gaussian_noise_severe", 0.025635),
        ("Mild blur: Wilcoxon Holm p = 0.0418", "gaussian_blur_mild", 0.041809),
        ("Moderate blur: Wilcoxon Holm p = 0.0010", "gaussian_blur_moderate", 0.00097656),
        ("Severe blur: Wilcoxon Holm p = 0.0005", "gaussian_blur_severe", 0.00054932),
        ("Severe contrast: Wilcoxon Holm p = 0.0013", "contrast_reduction_severe", 0.0012817),
    ]

    claims_verified = True
    for claim_desc, cond, target_val in claims:
        actual_val = reproduced_stats[cond]["w_pval_holm"]
        match = np.isclose(actual_val, target_val, atol=1e-4)
        if not match:
            claims_verified = False
        print(f"Claim: {claim_desc:45s} -> Recalculated = {actual_val:.6f} | Match: {match}")

    # Step-down monotonicity check
    sorted_holm_w = [item["holm_adjusted"] for item in audit_trail_w]
    is_monotonic_w = all(sorted_holm_w[i] <= sorted_holm_w[i+1] for i in range(len(sorted_holm_w)-1))
    print(f"\nWilcoxon Holm Adjusted Step-Down Strictly Monotonic: {is_monotonic_w}")

    audit_3_result = {
        "verdict": "VERIFIED" if (all_matched and claims_verified and is_monotonic_w) else "FAILED",
        "n_patients": len(patient_ids),
        "m_hypotheses": len(corruption_keys),
        "family_wise_error_rate_alpha": 0.05,
        "is_monotonic_w": is_monotonic_w,
        "reproduced_stats": reproduced_stats,
        "wilcoxon_audit_trail": audit_trail_w,
        "ttest_audit_trail": audit_trail_t,
        "claims_verification": [
            {"claim": desc, "condition": cond, "expected": target, "actual": reproduced_stats[cond]["w_pval_holm"], "verified": bool(np.isclose(reproduced_stats[cond]["w_pval_holm"], target, atol=1e-4))}
            for desc, cond, target in claims
        ]
    }

    out_file = PROJECT_ROOT / "scripts" / "audit" / "audit_3_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_3_result, f, indent=2)

    print(f"\nAudit 3 Verdict: {audit_3_result['verdict']}")
    return audit_3_result


if __name__ == "__main__":
    run_audit_3()
