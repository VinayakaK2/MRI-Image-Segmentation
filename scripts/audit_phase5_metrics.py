"""Independent Phase 5 verification and statistical audit script.

Audits per-patient arrays, verifies affine contrast invariance, recomputes
paired differences, and applies Holm-Bonferroni correction across all four clearly
defined comparison families, separating raw from adjusted p-values and verifying
against published CSV artifacts.
"""

import csv
import json
from pathlib import Path
import sys
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.patient_metrics import adjust_pvalues_holm, classify_significance
from src.utils.provenance import (
    CACHE_SCHEMA_VERSION,
    build_cache_provenance,
    validate_cache_provenance,
)


def step_down_holm(pvals):
    """Applies Holm-Bonferroni step-down procedure with monotonicity enforcement."""
    m = len(pvals)
    if m == 0:
        return [], []
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


CONDITION_FAMILY_MAP = {
    "clean_baseline": "clean",
    "gaussian_noise_mild": "gaussian_noise",
    "gaussian_noise_moderate": "gaussian_noise",
    "gaussian_noise_severe": "gaussian_noise",
    "gaussian_blur_mild": "gaussian_blur",
    "gaussian_blur_moderate": "gaussian_blur",
    "gaussian_blur_severe": "gaussian_blur",
    "contrast_reduction_mild": "contrast_reduction",
    "contrast_reduction_moderate": "contrast_reduction",
    "contrast_reduction_severe": "contrast_reduction",
}


def main():
    json_path = PROJECT_ROOT / "reports" / "mitigation_results.json"
    csv_path = PROJECT_ROOT / "reports" / "mitigation_results.csv"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(csv_path, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    # Verify patient keys consistency across all arms and conditions
    p_ids = sorted(list(data["unmitigated"]["clean_baseline"]["per_patient"].keys()))
    for arm_name in ["unmitigated", "postproc_alone", "clahe_alone", "combined_contrast", "combined_noise"]:
        for cid in data[arm_name]:
            curr_pts = sorted(list(data[arm_name][cid]["per_patient"].keys()))
            if curr_pts != p_ids:
                print(f"FATAL: Patient ID mismatch in {arm_name}/{cid}: expected {p_ids}, found {curr_pts}")
                sys.exit(1)

    print("=" * 80)
    print(" AUDIT 0: CRYPTOGRAPHIC CACHE PROVENANCE VERIFICATION")
    print("=" * 80)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model_frozen.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    cfg_path = PROJECT_ROOT / "configs" / "baseline_unet.yaml"
    man_path = PROJECT_ROOT / "data" / "processed" / "acdc_slice_manifest.json"

    expected_prov = build_cache_provenance(ckpt_path, cfg_path, man_path, CACHE_SCHEMA_VERSION)
    is_prov_valid, prov_reason = validate_cache_provenance(data, expected_prov)
    if is_prov_valid:
        print(f"PASS: Provenance metadata verified (Schema {CACHE_SCHEMA_VERSION}).")
        print(f"  - Checkpoint SHA-256: {expected_prov['checkpoint_sha256']}")
        print(f"  - Config SHA-256:     {expected_prov['config_sha256']}")
        print(f"  - Manifest SHA-256:   {expected_prov['manifest_sha256']}")
    else:
        print(f"NOTE: Cache provenance validation: {prov_reason}")

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
    print(" AUDIT 2: RECOMPUTATION OF PAIRED DIFFERENCES & P-VALUES (POST-PROCESSING ALONE, m=10)")
    print("=" * 80)

    postproc = data["postproc_alone"]
    unmit = data["unmitigated"]
    all_cond_keys = list(postproc.keys())

    # Family 1: Post-Processing Alone across all 10 conditions (m=10)
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
        sig = classify_significance(
            paired_t_pvalue_raw=t_raw,
            paired_t_pvalue_holm=t_adj,
            wilcoxon_pvalue_raw=w_raw,
            wilcoxon_pvalue_holm=w_adj,
            alpha=0.05,
            policy="primary_t_test",
        )
        print(f"{cid:28s} | {t_raw:12.4e} | {t_adj:12.4e} | {w_raw:12.4e} | {w_adj:12.4e} | {sig:22s}")

    print("\n" + "=" * 80)
    print(" AUDIT 3: CONTRAST RESTORATION PAIRED COMPARISONS (CLAHE & COMBINED, m=3 EACH)")
    print("=" * 80)

    clahe_keys = ["contrast_reduction_severe", "contrast_reduction_moderate", "contrast_reduction_mild"]
    clahe_t_p = []
    clahe_w_p = []
    comb_t_p = []
    comb_w_p = []

    print("\nFamily 2: CLAHE Alone on Contrast (m=3 family):")
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
        sig = classify_significance(
            paired_t_pvalue_raw=t_raw,
            paired_t_pvalue_holm=t_adj,
            wilcoxon_pvalue_raw=w_raw,
            wilcoxon_pvalue_holm=w_adj,
            alpha=0.05,
            policy="primary_t_test",
        )
        print(f"{cid:28s} | {t_raw:12.4e} | {t_adj:12.4e} | {w_raw:12.4e} | {w_adj:12.4e} | {sig:22s}")

    print("\nFamily 3: Combined (CLAHE + Post-Processing) on Contrast (m=3 family):")
    print(f"{'Condition ID':28s} | {'Unmit Dice':10s} | {'Comb Dice':10s} | {'Delta':10s} | {'t-test p_raw':12s} | {'Wilcoxon p_raw':12s}")
    print("-" * 92)
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
        sig = classify_significance(
            paired_t_pvalue_raw=t_raw,
            paired_t_pvalue_holm=t_adj,
            wilcoxon_pvalue_raw=w_raw,
            wilcoxon_pvalue_holm=w_adj,
            alpha=0.05,
            policy="primary_t_test",
        )
        print(f"{cid:28s} | {t_raw:12.4e} | {t_adj:12.4e} | {w_raw:12.4e} | {w_adj:12.4e} | {sig:22s}")

    print("\n" + "=" * 80)
    print(" AUDIT 4: COMBINED (MEDIAN + POST-PROCESSING) ON NOISE (m=3 family)")
    print("=" * 80)

    noise_keys = ["gaussian_noise_mild", "gaussian_noise_moderate", "gaussian_noise_severe"]
    comb_n = data["combined_noise"]
    combn_t_p = []
    combn_w_p = []
    combn_diffs = []

    print(f"{'Condition ID':28s} | {'Unmit Dice':10s} | {'Comb Dice':10s} | {'Delta':10s} | {'t-test p_raw':12s} | {'Wilcoxon p_raw':12s}")
    print("-" * 92)
    for cid in noise_keys:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        cn_vals = [comb_n[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(cn_vals) - np.array(u_vals)
        mean_diff = float(np.mean(diff))
        combn_diffs.append(mean_diff)
        t_res = stats.ttest_rel(cn_vals, u_vals)
        w_res = stats.wilcoxon(cn_vals, u_vals)
        t_pval = float(t_res.pvalue)
        w_pval = float(w_res.pvalue)
        combn_t_p.append(t_pval)
        combn_w_p.append(w_pval)
        u_mean = unmit[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        cn_mean = comb_n[cid]["patient_summary"]["mean_dice_foreground"]["mean"]
        print(f"{cid:28s} | {u_mean:10.4f} | {cn_mean:10.4f} | {mean_diff:+10.4f} | {t_pval:12.4e} | {w_pval:12.4e}")

    t_adj_combn, _ = step_down_holm(combn_t_p)
    w_adj_combn, _ = step_down_holm(combn_w_p)

    print("\nHolm-Adjusted P-values for Combined Noise Family (m=3):")
    print(f"{'Condition ID':28s} | {'t-test p_raw':12s} | {'t-test p_Holm':12s} | {'Wilcoxon p_raw':12s} | {'Wilcoxon p_Holm':12s} | {'Decision (alpha=0.05)':22s}")
    print("-" * 115)
    for i, cid in enumerate(noise_keys):
        t_raw = combn_t_p[i]
        t_adj = t_adj_combn[i]
        w_raw = combn_w_p[i]
        w_adj = w_adj_combn[i]
        sig = classify_significance(
            paired_t_pvalue_raw=t_raw,
            paired_t_pvalue_holm=t_adj,
            wilcoxon_pvalue_raw=w_raw,
            wilcoxon_pvalue_holm=w_adj,
            alpha=0.05,
            policy="primary_t_test",
        )
        print(f"{cid:28s} | {t_raw:12.4e} | {t_adj:12.4e} | {w_raw:12.4e} | {w_adj:12.4e} | {sig:22s}")

    print("\n" + "=" * 80)
    print(" AUDIT 5: CROSS-VERIFICATION AGAINST PUBLISHED CSV (reports/mitigation_results.csv)")
    print("=" * 80)

    # 1. Verify CSV row count and pair uniqueness
    csv_pairs = [(r["condition_id"], r["intervention_arm"]) for r in csv_rows]
    discrepancies = []
    if len(csv_pairs) != len(set(csv_pairs)):
        discrepancies.append(
            f"Duplicate (condition_id, intervention_arm) rows found in CSV: "
            f"{len(csv_pairs)} total rows vs {len(set(csv_pairs))} unique pairs"
        )
    if len(csv_rows) != 38:
        discrepancies.append(f"Unexpected total row count in CSV: expected 38, found {len(csv_rows)}")

    # 2. Cross-verify independent step_down_holm with src.evaluation.patient_metrics.adjust_pvalues_holm
    for fam_label, raw_pvals, calc_adj in [
        ("Family 1 t-test", post_t_p, t_adj_post),
        ("Family 1 Wilcoxon", post_w_p, w_adj_post),
        ("Family 2 t-test", clahe_t_p, t_adj_clahe),
        ("Family 2 Wilcoxon", clahe_w_p, w_adj_clahe),
        ("Family 3 t-test", comb_t_p, t_adj_comb),
        ("Family 3 Wilcoxon", comb_w_p, w_adj_comb),
        ("Family 4 t-test", combn_t_p, t_adj_combn),
        ("Family 4 Wilcoxon", combn_w_p, w_adj_combn),
    ]:
        lib_adj = adjust_pvalues_holm(raw_pvals)
        max_d = max(abs(a - b) for a, b in zip(calc_adj, lib_adj)) if raw_pvals else 0.0
        if max_d > 1e-6:
            discrepancies.append(f"Holm implementation mismatch in {fam_label}: max diff {max_d:.2e}")

    # 3. Cross-validate all 19 hypothesis comparisons across the 4 pre-specified families
    families_to_check = [
        ("Family 1 (Post-Proc, m=10)", all_cond_keys, "Post-Processing Alone (LCC+HoleFill)", post_diffs, post_t_p, t_adj_post, post_w_p, w_adj_post),
        ("Family 2 (CLAHE, m=3)", clahe_keys, "CLAHE Alone (clip=0.01)", [float(np.mean(np.array([clahe[c]['per_patient'][p]['mean_dice_foreground'] for p in p_ids]) - np.array([unmit[c]['per_patient'][p]['mean_dice_foreground'] for p in p_ids]))) for c in clahe_keys], clahe_t_p, t_adj_clahe, clahe_w_p, w_adj_clahe),
        ("Family 3 (Combined Contrast, m=3)", clahe_keys, "Combined (CLAHE + Post-Processing)", [float(np.mean(np.array([comb[c]['per_patient'][p]['mean_dice_foreground'] for p in p_ids]) - np.array([unmit[c]['per_patient'][p]['mean_dice_foreground'] for p in p_ids]))) for c in clahe_keys], comb_t_p, t_adj_comb, comb_w_p, w_adj_comb),
        ("Family 4 (Combined Noise, m=3)", noise_keys, "Combined (Median + Post-Processing)", combn_diffs, combn_t_p, t_adj_combn, combn_w_p, w_adj_combn),
    ]

    for fname, cond_list, arm_str, diffs_arr, t_raw_arr, t_holm_arr, w_raw_arr, w_holm_arr in families_to_check:
        for idx, cid in enumerate(cond_list):
            matches = [r for r in csv_rows if r["condition_id"] == cid and r["intervention_arm"] == arm_str]
            if not matches:
                discrepancies.append(f"Missing row in CSV: {cid}, {arm_str}")
                continue
            if len(matches) > 1:
                discrepancies.append(f"Duplicate row in CSV: {cid}, {arm_str} (found {len(matches)} matches)")
                continue
            row = matches[0]

            # Check family membership in CSV
            expected_fam = CONDITION_FAMILY_MAP.get(cid)
            if row.get("family") != expected_fam:
                discrepancies.append(f"Incorrect family membership for {cid}: expected '{expected_fam}', got '{row.get('family')}'")

            # Check delta
            csv_d = float(row["delta_dice_mean"])
            calc_d = round(diffs_arr[idx], 4)
            if abs(csv_d - calc_d) > 1e-4:
                discrepancies.append(f"Delta mismatch {fname} {cid}: CSV {csv_d} vs Calc {calc_d}")

            # Check paired t-test p-values
            csv_tp_raw = float(row["paired_t_pvalue_raw"])
            calc_tp_raw = t_raw_arr[idx]
            if abs(csv_tp_raw - calc_tp_raw) / max(calc_tp_raw, 1e-15) > 1e-3:
                discrepancies.append(f"t_raw mismatch {fname} {cid}: CSV {csv_tp_raw} vs Calc {calc_tp_raw}")
            csv_tp_holm = float(row["paired_t_pvalue_holm"])
            calc_tp_holm = t_holm_arr[idx]
            if abs(csv_tp_holm - calc_tp_holm) / max(calc_tp_holm, 1e-15) > 1e-3:
                discrepancies.append(f"t_holm mismatch {fname} {cid}: CSV {csv_tp_holm} vs Calc {calc_tp_holm}")

            # Check Wilcoxon p-values
            csv_wp_raw = float(row["wilcoxon_pvalue_raw"])
            calc_wp_raw = w_raw_arr[idx]
            if abs(csv_wp_raw - calc_wp_raw) / max(calc_wp_raw, 1e-15) > 1e-3:
                discrepancies.append(f"w_raw mismatch {fname} {cid}: CSV {csv_wp_raw} vs Calc {calc_wp_raw}")
            csv_wp_holm = float(row["wilcoxon_pvalue_holm"])
            calc_wp_holm = w_holm_arr[idx]
            if abs(csv_wp_holm - calc_wp_holm) / max(calc_wp_holm, 1e-15) > 1e-3:
                discrepancies.append(f"w_holm mismatch {fname} {cid}: CSV {csv_wp_holm} vs Calc {calc_wp_holm}")

            # Check FWER significance decision under pre-specified primary paired t-test policy
            calc_sig = classify_significance(
                paired_t_pvalue_raw=calc_tp_raw,
                paired_t_pvalue_holm=calc_tp_holm,
                wilcoxon_pvalue_raw=calc_wp_raw,
                wilcoxon_pvalue_holm=calc_wp_holm,
                alpha=0.05,
                policy="primary_t_test",
            )
            csv_sig = row.get("significance_fwer", "")
            if csv_sig.lower() != calc_sig.lower():
                discrepancies.append(f"Significance label mismatch {fname} {cid}: CSV '{csv_sig}' vs Calc '{calc_sig}'")

    if not is_prov_valid:
        discrepancies.append(f"Mitigation results cache provenance invalid: {prov_reason}")

    if discrepancies:
        print(f"\nDISCREPANCIES DETECTED ({len(discrepancies)}):")
        for d in discrepancies:
            print(f"  - {d}")
        sys.exit(1)
    else:
        print("ALL 4 FAMILIES (19 HYPOTHESIS COMPARISONS) PERFECTLY MATCH reports/mitigation_results.csv!")


if __name__ == "__main__":
    main()
