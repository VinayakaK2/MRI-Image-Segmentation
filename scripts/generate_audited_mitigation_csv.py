"""Generate audited mitigation CSV with full Holm FWER corrections and explicit columns.

Separates raw from Holm-adjusted p-values across clearly defined families:
1. Post-processing alone across all conditions (m=10)
2. CLAHE alone across contrast reduction conditions (m=3)
3. Combined CLAHE + post-processing across contrast conditions (m=3)
4. Combined Median + post-processing across noise conditions (m=3)
"""

import csv
import json
from pathlib import Path
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def step_down_holm(pvals):
    m = len(pvals)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    running_max = 0.0
    adjusted_indexed = []
    for rank_idx, (orig_idx, p) in enumerate(indexed):
        mult = m - rank_idx
        raw_adj = p * mult
        running_max = max(running_max, raw_adj)
        capped = min(1.0, float(running_max))
        adjusted_indexed.append((orig_idx, capped))
    final_adj = [0.0] * m
    for orig_idx, capped in adjusted_indexed:
        final_adj[orig_idx] = capped
    return final_adj


def main():
    json_path = PROJECT_ROOT / "reports" / "mitigation_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unmit = data["unmitigated"]
    postproc = data["postproc_alone"]
    clahe = data["clahe_alone"]
    comb_c = data["combined_contrast"]
    comb_n = data["combined_noise"]

    p_ids = sorted(list(unmit["clean_baseline"]["per_patient"].keys()))

    CONDITIONS = [
        {"id": "clean_baseline", "family": "clean", "severity": "none"},
        {"id": "gaussian_noise_mild", "family": "gaussian_noise", "severity": "mild"},
        {"id": "gaussian_noise_moderate", "family": "gaussian_noise", "severity": "moderate"},
        {"id": "gaussian_noise_severe", "family": "gaussian_noise", "severity": "severe"},
        {"id": "gaussian_blur_mild", "family": "gaussian_blur", "severity": "mild"},
        {"id": "gaussian_blur_moderate", "family": "gaussian_blur", "severity": "moderate"},
        {"id": "gaussian_blur_severe", "family": "gaussian_blur", "severity": "severe"},
        {"id": "contrast_reduction_mild", "family": "contrast_reduction", "severity": "mild"},
        {"id": "contrast_reduction_moderate", "family": "contrast_reduction", "severity": "moderate"},
        {"id": "contrast_reduction_severe", "family": "contrast_reduction", "severity": "severe"},
    ]

    all_cond_ids = [c["id"] for c in CONDITIONS]
    contrast_ids = ["contrast_reduction_mild", "contrast_reduction_moderate", "contrast_reduction_severe"]
    noise_ids = ["gaussian_noise_mild", "gaussian_noise_moderate", "gaussian_noise_severe"]

    # Compute stats for postproc (m=10)
    post_diffs, post_t_raw, post_w_raw = [], [], []
    for cid in all_cond_ids:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        t_vals = [postproc[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(t_vals) - np.array(u_vals)
        post_diffs.append(float(np.mean(diff)))
        post_t_raw.append(float(stats.ttest_rel(t_vals, u_vals).pvalue))
        post_w_raw.append(float(stats.wilcoxon(t_vals, u_vals).pvalue))
    post_t_holm = step_down_holm(post_t_raw)
    post_w_holm = step_down_holm(post_w_raw)

    # Compute stats for clahe on contrast (m=3)
    clahe_diffs, clahe_t_raw, clahe_w_raw = [], [], []
    for cid in contrast_ids:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        t_vals = [clahe[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(t_vals) - np.array(u_vals)
        clahe_diffs.append(float(np.mean(diff)))
        clahe_t_raw.append(float(stats.ttest_rel(t_vals, u_vals).pvalue))
        clahe_w_raw.append(float(stats.wilcoxon(t_vals, u_vals).pvalue))
    clahe_t_holm = step_down_holm(clahe_t_raw)
    clahe_w_holm = step_down_holm(clahe_w_raw)

    # Compute stats for combined on contrast (m=3)
    combc_diffs, combc_t_raw, combc_w_raw = [], [], []
    for cid in contrast_ids:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        t_vals = [comb_c[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(t_vals) - np.array(u_vals)
        combc_diffs.append(float(np.mean(diff)))
        combc_t_raw.append(float(stats.ttest_rel(t_vals, u_vals).pvalue))
        combc_w_raw.append(float(stats.wilcoxon(t_vals, u_vals).pvalue))
    combc_t_holm = step_down_holm(combc_t_raw)
    combc_w_holm = step_down_holm(combc_w_raw)

    # Compute stats for combined on noise (m=3)
    combn_diffs, combn_t_raw, combn_w_raw = [], [], []
    for cid in noise_ids:
        u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        t_vals = [comb_n[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
        diff = np.array(t_vals) - np.array(u_vals)
        combn_diffs.append(float(np.mean(diff)))
        combn_t_raw.append(float(stats.ttest_rel(t_vals, u_vals).pvalue))
        combn_w_raw.append(float(stats.wilcoxon(t_vals, u_vals).pvalue))
    combn_t_holm = step_down_holm(combn_t_raw)
    combn_w_holm = step_down_holm(combn_w_raw)

    rows = []

    # 1. Unmitigated rows
    for c in CONDITIONS:
        cid = c["id"]
        res = unmit[cid]
        psum = res["patient_summary"]
        rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Unmitigated Baseline",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": 0.0,
            "paired_t_pvalue_raw": "N/A",
            "paired_t_pvalue_holm": "N/A",
            "wilcoxon_pvalue_raw": "N/A",
            "wilcoxon_pvalue_holm": "N/A",
            "significance_fwer": "Reference Baseline",
        })

    # 2. Post-processing alone rows (m=10 family)
    for i, c in enumerate(CONDITIONS):
        cid = c["id"]
        res = postproc[cid]
        psum = res["patient_summary"]
        t_r, t_h = post_t_raw[i], post_t_holm[i]
        w_r, w_h = post_w_raw[i], post_w_holm[i]
        diff = post_diffs[i]

        sig = "Significant" if (w_h < 0.05 or t_h < 0.05) else "Not Significant"
        if sig == "Not Significant" and (w_r < 0.05 or t_r < 0.05):
            sig = "Nominal Only (Unadjusted)"

        rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Post-Processing Alone (LCC+HoleFill)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue_raw": f"{t_r:.4e}",
            "paired_t_pvalue_holm": f"{t_h:.4e}",
            "wilcoxon_pvalue_raw": f"{w_r:.4e}",
            "wilcoxon_pvalue_holm": f"{w_h:.4e}",
            "significance_fwer": sig,
        })

    # 3. CLAHE alone rows
    for c in CONDITIONS:
        cid = c["id"]
        res = clahe[cid]
        psum = res["patient_summary"]
        if cid in contrast_ids:
            idx = contrast_ids.index(cid)
            t_r, t_h = clahe_t_raw[idx], clahe_t_holm[idx]
            w_r, w_h = clahe_w_raw[idx], clahe_w_holm[idx]
            diff = clahe_diffs[idx]
            sig = "Significant" if (w_h < 0.05 or t_h < 0.05) else "Not Significant"
            if sig == "Not Significant" and (w_r < 0.05 or t_r < 0.05):
                sig = "Nominal Only (Unadjusted)"
            t_r_str, t_h_str = f"{t_r:.4e}", f"{t_h:.4e}"
            w_r_str, w_h_str = f"{w_r:.4e}", f"{w_h:.4e}"
        else:
            u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
            t_vals = [res["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
            diff = float(np.mean(np.array(t_vals) - np.array(u_vals)))
            t_r = float(stats.ttest_rel(t_vals, u_vals).pvalue)
            w_r = float(stats.wilcoxon(t_vals, u_vals).pvalue)
            t_r_str, t_h_str = f"{t_r:.4e}", "N/A"
            w_r_str, w_h_str = f"{w_r:.4e}", "N/A"
            sig = "Nominal Only (Unadjusted)" if (t_r < 0.05 or w_r < 0.05) else "Not Significant"

        rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "CLAHE Alone (clip=0.01)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue_raw": t_r_str,
            "paired_t_pvalue_holm": t_h_str,
            "wilcoxon_pvalue_raw": w_r_str,
            "wilcoxon_pvalue_holm": w_h_str,
            "significance_fwer": sig,
        })

    # 4. Combined CLAHE + Post-processing on Contrast
    for cid in ["clean_baseline"] + contrast_ids:
        c = next(item for item in CONDITIONS if item["id"] == cid)
        res = comb_c[cid]
        psum = res["patient_summary"]
        if cid in contrast_ids:
            idx = contrast_ids.index(cid)
            t_r, t_h = combc_t_raw[idx], combc_t_holm[idx]
            w_r, w_h = combc_w_raw[idx], combc_w_holm[idx]
            diff = combc_diffs[idx]
            sig = "Significant" if (w_h < 0.05 or t_h < 0.05) else "Not Significant"
            if sig == "Not Significant" and (w_r < 0.05 or t_r < 0.05):
                sig = "Nominal Only (Unadjusted)"
            t_r_str, t_h_str = f"{t_r:.4e}", f"{t_h:.4e}"
            w_r_str, w_h_str = f"{w_r:.4e}", f"{w_h:.4e}"
        else:
            u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
            t_vals = [res["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
            diff = float(np.mean(np.array(t_vals) - np.array(u_vals)))
            t_r = float(stats.ttest_rel(t_vals, u_vals).pvalue)
            w_r = float(stats.wilcoxon(t_vals, u_vals).pvalue)
            t_r_str, t_h_str = f"{t_r:.4e}", "N/A"
            w_r_str, w_h_str = f"{w_r:.4e}", "N/A"
            sig = "Not Significant"

        rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Combined (CLAHE + Post-Processing)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue_raw": t_r_str,
            "paired_t_pvalue_holm": t_h_str,
            "wilcoxon_pvalue_raw": w_r_str,
            "wilcoxon_pvalue_holm": w_h_str,
            "significance_fwer": sig,
        })

    # 5. Combined Median + Post-processing on Noise
    for cid in ["clean_baseline"] + noise_ids:
        c = next(item for item in CONDITIONS if item["id"] == cid)
        res = comb_n[cid]
        psum = res["patient_summary"]
        if cid in noise_ids:
            idx = noise_ids.index(cid)
            t_r, t_h = combn_t_raw[idx], combn_t_holm[idx]
            w_r, w_h = combn_w_raw[idx], combn_w_holm[idx]
            diff = combn_diffs[idx]
            sig = "Significant" if (w_h < 0.05 or t_h < 0.05) else "Not Significant"
            if sig == "Not Significant" and (w_r < 0.05 or t_r < 0.05):
                sig = "Nominal Only (Unadjusted)"
            t_r_str, t_h_str = f"{t_r:.4e}", f"{t_h:.4e}"
            w_r_str, w_h_str = f"{w_r:.4e}", f"{w_h:.4e}"
        else:
            u_vals = [unmit[cid]["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
            t_vals = [res["per_patient"][p]["mean_dice_foreground"] for p in p_ids]
            diff = float(np.mean(np.array(t_vals) - np.array(u_vals)))
            t_r = float(stats.ttest_rel(t_vals, u_vals).pvalue)
            w_r = float(stats.wilcoxon(t_vals, u_vals).pvalue)
            t_r_str, t_h_str = f"{t_r:.4e}", "N/A"
            w_r_str, w_h_str = f"{w_r:.4e}", "N/A"
            sig = "Not Significant"

        rows.append({
            "condition_id": cid,
            "family": c["family"],
            "severity": c["severity"],
            "intervention_arm": "Combined (Median + Post-Processing)",
            "slice_dice_fg": round(res["slice_level"]["mean_dice_foreground"], 4),
            "patient_dice_fg_mean": round(psum["mean_dice_foreground"]["mean"], 4),
            "patient_dice_fg_std": round(psum["mean_dice_foreground"]["std"], 4),
            "dice_rv": round(psum.get("dice_rv", {}).get("mean", 0.0), 4),
            "dice_myo": round(psum.get("dice_myo", {}).get("mean", 0.0), 4),
            "dice_lv": round(psum.get("dice_lv", {}).get("mean", 0.0), 4),
            "delta_dice_mean": round(diff, 4),
            "paired_t_pvalue_raw": t_r_str,
            "paired_t_pvalue_holm": t_h_str,
            "wilcoxon_pvalue_raw": w_r_str,
            "wilcoxon_pvalue_holm": w_h_str,
            "significance_fwer": sig,
        })

    out_csv = PROJECT_ROOT / "reports" / "mitigation_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Successfully generated audited CSV with {len(rows)} rows: {out_csv}")


if __name__ == "__main__":
    main()
