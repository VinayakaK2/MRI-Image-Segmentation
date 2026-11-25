"""Service for reading and filtering Phase 4 robustness and Phase 5 mitigation JSON/CSV artifacts."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.config import (
    CORRUPTION_TYPE_ALIASES,
    PHASE5_FAMILY_DEFINITIONS,
    ResearchPaths,
    build_research_paths,
)
from api.schemas.experiment import (
    ExperimentsResponse,
    Phase4ConditionRecord,
    Phase4Payload,
    Phase4StatisticalComparison,
    Phase5ArmConditionMetrics,
    Phase5FamilyGroup,
    Phase5HypothesisRecord,
    Phase5NegativeControlRecord,
    Phase5Payload,
)
from api.utils.validation import ApiError, validate_patient_id_format
from src.evaluation.patient_metrics import classify_significance


PHASE5_FAMILY_ALIASES: Dict[str, str] = {
    "family_1": "family_1_postproc_alone",
    "family_1_postproc_alone": "family_1_postproc_alone",
    "postproc_alone": "family_1_postproc_alone",
    "family_2": "family_2_clahe_alone",
    "family_2_clahe_alone": "family_2_clahe_alone",
    "clahe_alone": "family_2_clahe_alone",
    "family_3": "family_3_clahe_plus_postproc",
    "family_3_clahe_plus_postproc": "family_3_clahe_plus_postproc",
    "combined_contrast": "family_3_clahe_plus_postproc",
    "family_4": "family_4_median_plus_postproc",
    "family_4_median_plus_postproc": "family_4_median_plus_postproc",
    "combined_noise": "family_4_median_plus_postproc",
}


class ExperimentService:
    """Reads Phase 4 and Phase 5 canonical experiment artifacts without recomputing statistics."""

    def __init__(self, paths: Optional[ResearchPaths] = None) -> None:
        """Initializes artifact paths for Phase 4 and Phase 5 queries."""
        self.paths = paths or build_research_paths()

    def get_experiments(
        self,
        phase: Optional[int] = None,
        family: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> ExperimentsResponse:
        """Returns Phase 4 and/or Phase 5 experiment results with optional phase, family, and patient filters."""
        if phase is not None and phase not in (4, 5):
            raise ApiError(
                status_code=422,
                code="INVALID_PHASE_FILTER",
                message="Query parameter 'phase' must be 4 or 5.",
            )

        clean_patient: Optional[str] = None
        if patient_id is not None:
            clean_patient = validate_patient_id_format(patient_id)
            self._validate_test_patient_exists(clean_patient)

        phase4_payload: Optional[Phase4Payload] = None
        phase5_payload: Optional[Phase5Payload] = None

        if phase in (None, 4):
            phase4_payload = self._load_phase4(family=family if phase == 4 else None, patient_id=clean_patient)
            if phase == 4 and not phase4_payload.available:
                raise ApiError(
                    status_code=503,
                    code="ARTIFACT_UNAVAILABLE",
                    message=phase4_payload.error or "Phase 4 artifacts are unavailable.",
                )

        if phase in (None, 5):
            phase5_payload = self._load_phase5(family=family, patient_id=clean_patient)
            if phase == 5 and not phase5_payload.available:
                raise ApiError(
                    status_code=503,
                    code="ARTIFACT_UNAVAILABLE",
                    message=phase5_payload.error or "Phase 5 artifacts are unavailable.",
                )

        if phase is None and family is not None:
            # Validate family against Phase 4 or Phase 5 if phase was omitted
            if family.strip().lower() not in PHASE5_FAMILY_ALIASES and family.strip().lower() not in CORRUPTION_TYPE_ALIASES:
                raise ApiError(
                    status_code=422,
                    code="INVALID_FAMILY_FILTER",
                    message=f"Unknown experiment family '{family}'.",
                    details={
                        "phase4_families": ["clean", "gaussian_noise", "gaussian_blur", "contrast_reduction"],
                        "phase5_families": list(PHASE5_FAMILY_DEFINITIONS.keys()),
                    },
                )
            if family.strip().lower() in CORRUPTION_TYPE_ALIASES and phase4_payload is not None:
                phase4_payload = self._load_phase4(family=family, patient_id=clean_patient)

        return ExperimentsResponse(
            phase_filter=phase,
            family_filter=family,
            patient_id_filter=clean_patient,
            phase4=phase4_payload,
            phase5=phase5_payload,
        )

    def _validate_test_patient_exists(self, patient_id: str) -> None:
        """Verifies a patient filter belongs to the 15-patient held-out test cohort."""
        if not self.paths.slice_manifest_path.is_file():
            raise ApiError(status_code=503, code="MANIFEST_NOT_FOUND", message="Slice manifest is unavailable.")
        with open(self.paths.slice_manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        test_patients = {
            str(s["patient_id"])
            for s in manifest.get("slices", [])
            if str(s.get("split", "")).lower() == "test"
        }
        if patient_id not in test_patients:
            raise ApiError(
                status_code=404,
                code="PATIENT_NOT_FOUND",
                message=f"Test patient '{patient_id}' was not found in the held-out test cohort.",
            )

    def _load_phase4(self, family: Optional[str], patient_id: Optional[str]) -> Phase4Payload:
        """Loads Phase 4 robustness results JSON while keeping slice-level and patient-level metrics separate."""
        json_path = self.paths.phase4_results_json
        if not json_path.is_file():
            return Phase4Payload(available=False, error="Phase 4 robustness_results.json is missing.")

        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as exc:
            return Phase4Payload(available=False, error=f"Malformed Phase 4 JSON: {type(exc).__name__}")

        if not isinstance(raw, dict) or "conditions" not in raw:
            return Phase4Payload(available=False, error="Phase 4 JSON is missing 'conditions'.")

        family_filter: Optional[str] = None
        if family is not None:
            family_filter = CORRUPTION_TYPE_ALIASES.get(family.strip().lower())
            if family_filter is None:
                raise ApiError(
                    status_code=422,
                    code="INVALID_FAMILY_FILTER",
                    message=f"Unknown Phase 4 corruption family '{family}'.",
                    details={"allowed_phase4_families": ["clean", "gaussian_noise", "gaussian_blur", "contrast_reduction"]},
                )

        meta = raw.get("metadata", {})
        paired_map = raw.get("paired_comparisons", {})
        conditions_out: List[Phase4ConditionRecord] = []

        for cid, cdata in raw.get("conditions", {}).items():
            cinfo = cdata.get("condition_info", {})
            cfam = str(cinfo.get("family", ""))
            if family_filter is not None and cfam != family_filter:
                continue

            per_patient = cdata.get("patient_level", {})
            if patient_id is not None:
                per_patient = {k: v for k, v in per_patient.items() if k == patient_id}

            paired_obj: Optional[Phase4StatisticalComparison] = None
            if cid in paired_map:
                p_stats = paired_map[cid]
                t_raw = float(p_stats["paired_t_pvalue"])
                t_holm = float(p_stats["paired_t_pvalue_holm"])
                w_raw = float(p_stats["wilcoxon_pvalue"])
                w_holm = float(p_stats["wilcoxon_pvalue_holm"])
                n_pts = int(p_stats.get("n_patients", 15))
                paired_obj = Phase4StatisticalComparison(
                    condition_id=cid,
                    metric=str(p_stats.get("metric", "mean_dice_foreground")),
                    n_patients=n_pts,
                    degrees_of_freedom=max(n_pts - 1, 0),
                    mean_clean=float(p_stats["mean_clean"]),
                    mean_corrupt=float(p_stats["mean_corrupt"]),
                    mean_difference=float(p_stats["mean_absolute_change"]),
                    std_difference=float(p_stats["std_absolute_change"]) if "std_absolute_change" in p_stats else None,
                    mean_relative_change_pct_patient_wise=p_stats.get("mean_relative_change_pct_patient_wise"),
                    relative_change_pct_of_means=p_stats.get("relative_change_pct_of_means"),
                    paired_t_statistic=float(p_stats["paired_t_statistic"]),
                    paired_t_pvalue_raw=t_raw,
                    paired_t_pvalue_holm=t_holm,
                    wilcoxon_statistic=float(p_stats["wilcoxon_statistic"]),
                    wilcoxon_pvalue_raw=w_raw,
                    wilcoxon_pvalue_holm=w_holm,
                    correction_method="Holm-Bonferroni step-down",
                    correction_family_scope="Across all 9 non-clean Phase 4 degradation conditions (m=9)",
                    significance_decision=classify_significance(t_raw, t_holm, w_raw, w_holm, alpha=0.05),
                )

            conditions_out.append(
                Phase4ConditionRecord(
                    condition_id=cid,
                    family=cfam,
                    severity=str(cinfo.get("severity", "")),
                    param_value=float(cinfo.get("param_value", 0.0)),
                    units=str(cinfo.get("units", "none")),
                    display_name=str(cinfo.get("display_name", cid)),
                    slice_level={k: float(v) for k, v in cdata.get("slice_level", {}).items()},
                    patient_level_summary=cdata.get("patient_summary", {}),
                    per_patient_metrics=per_patient,
                    statistical_comparison_vs_clean=paired_obj,
                )
            )

        return Phase4Payload(
            available=True,
            experiment_id=meta.get("experiment_id"),
            checkpoint_sha256=meta.get("checkpoint_sha256"),
            total_test_slices=meta.get("total_test_slices"),
            n_patients=meta.get("n_patients"),
            conditions=conditions_out,
        )

    def _load_phase5(self, family: Optional[str], patient_id: Optional[str]) -> Phase5Payload:
        """Loads Phase 5 mitigation JSON and CSV artifacts preserving the 4 Holm family boundaries."""
        required_files = [
            self.paths.phase5_results_json,
            self.paths.phase5_primary_hypotheses_csv,
            self.paths.phase5_statistical_tests_csv,
            self.paths.phase5_repro_manifest_json,
        ]
        for fpath in required_files:
            if not fpath.is_file():
                return Phase5Payload(available=False, error=f"Missing required Phase 5 artifact: {fpath.name}")

        try:
            with open(self.paths.phase5_results_json, "r", encoding="utf-8") as handle:
                p5_json = json.load(handle)
            with open(self.paths.phase5_repro_manifest_json, "r", encoding="utf-8") as handle:
                repro = json.load(handle)
            primary_rows = self._read_csv_rows(self.paths.phase5_primary_hypotheses_csv)
            stat_rows = self._read_csv_rows(self.paths.phase5_statistical_tests_csv)
            neg_rows = (
                self._read_csv_rows(self.paths.phase5_negative_controls_csv)
                if self.paths.phase5_negative_controls_csv.is_file()
                else []
            )
            clahe_audit = self._read_json_optional(self.paths.phase5_clahe_stage_json)
        except Exception as exc:
            return Phase5Payload(available=False, error=f"Malformed Phase 5 artifact: {type(exc).__name__}")

        target_family_id: Optional[str] = None
        if family is not None:
            fam_key = family.strip().lower()
            if fam_key in PHASE5_FAMILY_ALIASES:
                target_family_id = PHASE5_FAMILY_ALIASES[fam_key]
            elif fam_key not in CORRUPTION_TYPE_ALIASES:
                raise ApiError(
                    status_code=422,
                    code="INVALID_FAMILY_FILTER",
                    message=f"Unknown Phase 5 hypothesis family '{family}'.",
                    details={"allowed_phase5_families": list(PHASE5_FAMILY_DEFINITIONS.keys())},
                )

        # Build lookup from statistical_tests.csv keyed by (family_name, condition_id)
        stat_lookup: Dict[tuple, Dict[str, str]] = {
            (r["family_name"], r["condition_id"]): r for r in stat_rows
        }

        families_out: List[Phase5FamilyGroup] = []
        total_hypotheses = 0

        for fam_id, fam_def in PHASE5_FAMILY_DEFINITIONS.items():
            if target_family_id is not None and fam_id != target_family_id:
                continue

            csv_fam_name = str(fam_def["csv_family_name"])
            family_prefix = f"Family {fam_id.split('_')[1]}:"
            matching_rows = [
                r for r in primary_rows
                if r.get("hypothesis_family", "").startswith(family_prefix)
            ]

            hypotheses: List[Phase5HypothesisRecord] = []
            for row in matching_rows:
                st = stat_lookup.get((csv_fam_name, row["condition_id"]))
                if st is None:
                    return Phase5Payload(
                        available=False,
                        error=f"Inconsistent Phase 5 statistical_tests.csv entry for {csv_fam_name} / {row['condition_id']}",
                    )
                hypotheses.append(
                    Phase5HypothesisRecord(
                        family_id=fam_id,
                        family_name=csv_fam_name,
                        condition_id=row["condition_id"],
                        condition_family=row["family"],
                        severity=row["severity"],
                        intervention_arm=row["intervention_arm"],
                        n_patients=int(st["n_patients"]),
                        degrees_of_freedom=int(st["degrees_of_freedom"]),
                        slice_dice_fg=float(row["slice_dice_fg"]),
                        patient_dice_fg_mean=float(row["patient_dice_fg_mean"]),
                        patient_dice_fg_std=float(row["patient_dice_fg_std"]),
                        dice_rv_patient_mean=float(row["dice_rv"]),
                        dice_myo_patient_mean=float(row["dice_myo"]),
                        dice_lv_patient_mean=float(row["dice_lv"]),
                        mean_baseline_dice=float(st["mean_baseline_dice"]),
                        mean_mitigated_dice=float(st["mean_mitigated_dice"]),
                        mean_difference=float(st["mean_difference"]),
                        std_difference=float(st["std_difference"]),
                        se_difference=float(st["se_difference"]),
                        ci_95_lower=float(st["ci_95_lower"]),
                        ci_95_upper=float(st["ci_95_upper"]),
                        effect_size_cohens_d=float(st["cohens_d"]),
                        paired_t_statistic=float(st["paired_t_statistic"]),
                        paired_t_pvalue_raw=float(st["paired_t_pvalue_raw"]),
                        paired_t_pvalue_holm=float(st["paired_t_pvalue_holm"]),
                        wilcoxon_w_statistic=float(st["wilcoxon_w_statistic"]),
                        wilcoxon_pvalue_raw=float(st["wilcoxon_pvalue_raw"]),
                        wilcoxon_pvalue_holm=float(st["wilcoxon_pvalue_holm"]),
                        significance_decision=str(st["significance_decision"]),
                        correction_method=str(fam_def["correction_method"]),
                        correction_family_scope=str(fam_def["correction_scope"]),
                    )
                )

            total_hypotheses += len(hypotheses)
            families_out.append(
                Phase5FamilyGroup(
                    family_id=fam_id,
                    family_name=csv_fam_name,
                    intervention_arm=str(fam_def["intervention_arm"]),
                    hypothesis_count_m=int(fam_def["hypothesis_count_m"]),
                    correction_method=str(fam_def["correction_method"]),
                    correction_scope=str(fam_def["correction_scope"]),
                    hypotheses=hypotheses,
                )
            )

        # Build arm-level metrics (slice_level + patient_summary + per_patient kept strictly distinct)
        allowed_arms = (
            [str(PHASE5_FAMILY_DEFINITIONS[target_family_id]["arm_key"]), "unmitigated"]
            if target_family_id is not None
            else ["unmitigated", "postproc_alone", "clahe_alone", "median_alone", "combined_contrast", "combined_noise"]
        )
        arm_metrics: List[Phase5ArmConditionMetrics] = []
        for arm_key in allowed_arms:
            arm_dict = p5_json.get(arm_key, {})
            if not isinstance(arm_dict, dict):
                continue
            for cid, cdata in arm_dict.items():
                per_pt = cdata.get("per_patient", {})
                if patient_id is not None:
                    per_pt = {k: v for k, v in per_pt.items() if k == patient_id}
                arm_metrics.append(
                    Phase5ArmConditionMetrics(
                        arm_key=arm_key,
                        condition_id=cid,
                        slice_level={k: float(v) for k, v in cdata.get("slice_level", {}).items()},
                        patient_summary=cdata.get("patient_summary", {}),
                        per_patient=per_pt,
                    )
                )

        neg_controls = [
            Phase5NegativeControlRecord(
                condition_id=r["condition_id"],
                family=r["family"],
                severity=r["severity"],
                intervention_arm=r["intervention_arm"],
                slice_dice_fg=float(r["slice_dice_fg"]),
                patient_dice_fg_mean=float(r["patient_dice_fg_mean"]),
                patient_dice_fg_std=float(r["patient_dice_fg_std"]),
                delta_dice_mean=float(r["delta_dice_mean"]),
                paired_t_pvalue_raw=self._optional_float(r.get("paired_t_pvalue_raw")),
                wilcoxon_pvalue_raw=self._optional_float(r.get("wilcoxon_pvalue_raw")),
                significance_fwer=r["significance_fwer"],
                control_classification=r["control_classification"],
            )
            for r in neg_rows
        ]

        return Phase5Payload(
            available=True,
            experiment_id=p5_json.get("experiment_id"),
            checkpoint_sha256=p5_json.get("provenance", {}).get("checkpoint_sha256"),
            total_primary_hypotheses=total_hypotheses,
            locked_mitigation_hyperparameters=repro.get("locked_mitigation_hyperparameters", {}),
            statistical_protocol=repro.get("statistical_protocol", {}),
            families=families_out,
            arm_metrics=arm_metrics,
            negative_controls=neg_controls,
            clahe_invariance_summary=clahe_audit,
        )

    @staticmethod
    def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
        """Reads a CSV file into a list of stripped string dictionaries."""
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
        """Reads an optional JSON file safely."""
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _optional_float(val: Optional[str]) -> Optional[float]:
        """Parses an optional numeric string, returning None for N/A or empty values."""
        if val is None or val.strip().upper() in ("", "N/A", "NONE", "NAN"):
            return None
        return float(val)
