"""Pydantic schemas for Phase 4 robustness and Phase 5 mitigation experiment results."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class Phase4StatisticalComparison(BaseModel):
    """Paired statistical comparison of a corrupted condition against clean baseline."""

    model_config = ConfigDict(extra="forbid")

    condition_id: str
    metric: str
    n_patients: int
    degrees_of_freedom: int
    mean_clean: float
    mean_corrupt: float
    mean_difference: float
    std_difference: Optional[float] = None
    mean_relative_change_pct_patient_wise: Optional[float] = None
    relative_change_pct_of_means: Optional[float] = None
    paired_t_statistic: float
    paired_t_pvalue_raw: float
    paired_t_pvalue_holm: float
    wilcoxon_statistic: float
    wilcoxon_pvalue_raw: float
    wilcoxon_pvalue_holm: float
    correction_method: str
    correction_family_scope: str
    significance_decision: str


class Phase4ConditionRecord(BaseModel):
    """Separate slice-level and patient-level 3D metrics for a Phase 4 condition."""

    model_config = ConfigDict(extra="forbid")

    condition_id: str
    family: str
    severity: str
    param_value: float
    units: str
    display_name: str
    slice_level: Dict[str, float]
    patient_level_summary: Dict[str, Dict[str, float]]
    per_patient_metrics: Dict[str, Dict[str, float]]
    statistical_comparison_vs_clean: Optional[Phase4StatisticalComparison] = None


class Phase4Payload(BaseModel):
    """Structured Phase 4 robustness experiment dataset."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    error: Optional[str] = None
    experiment_id: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    total_test_slices: Optional[int] = None
    n_patients: Optional[int] = None
    conditions: List[Phase4ConditionRecord] = []


class Phase5HypothesisRecord(BaseModel):
    """Single pre-registered Phase 5 statistical hypothesis test with family-wise Holm adjustment."""

    model_config = ConfigDict(extra="forbid")

    family_id: str
    family_name: str
    condition_id: str
    condition_family: str
    severity: str
    intervention_arm: str
    n_patients: int
    degrees_of_freedom: int
    slice_dice_fg: float
    patient_dice_fg_mean: float
    patient_dice_fg_std: float
    dice_rv_patient_mean: float
    dice_myo_patient_mean: float
    dice_lv_patient_mean: float
    mean_baseline_dice: float
    mean_mitigated_dice: float
    mean_difference: float
    std_difference: float
    se_difference: float
    ci_95_lower: float
    ci_95_upper: float
    effect_size_cohens_d: float
    paired_t_statistic: float
    paired_t_pvalue_raw: float
    paired_t_pvalue_holm: float
    wilcoxon_w_statistic: float
    wilcoxon_pvalue_raw: float
    wilcoxon_pvalue_holm: float
    significance_decision: str
    correction_method: str
    correction_family_scope: str


class Phase5FamilyGroup(BaseModel):
    """Pre-registered Phase 5 hypothesis family preserving strict Holm family boundaries."""

    model_config = ConfigDict(extra="forbid")

    family_id: str
    family_name: str
    intervention_arm: str
    hypothesis_count_m: int
    correction_method: str
    correction_scope: str
    hypotheses: List[Phase5HypothesisRecord]


class Phase5ArmConditionMetrics(BaseModel):
    """Slice-level and patient-level metrics for a Phase 5 mitigation arm and condition."""

    model_config = ConfigDict(extra="forbid")

    arm_key: str
    condition_id: str
    slice_level: Dict[str, float]
    patient_summary: Dict[str, Dict[str, float]]
    per_patient: Dict[str, Dict[str, float]]


class Phase5NegativeControlRecord(BaseModel):
    """Off-target negative control evaluation excluded from confirmatory Holm families."""

    model_config = ConfigDict(extra="forbid")

    condition_id: str
    family: str
    severity: str
    intervention_arm: str
    slice_dice_fg: float
    patient_dice_fg_mean: float
    patient_dice_fg_std: float
    delta_dice_mean: float
    paired_t_pvalue_raw: Optional[float] = None
    wilcoxon_pvalue_raw: Optional[float] = None
    significance_fwer: str
    control_classification: str


class Phase5Payload(BaseModel):
    """Structured Phase 5 mitigation experiment dataset covering all 4 hypothesis families (19 hypotheses)."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    error: Optional[str] = None
    experiment_id: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    total_primary_hypotheses: int = 0
    locked_mitigation_hyperparameters: Dict[str, object] = {}
    statistical_protocol: Dict[str, object] = {}
    families: List[Phase5FamilyGroup] = []
    arm_metrics: List[Phase5ArmConditionMetrics] = []
    negative_controls: List[Phase5NegativeControlRecord] = []
    clahe_invariance_summary: Optional[Dict[str, object]] = None


class ExperimentsResponse(BaseModel):
    """Response for GET /api/v1/experiments."""

    model_config = ConfigDict(extra="forbid")

    phase_filter: Optional[int] = None
    family_filter: Optional[str] = None
    patient_id_filter: Optional[str] = None
    phase4: Optional[Phase4Payload] = None
    phase5: Optional[Phase5Payload] = None
