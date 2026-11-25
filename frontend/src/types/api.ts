/** Exact TypeScript interfaces matching FastAPI backend schemas under api/schemas/. */

export interface ApiErrorDetail {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface ApiErrorPayload {
  error: ApiErrorDetail;
}

export interface ArtifactStatus {
  name: string;
  available: boolean;
  valid: boolean;
  detail: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "unavailable" | string;
  backend_version: string;
  python_version: string;
  pytorch_available: boolean;
  pytorch_version: string | null;
  model_initialized: boolean;
  canonical_checkpoint_id: string;
  loaded_checkpoint_ids: string[];
  model_load_error: string | null;
  device_active: string;
  cuda_available: boolean;
  cuda_device_count: number;
  research_artifacts_available: boolean;
  artifacts: ArtifactStatus[];
}

export interface ModelSpecification {
  architecture: string;
  parameter_count_total: number;
  parameter_count_trainable: number;
  in_channels: number;
  out_channels: number;
  base_channels: number;
  depth: number;
  dropout: number;
  use_batchnorm: boolean;
  bilinear_upsample: boolean;
  input_dimensions: number[];
  output_dimensions: number[];
  class_mapping: Record<string, string>;
}

export interface CheckpointOverview {
  checkpoint_id: string;
  display_name: string;
  is_canonical: boolean;
  expected_sha256: string;
  verified_sha256: string | null;
  integrity_verified: boolean;
  available: boolean;
  epochs_completed: number;
  best_epoch: number | null;
  best_val_foreground_dice: number | null;
}

export interface DatasetOverview {
  dataset_name: string;
  evaluated_phase: string;
  total_patients: number | null;
  total_slices: number | null;
  test_patient_count: number;
  test_slice_count: number;
  test_foreground_slice_count: number | null;
  test_patient_ids: string[];
  pathology_distribution_test: Record<string, number>;
}

export interface AggregatedMetricSummary {
  condition_id: string;
  aggregation_level: "slice_level" | "patient_level_3d_pooled" | string;
  sample_count: number;
  sample_unit: string;
  mean_dice_foreground: number | null;
  std_dice_foreground: number | null;
  mean_iou_foreground: number | null;
  dice_per_class: Record<string, number>;
  iou_per_class: Record<string, number> | null;
  source_artifact: string;
}

export interface ExperimentSummaryCard {
  experiment_id: string;
  phase: number;
  title: string;
  available: boolean;
  checkpoint_sha256: string | null;
  conditions_count: number | null;
  hypothesis_families_count: number | null;
  total_primary_hypotheses: number | null;
  key_findings: string[];
}

export interface OverviewResponse {
  project_title: string;
  model: ModelSpecification;
  checkpoints: CheckpointOverview[];
  dataset: DatasetOverview;
  baseline_metrics: AggregatedMetricSummary[];
  experiment_summaries: ExperimentSummaryCard[];
}

export interface Phase4StatisticalComparison {
  condition_id: string;
  metric: string;
  n_patients: number;
  degrees_of_freedom: number;
  mean_clean: number;
  mean_corrupt: number;
  mean_difference: number;
  std_difference: number | null;
  mean_relative_change_pct_patient_wise: number | null;
  relative_change_pct_of_means: number | null;
  paired_t_statistic: number;
  paired_t_pvalue_raw: number;
  paired_t_pvalue_holm: number;
  wilcoxon_statistic: number;
  wilcoxon_pvalue_raw: number;
  wilcoxon_pvalue_holm: number;
  correction_method: string;
  correction_family_scope: string;
  significance_decision: string;
}

export interface PatientSummaryStat {
  mean: number;
  std: number;
  median: number;
  iqr: number;
  q25: number;
  q75: number;
  ci_95_lower: number;
  ci_95_upper: number;
  n_patients: number;
}

export interface Phase4ConditionRecord {
  condition_id: string;
  family: string;
  severity: string;
  param_value: number;
  units: string;
  display_name: string;
  slice_level: Record<string, number>;
  patient_level_summary: Record<string, PatientSummaryStat>;
  per_patient_metrics: Record<string, Record<string, number>>;
  statistical_comparison_vs_clean: Phase4StatisticalComparison | null;
}

export interface Phase4Payload {
  available: boolean;
  error: string | null;
  experiment_id: string | null;
  checkpoint_sha256: string | null;
  total_test_slices: number | null;
  n_patients: number | null;
  conditions: Phase4ConditionRecord[];
}

export interface Phase5HypothesisRecord {
  family_id: string;
  family_name: string;
  condition_id: string;
  condition_family: string;
  severity: string;
  intervention_arm: string;
  n_patients: number;
  degrees_of_freedom: number;
  slice_dice_fg: number;
  patient_dice_fg_mean: number;
  patient_dice_fg_std: number;
  dice_rv_patient_mean: number;
  dice_myo_patient_mean: number;
  dice_lv_patient_mean: number;
  mean_baseline_dice: number;
  mean_mitigated_dice: number;
  mean_difference: number;
  std_difference: number;
  se_difference: number;
  ci_95_lower: number;
  ci_95_upper: number;
  effect_size_cohens_d: number;
  paired_t_statistic: number;
  paired_t_pvalue_raw: number;
  paired_t_pvalue_holm: number;
  wilcoxon_w_statistic: number;
  wilcoxon_pvalue_raw: number;
  wilcoxon_pvalue_holm: number;
  significance_decision: string;
  correction_method: string;
  correction_family_scope: string;
}

export interface Phase5FamilyGroup {
  family_id: string;
  family_name: string;
  intervention_arm: string;
  hypothesis_count_m: number;
  correction_method: string;
  correction_scope: string;
  hypotheses: Phase5HypothesisRecord[];
}

export interface Phase5ArmConditionMetrics {
  arm_key: string;
  condition_id: string;
  slice_level: Record<string, number>;
  patient_summary: Record<string, PatientSummaryStat>;
  per_patient: Record<string, Record<string, number>>;
}

export interface Phase5NegativeControlRecord {
  condition_id: string;
  family: string;
  severity: string;
  intervention_arm: string;
  slice_dice_fg: number;
  patient_dice_fg_mean: number;
  patient_dice_fg_std: number;
  delta_dice_mean: number;
  paired_t_pvalue_raw: number | null;
  wilcoxon_pvalue_raw: number | null;
  significance_fwer: string;
  control_classification: string;
}

export interface ClaheCausalDecomposition {
  clean_baseline_dice: number;
  severe_unmitigated_dice: number;
  unmitigated_drop: number;
  clahe_reported_rescue: number;
  zscore_alone_rescue: number;
  clahe_net_effect_relative_to_clean: number;
  clahe_net_effect_relative_to_zscore_alone: number;
}

export interface ClaheInvarianceSummary {
  sample_size?: { patients: number; slices: number };
  stage_C_vs_D_unit_interval?: {
    max_absolute_difference: number;
    mean_absolute_difference: number;
    all_slices_max_diff_below_1e_5: boolean;
    scientific_interpretation: string;
  };
  stage_E_vs_F_clahe_output?: {
    max_absolute_difference: number;
    mean_absolute_difference: number;
    prediction_pixel_agreement_mean_pct: number;
    scientific_interpretation: string;
  };
  causal_decomposition?: ClaheCausalDecomposition;
}

export interface Phase5Payload {
  available: boolean;
  error: string | null;
  experiment_id: string | null;
  checkpoint_sha256: string | null;
  total_primary_hypotheses: number;
  locked_mitigation_hyperparameters: Record<string, unknown>;
  statistical_protocol: Record<string, unknown>;
  families: Phase5FamilyGroup[];
  arm_metrics: Phase5ArmConditionMetrics[];
  negative_controls: Phase5NegativeControlRecord[];
  clahe_invariance_summary: ClaheInvarianceSummary | null;
}

export interface ExperimentsResponse {
  phase_filter: number | null;
  family_filter: string | null;
  patient_id_filter: string | null;
  phase4: Phase4Payload | null;
  phase5: Phase5Payload | null;
}
