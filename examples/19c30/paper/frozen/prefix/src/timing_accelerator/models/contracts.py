from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class FakeToaGenerationConfig:
    tim_path: Path
    start_mjd: float
    end_mjd: float
    ntoas: int
    error_us: float = 1.0
    obs: str = "@"
    freq_mhz: float = 1400.0
    add_noise: bool = False


@dataclass
class CalibrationTargetConfig:
    par_path: Path
    pint_clock_override_path: Path | None = None


@dataclass
class GapWrapPrior:
    family: str
    min_wrap: int
    max_wrap: int
    scale: float = 1.0


@dataclass
class ClusterDefinition:
    cluster_id: str
    source_filenames: list[str]
    toa_indices: list[int] | None = None
    parent_cluster_id: str | None = None


@dataclass
class GapDefinition:
    gap_id: str
    left_cluster: str
    right_cluster: str
    wrap_prior: GapWrapPrior


@dataclass
class ClusterManifest:
    clusters: list[ClusterDefinition]
    gaps: list[GapDefinition]


@dataclass
class StructureModelConfig:
    structure_id: str
    binary_model: str | None
    active_fit_params: list[str]
    core_params: list[str]
    ignored_fit_params: list[str] | None = None
    ignored_core_params: list[str] | None = None


@dataclass
class NestedEngineConfig:
    backend: str = "pymultinest"
    live_points: int = 400
    evidence_tolerance: float = 0.5
    sampling_efficiency: float = 0.3
    seed: int = 1
    resume: bool = False
    reference_par_path: Path | None = None
    pint_clock_override_path: Path | None = None
    fixed_parameters: list[str] = field(default_factory=lambda: ["DM"])
    uniqueness_delta_logz_min: float = 5.0
    uniqueness_structure_mass_min: float = 0.95
    uniqueness_wrap_mass_min: float = 0.90
    uniqueness_wrap_entropy_max: float = 0.15
    uniqueness_quality_ratio_max: float = 1.2
    uniqueness_max_reduced_chi2: float | None = None
    uniqueness_max_weighted_rms_us: float | None = None
    refinement_mode: str = "none"
    refinement_walkers: int = 32
    refinement_steps: int = 1000
    refinement_burnin: int = 200
    refinement_seed: int = 1
    refinement_whitened: bool = True
    refinement_target: str = "top1"


@dataclass
class RecoveryStageConfig:
    stage_id: str
    kind: str
    hard_params: list[str] = field(default_factory=list)
    fit_params: list[str] = field(default_factory=list)
    local_solver_mode: str = "robust-lm"
    local_solver_max_iterations: int = 8
    local_solver_initial_damping: float = 1.0
    local_solver_damping_multiplier: float = 8.0
    local_solver_min_fractional_improvement: float = 1.0e-6
    final_block_core_blocks: list[list[str]] | None = None
    final_block_marginalized_params: list[str] | None = None
    final_block_nuisance_groups: list[list[str]] | None = None
    final_block_expansion_blocks: list[list[str]] | None = None
    final_block_score_threshold: float = 0.0
    final_block_fallback_block: list[str] | None = None
    automatic_final_structures: list["AutomaticFinalStructureConfig"] | None = None
    planning_source: str | None = None
    planning_confidence: float | None = None
    top_k: int | None = None
    grid_offsets: dict[str, list[float]] | None = None
    adaptive_exact_search_rounds: int = 0
    adaptive_exact_search_branching: int = 1
    adaptive_exact_search_shrink_factor: float = 0.5
    adaptive_exact_search_expand_edge_factor: float = 2.0
    use_posterior_guidance: bool = False
    rewrite_hard_params_from_posterior: bool = False
    rewrite_fit_params_from_posterior: bool = False
    planning_confidence_threshold: float = 0.7
    max_posterior_group_size: int | None = None
    basin_shaping_center_offsets: dict[str, float] | None = None
    basin_shaping_rank_weights: dict[str, float] | None = None
    bayesian_release_blocks: list[list[str]] | None = None
    bayesian_release_score_threshold: float = 0.0
    bayesian_release_complexity_penalty_scale: float = 1.0
    bayesian_release_fallback_order: list[list[str]] | None = None
    basin_score_threshold: float | None = None
    full_solution_stage: bool = False
    require_trusted_posterior_unlock: bool = False
    posterior_unlock_max_condition_number: float | None = None
    quality_gate_max_reduced_chi2: float | None = None
    quality_gate_max_weighted_rms_us: float | None = None


@dataclass
class SimulationConfig:
    hard_params: list[str]
    soft_params: list[str]
    truth: dict[str, float]
    grid_offsets: dict[str, list[float]]
    regime: str = "bootstrap"
    execution_layer: str | None = None
    especially_sparse_benchmark: bool = False
    top_k: int = 5
    approximation_penalty: float = 0.1
    certification_backend: str = "mock"
    truth_par_path: Path | None = None
    generate_fake_toas: FakeToaGenerationConfig | None = None
    certification_fit_params: list[str] | None = None
    post_certification_fit_params: list[str] | None = None
    reference_target: CalibrationTargetConfig | None = None
    recovery_stages: list[RecoveryStageConfig] | None = None


@dataclass
class BootstrapRequest:
    par_path: Path
    tim_path: Path
    simulation_config_path: Path
    initial_parameters: dict[str, float]
    toa_count: int
    simulation: SimulationConfig


@dataclass
class PreflightResult:
    ok_for_local_accelerator: bool
    phase_wrap_risk: str
    max_phase_sigma_cycles: float
    linearization_quality: str
    conditioning: str
    notes: list[str]


@dataclass
class PriorModel:
    center: dict[str, float]
    widths: dict[str, float]


@dataclass
class ParameterPartition:
    hard_params: list[str]
    soft_params: list[str]


@dataclass
class Candidate:
    candidate_id: str
    hard_parameters: dict[str, float]
    approximate_score: float | None = None
    exact_score: float | None = None
    basin_score: float | None = None
    fitted_parameters: dict[str, float] | None = None
    fit_chi2: float | None = None
    fit_reduced_chi2: float | None = None


@dataclass
class FailureState:
    stage: str
    code: str
    message: str
    suggestion: str | None = None


@dataclass
class PosteriorSummary:
    status: str
    source_stage: str
    guidance_strategy: str
    parameter_sigmas: dict[str, float] | None = None
    parameter_correlations: dict[str, float] | None = None
    condition_number: float | None = None
    adapted_grid_offsets: dict[str, list[float]] | None = None
    recommended_joint_parameters: list[list[str]] | None = None
    message: str | None = None


@dataclass
class PosteriorUnlockState:
    status: str
    unlocked_by_stage: str | None = None
    reason: str | None = None


@dataclass
class LikelyErrorPoint:
    code: str
    message: str
    evidence: str | None = None


@dataclass
class BayesianReleaseScore:
    block: list[str]
    approximate_log_bayes_factor: float
    log_likelihood_gain: float
    log_prior_volume: float
    log_posterior_volume: float
    laplace_constant_term: float
    coupling_strength: float
    condition_penalty: float
    heuristic_prior_bonus: float
    evidence_score: float
    selected: bool = False
    fallback_used: bool = False


@dataclass
class FinalBlockModelScore:
    core_block: list[str]
    marginalized_params: list[str]
    approximate_log_bayes_factor: float
    log_likelihood_gain: float
    log_prior_volume: float
    log_posterior_volume: float
    condition_penalty: float
    evidence_score: float
    selected: bool = False
    fallback_used: bool = False


@dataclass
class FinalBlockFollowupStep:
    step_kind: str
    block: list[str]
    cumulative_fit_params: list[str]
    fit_reduced_chi2: float | None = None
    weighted_rms_us: float | None = None
    accepted: bool = True


@dataclass
class ExactSearchRoundSummary:
    round_index: int
    scale_factor: float
    candidate_count: int
    retained_candidate_count: int
    best_candidate_id: str | None = None
    best_exact_score: float | None = None
    edge_hit: bool = False


@dataclass
class AutomaticFinalStructureConfig:
    structure_id: str
    core_block: list[str]
    marginalized_params: list[str] = field(default_factory=list)
    nuisance_groups: list[list[str]] | None = None
    expansion_blocks: list[list[str]] | None = None


@dataclass
class AutomatedFinalStructureScore:
    structure_id: str
    core_block: list[str]
    marginalized_params: list[str]
    nuisance_groups: list[list[str]] | None
    expansion_blocks: list[list[str]] | None
    approximate_log_bayes_factor: float
    log_likelihood_gain: float
    log_prior_volume: float
    log_posterior_volume: float
    complexity_penalty: float
    evidence_score: float
    selected: bool = False


@dataclass
class NestedState:
    structure_id: str
    wraps: list[int]
    theta: dict[str, float]
    active_theta: dict[str, float] | None = None
    inactive_theta: dict[str, float] | None = None


@dataclass
class StructurePosteriorSummary:
    structure_id: str
    posterior_mass: float
    sample_count: int
    rank: int


@dataclass
class WrapPosteriorSummary:
    gap_id: str
    map_wrap: int
    posterior_entropy: float
    posterior_masses: dict[int, float]


@dataclass
class NestedRunResult:
    output_dir: Path
    log_evidence: float | None
    structure_posteriors: list[StructurePosteriorSummary]
    wrap_posteriors: list[WrapPosteriorSummary]
    best_state: NestedState | None
    refined_best_state: NestedState | None
    best_log_likelihood: float | None
    posterior_sample_count: int
    diagnostics_path: Path
    uniqueness_gate_report: dict[str, Any] | None = None
    unique_solution_par_path: Path | None = None
    refinement_summary: dict[str, Any] | None = None


@dataclass
class RecoveryStageResult:
    stage_id: str
    kind: str
    local_solver_mode: str | None = None
    local_solver_diagnostics: dict[str, Any] | None = None
    planning_source: str | None = None
    planning_confidence: float | None = None
    original_hard_params: list[str] | None = None
    original_fit_params: list[str] | None = None
    planned_hard_params: list[str] | None = None
    planned_fit_params: list[str] | None = None
    input_candidate_id: str | None = None
    input_hard_parameters: dict[str, float] | None = None
    candidate_count: int = 0
    retained_candidate_count: int = 0
    exact_search_rounds: list[ExactSearchRoundSummary] | None = None
    selected_candidate: Candidate | None = None
    wall_time_seconds: float = 0.0
    weighted_rms_us: float | None = None
    basin_score_threshold: float | None = None
    basin_entry_established: bool | None = None
    full_solution_stage: bool = False
    full_solution_accepted: bool | None = None
    basin_shaping_center_offsets: dict[str, float] | None = None
    basin_shaping_rank_weights: dict[str, float] | None = None
    bayesian_release_selected_block: list[str] | None = None
    bayesian_release_heuristic_block: list[str] | None = None
    bayesian_release_scores: list[BayesianReleaseScore] | None = None
    final_block_selected_core_block: list[str] | None = None
    final_block_scores: list[FinalBlockModelScore] | None = None
    final_block_followup_steps: list[FinalBlockFollowupStep] | None = None
    automated_final_structure_selected_id: str | None = None
    automated_final_structure_scores: list[AutomatedFinalStructureScore] | None = None
    guidance_strategy: str | None = None
    posterior_unlock_state: str | None = None
    posterior_unlock_reason: str | None = None
    posterior_unlocked_by_stage: str | None = None
    posterior_summary: PosteriorSummary | None = None
    failures: list[FailureState] | None = None
    notes: list[str] | None = None


@dataclass
class CalibrationResult:
    status: str
    evaluation_mode: str
    debug_candidate_id: str | None
    reference_par_path: Path
    reference_fit_chi2: float
    reference_fit_reduced_chi2: float
    reference_weighted_rms_us: float
    recovered_fit_chi2: float | None = None
    recovered_fit_reduced_chi2: float | None = None
    recovered_weighted_rms_us: float | None = None
    quality_ratio_to_reference: float | None = None
    reduced_chi2_gap_to_reference: float | None = None
    weighted_rms_gap_us_to_reference: float | None = None
    hard_parameter_deltas: dict[str, float] | None = None
    fitted_parameter_deltas: dict[str, float] | None = None
    likely_error_points: list[LikelyErrorPoint] | None = None


@dataclass
class CertificationResult:
    candidates: list[Candidate]
    best_candidate: Candidate
    metrics: dict[str, float] | None = None


@dataclass
class PipelineResult:
    preflight: PreflightResult
    prior: PriorModel
    partition: ParameterPartition
    generated_candidates: list[Candidate]
    ranked_candidates: list[Candidate]
    certified_candidate: Candidate | None
    debug_candidate: Candidate | None
    official_status: str
    output_dir: Path
    execution_layer: str | None = None
    metrics: dict[str, float] | None = None
    stages: list[RecoveryStageResult] | None = None
    posterior_unlock_state: str | None = None
    posterior_unlock_reason: str | None = None
    posterior_unlocked_by_stage: str | None = None
    calibration: CalibrationResult | None = None
    failures: list[FailureState] | None = None


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value
