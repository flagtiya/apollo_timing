from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

from astropy import units as u
from astropy.coordinates import Angle
from pint.residuals import Residuals

from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas
from apollo_timing.inference.prior_transform import ProductSpaceParameterPrior
from apollo_timing.inference.pulse_number_reference import ensure_pulse_numbers
from apollo_timing.inference.wrap_state import apply_wraps_to_toas
from apollo_timing.models.contracts import ClusterManifest, NestedState, StructureModelConfig


@dataclass
class LikelihoodEvaluation:
    log_likelihood: float
    chi2: float | None = None
    reduced_chi2: float | None = None
    weighted_rms_us: float | None = None
    failure_code: str | None = None
    failure_message: str | None = None


class ExactPintLikelihood:
    def __init__(
        self,
        par_path: Path,
        tim_path: Path,
        cluster_manifest: ClusterManifest,
        structures: list[StructureModelConfig],
        parameter_priors: list[ProductSpaceParameterPrior],
        clock_override_path: Path | None = None,
    ) -> None:
        self.par_path = Path(par_path)
        self.tim_path = Path(tim_path)
        self.cluster_manifest = cluster_manifest
        self.structure_map = {item.structure_id: item for item in structures}
        self.parameter_priors = {item.name: item for item in parameter_priors}
        self.clock_override_path = clock_override_path
        self.seed_model = _load_pint_model(self.par_path)
        self.base_toas = _load_pint_toas(
            self.tim_path,
            seed_model=self.seed_model,
            clock_override_path=self.clock_override_path,
        )

    def evaluate_state(self, state: NestedState) -> LikelihoodEvaluation:
        if state.structure_id not in self.structure_map:
            return LikelihoodEvaluation(
                log_likelihood=float("-inf"),
                failure_code="invalid-structure",
                failure_message=f"Unknown structure_id '{state.structure_id}'.",
            )
        if len(state.wraps) != len(self.cluster_manifest.gaps):
            return LikelihoodEvaluation(
                log_likelihood=float("-inf"),
                failure_code="invalid-wrap-dimension",
                failure_message=(
                    f"Expected {len(self.cluster_manifest.gaps)} gap wraps, received {len(state.wraps)}."
                ),
            )

        try:
            model = self.build_model_from_state(state)
            wrapped_toas = self._build_wrapped_toas(model, state.wraps)
            residuals = Residuals(
                wrapped_toas,
                model,
                use_abs_phase=True,
                track_mode="use_pulse_numbers",
            )
            chi2 = float(residuals.chi2)
            sigma_seconds = wrapped_toas.get_errors().to(u.s).value
            log_sigma_term = float(sum(math.log(2.0 * math.pi * sigma * sigma) for sigma in sigma_seconds))
            log_likelihood = -0.5 * (chi2 + log_sigma_term)
            return LikelihoodEvaluation(
                log_likelihood=log_likelihood,
                chi2=chi2,
                reduced_chi2=float(residuals.reduced_chi2),
                weighted_rms_us=float(residuals.rms_weighted().to(u.us).value),
            )
        except Exception as exc:  # pragma: no cover - runtime integration path
            return LikelihoodEvaluation(
                log_likelihood=float("-inf"),
                failure_code="pint-likelihood-failed",
                failure_message=str(exc),
            )

    def build_model_from_state(self, state: NestedState):
        structure = self.structure_map[state.structure_id]
        model = _load_pint_model(self.par_path)
        if "BINARY" in model.params and structure.binary_model is not None:
            model["BINARY"].value = structure.binary_model
        active_theta = state.active_theta or state.theta
        for name, value in active_theta.items():
            if name not in model.params:
                continue
            parameter = model[name]
            if name == "RAJ":
                angle = Angle(float(value), unit=u.hourangle)
                if hasattr(parameter, "quantity"):
                    parameter.quantity = angle
                else:
                    parameter.value = angle.to_string(unit=u.hourangle, sep=":", precision=10, pad=True)
                continue
            if name == "DECJ":
                angle = Angle(float(value), unit=u.deg)
                if hasattr(parameter, "quantity"):
                    parameter.quantity = angle
                else:
                    parameter.value = angle.to_string(unit=u.deg, sep=":", precision=10, alwayssign=True, pad=True)
                continue
            parameter.value = float(value)
        return model

    def extract_state_parameter(self, model, name: str) -> float:
        parameter = model[name]
        if name == "RAJ":
            return float(parameter.quantity.to(u.hourangle).value)
        if name == "DECJ":
            return float(parameter.quantity.to(u.deg).value)
        return float(parameter.value)

    def build_reference_comparison(self, state: NestedState, reference_par_path: Path) -> dict[str, object]:
        evaluation = self.evaluate_state(state)
        if evaluation.log_likelihood == float("-inf"):
            return {
                "status": "failure",
                "failure_code": evaluation.failure_code,
                "failure_message": evaluation.failure_message,
            }
        reference_model = _load_pint_model(reference_par_path)
        reference_chi2, reference_reduced_chi2, reference_weighted_rms_us = self._evaluate_reference_model(
            reference_model
        )
        candidate_model = self.build_model_from_state(state)
        parameter_deltas = {}
        for name, value in (state.active_theta or {}).items():
            if name in reference_model.params:
                parameter_deltas[name] = float(value) - self.extract_state_parameter(reference_model, name)
        quality_ratio = (
            evaluation.reduced_chi2 / reference_reduced_chi2 if reference_reduced_chi2 not in (None, 0.0) else None
        )
        weighted_rms_ratio = (
            evaluation.weighted_rms_us / reference_weighted_rms_us
            if reference_weighted_rms_us not in (None, 0.0)
            else None
        )
        return {
            "status": "success",
            "reference_fit_chi2": reference_chi2,
            "reference_fit_reduced_chi2": reference_reduced_chi2,
            "reference_weighted_rms_us": reference_weighted_rms_us,
            "recovered_fit_chi2": evaluation.chi2,
            "recovered_fit_reduced_chi2": evaluation.reduced_chi2,
            "recovered_weighted_rms_us": evaluation.weighted_rms_us,
            "quality_ratio_to_reference": quality_ratio,
            "reduced_chi2_ratio_to_reference": quality_ratio,
            "weighted_rms_ratio_to_reference": weighted_rms_ratio,
            "parameter_deltas": parameter_deltas or None,
            "structure_id": state.structure_id,
            "wraps": list(state.wraps),
            "candidate_model_active_params": {
                name: self.extract_state_parameter(candidate_model, name)
                for name in (state.active_theta or {})
                if name in candidate_model.params
            },
        }

    def _build_wrapped_toas(self, model, wraps: list[int]):
        toas = deepcopy(self.base_toas)
        ensure_pulse_numbers(toas, model)
        return apply_wraps_to_toas(toas, self.cluster_manifest, wraps)

    def _evaluate_reference_model(self, model) -> tuple[float, float, float]:
        residuals = Residuals(self.base_toas, model, use_abs_phase=False)
        return (
            float(residuals.chi2),
            float(residuals.reduced_chi2),
            float(residuals.rms_weighted().to(u.us).value),
        )
