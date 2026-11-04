from __future__ import annotations

import math
from collections import defaultdict

from timing_accelerator.inference.prior_transform import ProductSpaceParameterPrior
from timing_accelerator.models.contracts import (
    ClusterManifest,
    NestedState,
    StructureModelConfig,
    StructurePosteriorSummary,
    WrapPosteriorSummary,
)


def _discrete_laplace_weights(min_wrap: int, max_wrap: int, scale: float) -> tuple[list[int], list[float]]:
    support = list(range(int(min_wrap), int(max_wrap) + 1))
    weights = [math.exp(-abs(value) / max(scale, 1.0e-9)) for value in support]
    total = sum(weights)
    return support, [weight / total for weight in weights]


def _equal_structure_weights(structure_count: int) -> list[float]:
    return [1.0 / structure_count] * structure_count


def _map_unit_to_discrete(unit_value: float, support: list[int], weights: list[float]) -> int:
    clipped = min(max(float(unit_value), 0.0), 0.999999999999)
    cumulative = 0.0
    for value, weight in zip(support, weights):
        cumulative += weight
        if clipped < cumulative:
            return value
    return support[-1]


def _linear_transform(unit_value: float, lower: float, upper: float) -> float:
    clipped = min(max(float(unit_value), 0.0), 1.0)
    return lower + (upper - lower) * clipped


class ProductSpaceHyperModel:
    def __init__(
        self,
        cluster_manifest: ClusterManifest,
        structures: list[StructureModelConfig],
        parameter_priors: list[ProductSpaceParameterPrior],
    ) -> None:
        if not structures:
            raise ValueError("Product-space hypermodel requires at least one structure.")
        self.cluster_manifest = cluster_manifest
        self.structures = structures
        self.parameter_priors = parameter_priors
        self.parameter_names = [item.name for item in parameter_priors]
        self.ndim = 1 + len(cluster_manifest.gaps) + len(parameter_priors)
        self.dimension_labels = (
            ["m_index"]
            + [f"wrap:{gap.gap_id}" for gap in cluster_manifest.gaps]
            + [f"theta:{item.name}" for item in parameter_priors]
        )

    def decode_unit_cube(self, cube: list[float]) -> NestedState:
        if len(cube) != self.ndim:
            raise ValueError(f"Expected unit cube of length {self.ndim}, got {len(cube)}.")
        structure_index = _map_unit_to_discrete(
            cube[0],
            list(range(len(self.structures))),
            _equal_structure_weights(len(self.structures)),
        )
        structure = self.structures[structure_index]
        wraps: list[int] = []
        for gap_index, gap in enumerate(self.cluster_manifest.gaps, start=1):
            support, weights = _discrete_laplace_weights(
                gap.wrap_prior.min_wrap,
                gap.wrap_prior.max_wrap,
                gap.wrap_prior.scale,
            )
            wraps.append(_map_unit_to_discrete(cube[gap_index], support, weights))

        theta: dict[str, float] = {}
        active_theta: dict[str, float] = {}
        inactive_theta: dict[str, float] = {}
        theta_offset = 1 + len(self.cluster_manifest.gaps)
        active_names = set(structure.active_fit_params)
        for prior_index, prior in enumerate(self.parameter_priors):
            value = _linear_transform(cube[theta_offset + prior_index], prior.lower, prior.upper)
            theta[prior.name] = value
            if prior.name in active_names:
                active_theta[prior.name] = value
            else:
                inactive_theta[prior.name] = value
        return NestedState(
            structure_id=structure.structure_id,
            wraps=wraps,
            theta=theta,
            active_theta=active_theta,
            inactive_theta=inactive_theta,
        )

    def decode_physical_vector(self, vector: list[float]) -> NestedState:
        if len(vector) < self.ndim:
            raise ValueError(f"Expected physical vector with at least {self.ndim} entries.")
        structure_index = int(round(vector[0]))
        structure_index = min(max(structure_index, 0), len(self.structures) - 1)
        structure = self.structures[structure_index]
        wraps = [int(round(value)) for value in vector[1 : 1 + len(self.cluster_manifest.gaps)]]
        theta_values = vector[1 + len(self.cluster_manifest.gaps) : self.ndim]

        theta: dict[str, float] = {}
        active_theta: dict[str, float] = {}
        inactive_theta: dict[str, float] = {}
        active_names = set(structure.active_fit_params)
        for prior, value in zip(self.parameter_priors, theta_values):
            theta[prior.name] = float(value)
            if prior.name in active_names:
                active_theta[prior.name] = float(value)
            else:
                inactive_theta[prior.name] = float(value)
        return NestedState(
            structure_id=structure.structure_id,
            wraps=wraps,
            theta=theta,
            active_theta=active_theta,
            inactive_theta=inactive_theta,
        )

    def prior_callback(self, cube, ndim: int, nparams: int) -> None:
        state = self.decode_unit_cube([cube[index] for index in range(ndim)])
        structure_index = next(
            index for index, structure in enumerate(self.structures) if structure.structure_id == state.structure_id
        )
        cube[0] = structure_index
        for gap_index, wrap in enumerate(state.wraps, start=1):
            cube[gap_index] = wrap
        theta_offset = 1 + len(self.cluster_manifest.gaps)
        for index, prior in enumerate(self.parameter_priors):
            cube[theta_offset + index] = state.theta[prior.name]


def summarize_posterior_samples(
    states: list[NestedState],
    log_likelihoods: list[float],
    sample_weights: list[float],
    gap_ids: list[str],
) -> dict[str, object]:
    if not (len(states) == len(log_likelihoods) == len(sample_weights)):
        raise ValueError("states, log_likelihoods, and sample_weights must have equal lengths.")
    if not states:
        return {
            "log_evidence": float("-inf"),
            "structure_posteriors": [],
            "wrap_posteriors": [],
        }

    structure_masses: dict[str, float] = defaultdict(float)
    structure_counts: dict[str, int] = defaultdict(int)
    wrap_masses: list[dict[int, float]] = [defaultdict(float) for _ in gap_ids]

    for state, weight in zip(states, sample_weights):
        structure_masses[state.structure_id] += float(weight)
        structure_counts[state.structure_id] += 1
        for gap_index, gap_id in enumerate(gap_ids):
            wrap_masses[gap_index][int(state.wraps[gap_index])] += float(weight)

    ranked_structures = sorted(structure_masses.items(), key=lambda item: item[1], reverse=True)
    structure_posteriors = [
        StructurePosteriorSummary(
            structure_id=structure_id,
            posterior_mass=posterior_mass,
            sample_count=structure_counts[structure_id],
            rank=rank,
        )
        for rank, (structure_id, posterior_mass) in enumerate(ranked_structures, start=1)
    ]

    wrap_posteriors: list[WrapPosteriorSummary] = []
    for gap_index, gap_id in enumerate(gap_ids):
        masses = wrap_masses[gap_index]
        if not masses:
            continue
        total = sum(masses.values())
        normalized = {wrap: mass / total for wrap, mass in masses.items()}
        entropy = -sum(prob * math.log(prob) for prob in normalized.values() if prob > 0.0)
        map_wrap = max(normalized.items(), key=lambda item: item[1])[0]
        wrap_posteriors.append(
            WrapPosteriorSummary(
                gap_id=gap_id,
                map_wrap=map_wrap,
                posterior_entropy=entropy,
                posterior_masses=normalized,
            )
        )

    max_log_likelihood = max(log_likelihoods)
    log_evidence = max_log_likelihood + math.log(
        sum(math.exp(item - max_log_likelihood) * weight for item, weight in zip(log_likelihoods, sample_weights))
    )
    return {
        "log_evidence": log_evidence,
        "structure_posteriors": structure_posteriors,
        "wrap_posteriors": wrap_posteriors,
    }
