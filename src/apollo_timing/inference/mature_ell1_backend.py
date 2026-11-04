"""Adapt mature ELL1 numerics without owning certification or run artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np

from apollo_timing.inference.bayesian_binary_lattice import (
    BinaryAnnealedMode,
    BinaryAnnealingStep,
    BinaryConditionalMode,
    BinaryLatticeSearchContext,
    PintBinaryLatticeProjector,
)
from apollo_timing.inference.bayesian_ell1_dual_lattice import (
    ELL1ConditionalMode,
    ELL1_FAST_NAMES,
    ELL1_SLOW_NAMES,
    anneal_ell1_conditional_mode_diagnosed,
    build_ell1_dual_lattice_search_context,
    search_ell1_orbital_cycle_cell,
)


@dataclass(frozen=True)
class _SupportView:
    context: Any

    @property
    def dimension(self) -> int:
        return len(self.context.evaluator.parameter_names)

    @property
    def physical_lower(self) -> np.ndarray:
        return self.context.support.physical_lower

    @property
    def physical_upper(self) -> np.ndarray:
        return self.context.support.physical_upper

    @property
    def search_lower(self) -> np.ndarray:
        return np.concatenate(
            (self.context.support.slow_lower, self.context.support.fast_lower)
        )

    @property
    def search_upper(self) -> np.ndarray:
        return np.concatenate(
            (self.context.support.slow_upper, self.context.support.fast_upper)
        )

    def contains_physical(self, values: Any) -> bool:
        return self.context.support.contains_physical(values)

    def denormalize_physical(self, values: Any) -> np.ndarray:
        slow, fast = self.context.support.denormalize(values)
        return self.context.coordinates.combine_physical(slow, fast)


@dataclass(frozen=True)
class _CoordinateView:
    observation_span_days: float
    family: str = "ell1"
    coordinate_names: tuple[str, ...] = (*ELL1_SLOW_NAMES, *ELL1_FAST_NAMES)


class _EvaluatorView:
    model_name = "ELL1"

    def __init__(self, evaluator: Any) -> None:
        self._evaluator = evaluator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evaluator, name)


class MatureELL1Backend:
    """Model-specific search and annealing behind the common binary lifecycle."""

    engine = "mature_ell1_phase_adapted"

    def __init__(self, *, config: Any) -> None:
        self.config = config
        self.context: Any = None
        self.last_annealing_diagnostics: dict[str, Any] = {}

    def build_context(
        self, *, par_path: Any, tim_path: Any, clock_override_path: Any
    ) -> BinaryLatticeSearchContext:
        active = self.config
        self.context = build_ell1_dual_lattice_search_context(
            par_path=par_path,
            tim_path=tim_path,
            maximum_cell_width=active.maximum_cell_width,
            support_width_scale=active.support_width_scale,
            minimum_f1_half_width=active.minimum_f1_half_width,
            minimum_eccentricity_half_width=active.minimum_eccentricity_half_width,
            astrometric_bound_arcsec=active.astrometric_bound_arcsec,
            clock_override_path=clock_override_path,
        )
        return BinaryLatticeSearchContext(
            evaluator=_EvaluatorView(self.context.evaluator),
            adapter=_CoordinateView(self.context.coordinates.observation_span_days),
            support=_SupportView(self.context),
            cells=self.context.cells,
        )

    def search_cell(
        self, *, context: Any, cell: Any, config: Any = None
    ) -> tuple[BinaryConditionalMode, ...]:
        modes = search_ell1_orbital_cycle_cell(
            context=self.context,
            cell=cell,
            config=self.config.cell_search,
        )
        result = []
        for mode in modes:
            exact = (
                PintBinaryLatticeProjector(context).evaluate_fixed_path(
                    mode.normalized_values,
                    local_integer_offsets=mode.local_integer_offsets,
                    integer_offsets=mode.integer_offsets,
                )
                if mode.posterior_in_support
                else None
            )
            result.append(
                BinaryConditionalMode(
                    **{
                        item.name: getattr(mode, item.name)
                        for item in fields(ELL1ConditionalMode)
                        if item.name != "shape_chi2"
                    },
                    model_name="ELL1",
                    shape_chi2=exact.shape_chi2 if exact is not None else np.inf,
                    phase_chi2=exact.phase_chi2 if exact is not None else np.inf,
                    total_chi2=exact.total_chi2 if exact is not None else np.inf,
                )
            )
        return tuple(result)

    def anneal_mode(
        self, *, context: Any, mode: BinaryConditionalMode, config: Any = None
    ) -> tuple[BinaryAnnealedMode, ...]:
        source = ELL1ConditionalMode(
            **{
                item.name: getattr(mode, item.name)
                for item in fields(ELL1ConditionalMode)
            }
        )
        result = anneal_ell1_conditional_mode_diagnosed(
            context=self.context,
            mode=source,
            config=(replace(self.config.annealing, **{
                item.name: getattr(config, item.name)
                for item in fields(self.config.annealing)
                if hasattr(config, item.name)
            }) if config is not None else self.config.annealing),
        )
        self.last_annealing_diagnostics = {
            "completed_schedule": result.completed_schedule,
            "failure_stage_index": result.failure_stage_index,
            "failure_reason": result.failure_reason,
            "stages": tuple(asdict(stage) for stage in result.stages),
        }
        return tuple(
            BinaryAnnealedMode(
                model_name="ELL1",
                **{
                    item.name: getattr(item_mode, item.name)
                    for item in fields(item_mode)
                    if item.name != "trace"
                },
                trace=tuple(
                    BinaryAnnealingStep(**asdict(step)) for step in item_mode.trace
                ),
            )
            for item_mode in result.modes
        )
