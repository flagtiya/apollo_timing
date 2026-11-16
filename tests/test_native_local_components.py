from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from pint.models import get_model
from pint.simulation import make_fake_toas_fromMJDs

from apollo_timing.backend.pint_backend import (
    _load_pint_toas,
    _prime_local_ephemeris,
    _temporary_ephemeris_override,
)
from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    TwoLevelPulseProfile,
    local_pulse_component_candidates,
)
from apollo_timing.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from apollo_timing.pipeline.apollo_timing import solve_apollo_timing
from apollo_timing.pipeline.bayesian_binary_lattice_solver import (
    BayesianBinaryLatticeSolverConfig,
)

BASE = """PSR synthetic_native_audit
RAJ 10:00:00 0
DECJ 25:00:00 0
F0 300 0
F1 -4e-15 0
PEPOCH 59000
DM 0
EPHEM DE421
CLK TT(TAI)
PLANET_SHAPIRO N
CORRECT_TROPOSPHERE N
BINARY DD
PB .17 0
A1 .5 0
ECC .001 0
OM 40 0
T0 59000.01 1
"""


def profile_for(component):
    return TwoLevelPulseProfile(
        normalized_local_residuals=np.zeros(sum(map(len, component))),
        session_intercepts=np.zeros(len(component)),
        session_intercept_variances=np.ones(len(component)),
        local_integer_offsets=component,
        local_chi2=0.0,
        local_log_likelihood=0.0,
    )


@pytest.mark.parametrize("component", [((0, 0), (0,)), ((0, 1, -1), (0, 2))])
def test_component_proposals_keep_reference_and_profiled_without_duplicates(component):
    candidates = local_pulse_component_candidates(profile_for(component))
    reference = tuple(tuple(0 for _ in row) for row in component)
    assert candidates[0] == reference
    assert component in candidates
    assert len(candidates) == len(set(candidates)) <= 2


@pytest.mark.parametrize("backend", ["generic", "ell1"])
@pytest.mark.parametrize("failing_component", [None, "reference", "profiled"])
def test_native_acquisition_tries_both_paths_independently(
    monkeypatch, backend, failing_component
):
    if backend == "generic":
        from apollo_timing.inference import bayesian_binary_lattice as module
        search = module.search_binary_orbital_cycle_cell
        dimension = 1
    else:
        from apollo_timing.inference import bayesian_ell1_dual_lattice as module
        search = module.search_ell1_orbital_cycle_cell
        dimension = 9
    reference, profiled = ((0, 0), (0,)), ((0, 1), (0,))
    calls = []
    support = SimpleNamespace(
        dimension=dimension,
        search_center=np.zeros(dimension),
        slow_center=np.zeros(2), fast_center=np.zeros(7),
        normalize_search=lambda x: x,
        normalize=lambda *x: np.zeros(dimension),
        denormalize_physical=lambda x: x,
        denormalize=lambda x: (x[:2], x[2:]),
        contains_physical=lambda x: True,
    )
    evaluator = SimpleNamespace(
        model_name="DD" if backend == "generic" else "ELL1",
        evaluate_values=lambda *args, **kwargs: profile_for(profiled),
    )
    context = SimpleNamespace(
        support=support, evaluator=evaluator, continuous_prior_components=(),
        adapter=SimpleNamespace(orbital_coordinate_index=None, phase_coordinate_index=None),
        coordinates=SimpleNamespace(combine_physical=lambda slow, fast: np.r_[slow, fast]),
    )

    class Projector:
        def __init__(self, ctx):
            self.support, self.evaluator, self.coordinates = (
                ctx.support, ctx.evaluator, ctx.coordinates
            )

        def profile_full_shape(self, state, *, local_integer_offsets, **kwargs):
            calls.append(local_integer_offsets)
            if (failing_component == "reference" and local_integer_offsets == reference
                    or failing_component == "profiled" and local_integer_offsets == profiled):
                raise ValueError("Deliberate failure of one discrete proposal")
            return SimpleNamespace(
                normalized_values=np.asarray(state), parameter_values=np.asarray(state),
                profile=profile_for(local_integer_offsets),
            )

        def linearize_full(self, conditional, **kwargs):
            return conditional

    path = SimpleNamespace(integer_offsets=(0,), spin_posterior_mean=np.zeros(dimension),
                           posterior_weight=1.0)
    marginal = SimpleNamespace(
        path_modes=(path,), log_likelihood=0.0, shape_quadratic=0.0,
        lattice_mass_converged=True, lattice_log_mass_increment=0.0,
        lattice_visited_nodes=1,
    )
    path.squared_mahalanobis = 0.0
    monkeypatch.setattr(module, "marginalize_spin_and_integer_paths", lambda *a, **k: marginal)
    lower, upper = -np.ones(dimension), np.ones(dimension)
    if backend == "generic":
        monkeypatch.setattr(module, "PintBinaryLatticeProjector", Projector)
        monkeypatch.setattr(module, "normalized_bounds_for_binary_cell", lambda *a: (lower, upper))
        monkeypatch.setattr(module, "_project_to_support_segment", lambda **k: k["reference"])
        def exact(**kwargs):
            return SimpleNamespace(
                normalized_values=np.zeros(dimension), parameter_values=np.zeros(dimension),
                local_integer_offsets=kwargs["local_integer_offsets"],
                shape_chi2=0.0, phase_chi2=0.0, total_chi2=0.0,
            )
        monkeypatch.setattr(module, "_select_coherent_fixed_path_state", exact)
    else:
        monkeypatch.setattr(module, "PintELL1DualLatticeProjector", Projector)
        monkeypatch.setattr(module, "normalized_bounds_for_ell1_cell", lambda *a: (lower, upper))
        monkeypatch.setattr(module, "ell1_physical_prior_covariance", lambda **k: np.eye(dimension))
        monkeypatch.setattr(module, "transport_ell1_posterior_to_support", lambda **k: None)
    modes = search(context=context, cell=SimpleNamespace(cell_id="test"))
    assert set(calls) == {reference, profiled}
    expected = {reference, profiled}
    if failing_component is not None:
        expected.remove(reference if failing_component == "reference" else profiled)
    assert {mode.local_integer_offsets for mode in modes} == expected


@pytest.fixture
def native_case(tmp_path):
    kernel = Path.home() / ".cache/timing-accelerator/ephemerides/de421.bsp"
    if not kernel.is_file():
        pytest.skip("Offline DE421 is required for native integration.")
    truth = get_model(StringIO(BASE))
    epochs = np.r_[59000.035 + np.arange(100) * 60 / 86400,
                   59003.08 + np.arange(100) * 60 / 86400]
    with _temporary_ephemeris_override(ephem_file=kernel):
        _prime_local_ephemeris("DE421")
        toas = make_fake_toas_fromMJDs(
            Time(epochs, format="mjd", scale="tdb"), truth, obs="@",
            freq=1400 * u.MHz, error=1 * u.us, add_noise=False,
        )
        phase = truth.phase(toas, abs_phase=True)
        expected_pn = np.rint(np.asarray(phase.int.value, dtype=np.longdouble)
                              + np.asarray(phase.frac.value, dtype=np.longdouble))
        if "pulse_number" in toas.table.colnames:
            toas.table.remove_column("pulse_number")
        tim = tmp_path / "no_pn.tim"
        toas.write_TOA_file(str(tim), format="tempo2", include_pn=False)
        bounds = {"T0": (float(truth.T0.value) - 30 / 86400,
                          float(truth.T0.value) + 30 / 86400)}
        yield SimpleNamespace(truth=truth, tim=tim, bounds=bounds, expected_pn=expected_pn)


@pytest.mark.parametrize(
    "seed_error_seconds, corrupt_reference",
    [(0.0, False), (-20.0, False), (0.0, True)],
)
def test_public_entry_recovers_reference_path_and_disk_reload(
    tmp_path, native_case, seed_error_seconds, corrupt_reference
):
    from apollo_timing.inference.bayesian_binary_lattice import build_phase_lattice_search_context
    from apollo_timing.pipeline.bayesian_binary_lattice_solver import SOLVER_SCHEMA_VERSION
    import json

    case = native_case
    seed = deepcopy(case.truth)
    seed.T0.value = np.longdouble(seed.T0.value) + np.longdouble(seed_error_seconds) / 86400
    par = tmp_path / "seed.par"
    write_pint_model_parfile_roundtrip_safe(model=seed, path=par)
    if corrupt_reference:
        # Supplied per-TOA pulse numbers are a reference, not necessarily correct.
        # This synthetic one-count slip requires the profiled alternative path.
        rows = case.tim.read_text().splitlines()
        observation = 0
        for index, row in enumerate(rows):
            fields = row.split()
            if len(fields) < 5 or fields[0] in {"FORMAT", "C", "#"}:
                continue
            slip = int(observation % 100 >= 50)
            rows[index] += " -pn " + str(int(case.expected_pn[observation]) + slip)
            observation += 1
        assert observation == len(case.expected_pn)
        case.tim.write_text("\n".join(rows) + "\n")
    original_par, original_tim = par.read_bytes(), case.tim.read_bytes()
    context = build_phase_lattice_search_context(
        par_path=par, tim_path=case.tim, custom_bounds=case.bounds
    )
    truth_values = np.array([float(case.truth.T0.value)])
    assert context.support.contains_physical(truth_values)
    reference_chi2 = context.evaluator.evaluate_fixed_local_component(truth_values).local_chi2
    if corrupt_reference:
        assert reference_chi2 > 1e6
        assert context.evaluator.evaluate_values(truth_values).local_chi2 < 1e-3
    else:
        assert reference_chi2 < 1e-3
    if seed_error_seconds:
        # The original acquisition selected only this wrong nonzero component.
        at_seed = context.evaluator.evaluate_values(context.evaluator.base_parameter_values)
        assert any(any(row) for row in at_seed.local_integer_offsets)
    config = BayesianBinaryLatticeSolverConfig(
        custom_bounds=case.bounds, use_mature_ell1_engine=False,
        plan_observations_on_failure=False,
    )
    output = tmp_path / "native"
    kwargs = dict(par_path=par, tim_path=case.tim, output_dir=output, binary_config=config,
                  source_type="binary", discover_undeclared_binary=False)
    result = solve_apollo_timing(**kwargs, resume=False, output_policy="error")
    assert result.accepted and result.target_reached
    assert result.chi2 < 1e-3 and result.disk_reload_chi2 < 1e-3
    assert abs(result.chi2 - result.disk_reload_chi2) < 1e-6
    assert tuple(result.fit_parameters) == ("T0",)
    saved = get_model(result.solution_par)
    assert saved.RAJ.frozen and saved.DECJ.frozen and saved.F0.frozen and saved.F1.frozen
    assert saved.RAJ.value == seed.RAJ.value and saved.DECJ.value == seed.DECJ.value
    assert abs(float(saved.T0.value - case.truth.T0.value)) * 86400 < 1e-4
    reloaded = _load_pint_toas(Path(result.solution_tim), seed_model=saved)
    actual_pn = np.asarray(reloaded.table["pulse_number"], dtype=np.longdouble)
    assert len(actual_pn) == 200
    np.testing.assert_array_equal(actual_pn - actual_pn[0], case.expected_pn - case.expected_pn[0])
    assert par.read_bytes() == original_par and case.tim.read_bytes() == original_tim
    resumed = solve_apollo_timing(**kwargs, resume=True, output_policy="error")
    assert resumed.resumed and resumed.accepted
    audit_path = output / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["schema_version"] = SOLVER_SCHEMA_VERSION - 1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises((ValueError, RuntimeError), match="schema|Schema|version"):
        solve_apollo_timing(**kwargs, resume=True, output_policy="error")


def test_mature_ell1_reference_component_uses_native_fixed_path_profile(tmp_path):
    from apollo_timing.inference.bayesian_ell1_dual_lattice import (
        PintELL1DualLatticeProjector,
        _local_components_at_state,
        build_ell1_dual_lattice_search_context,
    )

    kernel = Path.home() / ".cache/timing-accelerator/ephemerides/de421.bsp"
    if not kernel.is_file():
        pytest.skip("Offline DE421 is required for native integration.")
    text = BASE.replace("BINARY DD", "BINARY ELL1").replace(
        "ECC .001 0\nOM 40 0\nT0 59000.01 1",
        "EPS1 .00064 1\nEPS2 .00077 1\nTASC 59000.01 1",
    )
    # Mature ELL1 has an explicit nine-parameter native contract. These fit
    # flags belong only to this synthetic consumer regression, not real data.
    lines = []
    for line in text.splitlines():
        if line.split()[0] in {"RAJ", "DECJ", "F0", "F1", "PB", "A1"}:
            line = line.rsplit(" ", 1)[0] + " 1"
        lines.append(line)
    truth = get_model(StringIO("\n".join(lines) + "\n"))
    epochs = np.r_[59000.035 + np.arange(100) * 60 / 86400,
                   59003.08 + np.arange(100) * 60 / 86400]
    with _temporary_ephemeris_override(ephem_file=kernel):
        _prime_local_ephemeris("DE421")
        toas = make_fake_toas_fromMJDs(
            Time(epochs, format="mjd", scale="tdb"), truth, obs="@",
            freq=1400 * u.MHz, error=1 * u.us, add_noise=False,
        )
        if "pulse_number" in toas.table.colnames:
            toas.table.remove_column("pulse_number")
        tim = tmp_path / "ell1_no_pn.tim"
        toas.write_TOA_file(str(tim), format="tempo2", include_pn=False)
        seed = deepcopy(truth)
        seed.TASC.value = np.longdouble(seed.TASC.value) - np.longdouble(20) / 86400
        par = tmp_path / "ell1_seed.par"
        write_pint_model_parfile_roundtrip_safe(model=seed, path=par)
        context = build_ell1_dual_lattice_search_context(par_path=par, tim_path=tim)
        projector = PintELL1DualLatticeProjector(context)
        state = context.support.normalize(
            context.support.slow_center, context.support.fast_center
        )
        components = _local_components_at_state(projector, state, radius=0)
        assert len(components) == 2
        assert not any(any(row) for row in components[0])
        assert any(any(row) for row in components[1])
        profiles = [
            projector.profile_full_shape(
                state, normalized_lower=-np.ones(9) + 1e-14,
                normalized_upper=np.ones(9) - 1e-14,
                local_integer_offsets=component, maximum_function_evaluations=60,
            )
            for component in components
        ]
        assert profiles[0].profile.local_chi2 < 1e-3
        for component, profile in zip(components, profiles, strict=True):
            assert profile.profile.local_integer_offsets == component
