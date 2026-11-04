from __future__ import annotations

import os
import time
import warnings
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, solar_system_ephemeris
from loguru import logger
import pint.observatory as observatory
import pint.observatory.global_clock_corrections as global_clock_corrections
import pint.observatory.topo_obs as topo_obs
import pint.solar_system_ephemerides as solar_system_ephemerides
from pint.exceptions import ClockCorrectionError
from pint.observatory.clock_file import ClockFile, GlobalClockFile
from pint.models.model_builder import ModelBuilder, convert_tcb_tdb
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_uniform
from pint.toa import get_TOAs

from apollo_timing.models.contracts import (
    BootstrapRequest,
    CalibrationResult,
    Candidate,
    CertificationResult,
    FailureState,
    LikelyErrorPoint,
    PosteriorSummary,
    SimulationConfig,
)
from apollo_timing.io.par_compatibility import normalize_par_text
from apollo_timing.io.artifacts import sha256_file


class TimingResourceError(RuntimeError):
    """Required timing resources are unavailable, not a rejected orbit."""


def _resolve_clock_directory(clock_dir=None) -> Path | None:
    explicit = (clock_dir or os.environ.get("APOLLO_TIMING_CLOCK_DIR")
                or os.environ.get("PINT_CLOCK_OVERRIDE"))
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise TimingResourceError(f"Local clock directory does not exist: '{root}'.")
        return root
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "clocks",
        repo_root / "runtime" / "clocks",
        repo_root / "result" / "_resources" / "clocks",
    ]
    cache_root = os.environ.get("APOLLO_TIMING_CACHE_DIR")
    if cache_root:
        candidates.append(Path(cache_root).expanduser() / "clocks")
    candidates.append(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                      / "apollo-timing" / "clocks")
    return next((path for path in candidates if path.is_dir()), None)


def _local_ephemeris_identity(par_path):
    ephem = "DE421"
    for line in Path(par_path).read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) >= 2 and words[0].upper() == "EPHEM":
            ephem = words[1]
    path = _resolve_local_ephemeris_path(ephem)
    if path is None:
        return None
    try:
        return dict(ephem=ephem, path=str(path.resolve()), sha256=sha256_file(path))
    except OSError as exc:
        raise TimingResourceError(f"Cannot read local ephemeris '{path}': {exc}") from exc


@contextmanager
def _temporary_ephemeris_override(ephem_file=None, ephem_dir=None):
    if ephem_file is not None and ephem_dir is not None:
        raise ValueError("Specify an ephemeris file or directory, not both.")
    keys = ("APOLLO_TIMING_EPHEM_PATH", "APOLLO_TIMING_EPHEM_DIR")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if ephem_file is not None or ephem_dir is not None:
            for key in keys:
                os.environ.pop(key, None)
            key = keys[0] if ephem_file is not None else keys[1]
            os.environ[key] = str(Path(ephem_file if ephem_file is not None else ephem_dir).expanduser().resolve())
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_pint_model(par_path) -> object:
    path = Path(par_path)
    normalized = normalize_par_text(path.read_text(encoding="utf-8"))
    source = StringIO(normalized.text)
    source.name = str(path)
    model = ModelBuilder()(source, allow_tcb=True)
    if "UNITS" in model.params and model["UNITS"].value == "TCB":
        convert_tcb_tdb(model)
    return model


def _resolve_local_ephemeris_path(ephem: str) -> Path | None:
    ephem_name = str(ephem or "DE421")
    ephem_bsp = f"{ephem_name.lower()}.bsp"
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [Path(__file__).resolve().parents[1] / "data" / "ephemerides" / ephem_bsp]

    explicit_path = os.environ.get("APOLLO_TIMING_EPHEM_PATH")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise TimingResourceError(
                f"APOLLO_TIMING_EPHEM_PATH is not a readable ephemeris file: '{path}'. "
                "Correct the path or unset it to allow cache/network lookup."
            )
        return path

    explicit_dir = os.environ.get("APOLLO_TIMING_EPHEM_DIR")
    if explicit_dir:
        path = Path(explicit_dir).expanduser() / ephem_bsp
        if not path.is_file():
            raise TimingResourceError(
                f"Requested {ephem_name} is missing from APOLLO_TIMING_EPHEM_DIR: '{path}'. "
                "Supply this kernel or unset the directory to allow cache/network lookup."
            )
        return path

    cache_root = os.environ.get("APOLLO_TIMING_CACHE_DIR")
    if cache_root:
        candidates.append(Path(cache_root) / "ephemerides" / ephem_bsp)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(
            Path(xdg_cache) / "apollo-timing" / "ephemerides" / ephem_bsp
        )
    candidates.append(
        Path.home() / ".cache" / "apollo-timing" / "ephemerides" / ephem_bsp
    )

    candidates.extend(
        [
            repo_root / "runtime" / "ephemerides" / ephem_bsp,
            repo_root / "result" / "_resources" / ephem_bsp,
            Path("/opt/apollo-timing/runtime") / ephem_bsp,
            repo_root / "docker" / "runtime" / ephem_bsp,
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _prime_local_ephemeris(ephem: str) -> Path | None:
    ephem_name = str(ephem or "DE421")
    local_path = _resolve_local_ephemeris_path(ephem_name)
    if local_path is None:
        return None
    try:
        # PINT's loader can ignore a new path if the name was already loaded,
        # and silently download after a local I/O failure. Explicit files must
        # instead be activated directly and fail locally if they are invalid.
        solar_system_ephemeris.set(str(local_path.resolve()))
        solar_system_ephemerides.loaded_ephems[ephem_name.lower()] = str(local_path.resolve())
        logger.info("Using local {} ephemeris: {}", ephem_name, local_path.resolve())
    except (OSError, ValueError) as exc:
        raise TimingResourceError(
            f"Cannot read local {ephem_name} ephemeris '{local_path}'. "
            "Check the BSP file and permissions; no network fallback was attempted. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    return local_path


def _record_clock_coverage(clock, t, reports, path, name):
    if not clock.valid_beyond_ends and len(clock.time):
        outside = (t < clock.time[0]) | (t > clock.time[-1])
        if np.any(outside):
            reports.append(dict(
                file=str(path), clock=name, toa_count=int(np.size(outside)),
                affected_toa_count=int(np.count_nonzero(outside)),
                clock_mjd_start=float(clock.time[0].mjd),
                clock_mjd_end=float(clock.time[-1].mjd),
                requested_mjd_start=float(np.min(t.mjd)),
                requested_mjd_end=float(np.max(t.mjd)),
                action="nearest_endpoint_correction_outside_coverage",
            ))


@contextmanager
def _temporary_cached_clock_fallback():
    previous_update, previous_evaluate = GlobalClockFile.update, GlobalClockFile.evaluate
    reports = []

    def update(clock):
        old_file, old_path, old_hash = clock.clock_file, clock.f, clock.hash
        try:
            return previous_update(clock)
        except OSError as exc:
            if not len(old_file.time):
                raise
            # A failed renewal must not discard a usable in-memory clock. Parse
            # errors still propagate; this fallback is only for I/O failures.
            clock.clock_file, clock.f, clock.hash = old_file, old_path, old_hash
            warnings.warn(
                f"Could not update cached clock '{clock.friendly_name}': {exc}. "
                f"Continuing with the existing cached clock '{old_path}'; "
                "review its coverage or select local files with --clock-dir DIRECTORY.",
                UserWarning, stacklevel=2,
            )

    def evaluate(clock, t, limits="warn"):
        if not len(clock.time):
            raise TimingResourceError(
                f"The cached clock '{clock.friendly_name}' is empty. "
                "Provide a nonempty file with --clock-dir DIRECTORY."
            )
        values = previous_evaluate(clock, t, limits=limits)
        _record_clock_coverage(clock.clock_file, t, reports, clock.f, clock.friendly_name)
        return values

    GlobalClockFile.update, GlobalClockFile.evaluate = update, evaluate
    try:
        yield reports
    finally:
        GlobalClockFile.update, GlobalClockFile.evaluate = previous_update, previous_evaluate


@contextmanager
def _temporary_clock_override(clock_override_path):
    clock_override_path = _resolve_clock_directory(clock_override_path)
    if clock_override_path is None:
        with _temporary_cached_clock_fallback() as reports:
            yield reports
        return

    root = Path(clock_override_path).expanduser().resolve()
    if not root.is_dir():
        raise TimingResourceError(f"Local clock directory does not exist: '{root}'.")
    previous = os.environ.get("PINT_CLOCK_OVERRIDE")
    previous_url_base = global_clock_corrections.global_clock_correction_url_base
    previous_url_mirrors = global_clock_corrections.global_clock_correction_url_mirrors
    os.environ["PINT_CLOCK_OVERRIDE"] = str(root)
    local_repo_base = root.as_uri().rstrip("/") + "/"
    global_clock_corrections.global_clock_correction_url_base = local_repo_base
    global_clock_corrections.global_clock_correction_url_mirrors = [local_repo_base]
    previous_finders = observatory.find_clock_file, topo_obs.find_clock_file
    previous_gps = observatory._gps_clock
    previous_bipm = observatory._bipm_clock_versions.copy()
    previous_clocks = {name: site._clock for name, site in observatory.Observatory._registry.items()
                       if hasattr(site, "_clock")}
    coverage_warnings = []

    def find_local_clock(name, format, bogus_last_correction=False, url_base=None,
                         clock_dir=None, valid_beyond_ends=False):
        if clock_dir is not None and str(clock_dir).upper() != "PINT":
            return previous_finders[0](
                name, format, bogus_last_correction=bogus_last_correction,
                url_base=url_base, clock_dir=clock_dir, valid_beyond_ends=valid_beyond_ends,
            )
        subdir = "tempo/clock" if format == "tempo" else "T2runtime/clock"
        path = next((p for p in (root / name, root / subdir / name) if p.is_file()), None)
        if path is None:
            raise TimingResourceError(
                f"Required clock file '{name}' is missing from '{root}'. "
                f"Place it directly in this directory or under '{subdir}'. "
                "A local clock directory needs no index.txt; no network fallback was attempted."
            )
        try:
            logger.info("Using local clock file: {}", path)
            clock = ClockFile.read(
                path, format=format, bogus_last_correction=bogus_last_correction,
                friendly_name=name, valid_beyond_ends=valid_beyond_ends,
            )
            original_evaluate = clock.evaluate

            def evaluate(t, limits="warn"):
                _record_clock_coverage(clock, t, coverage_warnings, path, name)
                return original_evaluate(t, limits=limits)

            clock.evaluate = evaluate
            return clock
        except (OSError, ValueError) as exc:
            raise TimingResourceError(f"Cannot read clock file '{path}': {exc}") from exc

    # PINT caches clocks on global observatory instances, independently of the
    # override directory. Isolate each local load, including failed loads.
    observatory.find_clock_file = topo_obs.find_clock_file = find_local_clock
    observatory._gps_clock = None
    observatory._bipm_clock_versions.clear()
    for name in previous_clocks:
        observatory.Observatory._registry[name]._clock = None
    try:
        yield coverage_warnings
    finally:
        observatory.find_clock_file, topo_obs.find_clock_file = previous_finders
        observatory._gps_clock = previous_gps
        observatory._bipm_clock_versions.clear()
        observatory._bipm_clock_versions.update(previous_bipm)
        for name, site in observatory.Observatory._registry.items():
            if hasattr(site, "_clock"):
                site._clock = previous_clocks.get(name)
        global_clock_corrections.global_clock_correction_url_base = previous_url_base
        global_clock_corrections.global_clock_correction_url_mirrors = previous_url_mirrors
        if previous is None:
            os.environ.pop("PINT_CLOCK_OVERRIDE", None)
        else:
            os.environ["PINT_CLOCK_OVERRIDE"] = previous


def _load_pint_toas(
    tim_path,
    seed_model,
    clock_override_path=None,
):
    ephem = seed_model["EPHEM"].value if "EPHEM" in seed_model.params else "DE421"
    planets = False
    if "PLANET_SHAPIRO" in seed_model.params:
        planet_shapiro = seed_model["PLANET_SHAPIRO"].value
        if isinstance(planet_shapiro, str):
            planets = planet_shapiro.strip().upper() in {"1", "TRUE", "Y", "YES"}
        else:
            planets = bool(planet_shapiro)
    _prime_local_ephemeris(ephem)
    with _temporary_clock_override(clock_override_path) as coverage_warnings:
        try:
            with warnings.catch_warnings(record=True) as caught:
                toas = get_TOAs(
                    str(tim_path),
                    ephem=ephem,
                    planets=planets,
                    include_bipm=False,
                    # Preserve serialized APOLLO pulse paths during certification reload.
                    include_pn=True,
                )
            for warning in caught:
                message = str(warning.message)
                if "Data points out of range in clock file" in message:
                    details = next((item for item in coverage_warnings
                                    if f"'{item['clock']}'" in message), None)
                    if details is not None:
                        message += (
                            f"; clock coverage MJD {details['clock_mjd_start']:.6f}--"
                            f"{details['clock_mjd_end']:.6f}; requested MJD "
                            f"{details['requested_mjd_start']:.6f}--{details['requested_mjd_end']:.6f}; "
                            f"{details['affected_toa_count']}/{details['toa_count']} TOAs outside coverage"
                        )
                    message += (
                        ". Continuing with PINT's nearest endpoint correction outside "
                        "the clock coverage, not a measured correction at those epochs. "
                        "Review results and update the clock file when available; "
                        "use --clock-dir DIRECTORY to select local clock files."
                    )
                elif "No data points in clock file" in message:
                    raise TimingResourceError(
                        message + ". An empty clock file cannot supply corrections. "
                        "Provide a nonempty file with --clock-dir DIRECTORY."
                    )
                warnings.warn(message, warning.category, stacklevel=2)
            if coverage_warnings:
                toas.apollo_clock_coverage_warnings = coverage_warnings
            return toas
        except TimingResourceError:
            raise
        except (ClockCorrectionError, OSError) as exc:
            raise TimingResourceError(
                f"Cannot load required clock/ephemeris resources for '{tim_path}' "
                f"(EPHEM {ephem}). Supply local files with --clock-dir DIRECTORY "
                "and --ephem-file FILE.bsp, or check the network/cache. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - exercised by integration/runtime failures
            override_hint = (
                f" using clock override directory '{clock_override_path}'"
                if clock_override_path is not None
                else ""
            )
            raise RuntimeError(
                f"PINT failed to load TOAs for '{tim_path}'{override_hint}. "
                "Check observatory clock correction resources and timing-model compatibility. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc


def maybe_generate_fake_toas(simulation: SimulationConfig) -> None:
    config = simulation.generate_fake_toas
    if config is None:
        return
    if simulation.truth_par_path is None:
        raise ValueError("PINT fake TOA generation requires truth_par_path.")

    config.tim_path.parent.mkdir(parents=True, exist_ok=True)
    model = _load_pint_model(simulation.truth_par_path)
    ephem = model["EPHEM"].value if "EPHEM" in model.params else "DE421"
    _prime_local_ephemeris(ephem)
    toas = make_fake_toas_uniform(
        config.start_mjd,
        config.end_mjd,
        config.ntoas,
        model=model,
        error=config.error_us * u.us,
        freq=config.freq_mhz * u.MHz,
        obs=config.obs,
        add_noise=config.add_noise,
        include_bipm=False,
    )
    toas.write_TOA_file(str(config.tim_path), name="simulated")


def _freeze_all_parameters(model) -> None:
    for name in model.params:
        parameter = model[name]
        if hasattr(parameter, "frozen"):
            parameter.frozen = True


def _set_numeric_model_parameter_value(model, parameter_name: str, value: float) -> None:
    name = str(parameter_name).upper()
    if name not in model.params:
        raise ValueError(f"PINT parameter '{name}' is missing from the timing model.")
    parameter = model[name]
    if name == "RAJ":
        parameter.quantity = Angle(float(value), unit=u.hourangle)
    elif name == "DECJ":
        parameter.quantity = Angle(float(value), unit=u.deg)
    elif getattr(parameter, "scale_factor", None) not in (None, 1, 1.0) and parameter.units is not None:
        parameter.quantity = float(value) * parameter.units
    else:
        parameter.value = float(value)


def _extract_parameters(model, names: set[str]) -> dict[str, float | str]:
    extracted: dict[str, float | str] = {}
    for name in sorted(names):
        if name not in model.params:
            continue
        value = model[name].value
        try:
            extracted[name] = float(value)
        except (TypeError, ValueError):
            extracted[name] = str(value)
    return extracted


def _extract_numeric_model_parameters(model, names: Sequence[str]) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for name in names:
        if name not in model.params:
            continue
        try:
            extracted[name] = float(model[name].value)
        except (TypeError, ValueError):
            continue
    return extracted


def _build_candidate_model(request: BootstrapRequest, candidate: Candidate, fit_params: list[str]) -> object:
    model = _load_pint_model(request.par_path)
    _freeze_all_parameters(model)
    for name, value in (candidate.fitted_parameters or {}).items():
        if name in model.params:
            _set_numeric_model_parameter_value(model, name, float(value))
    for name, value in candidate.hard_parameters.items():
        if name not in model.params:
            raise ValueError(f"PINT candidate parameter {name} is missing from the par file.")
        _set_numeric_model_parameter_value(model, name, float(value))
        model[name].frozen = True
    for name in fit_params:
        if name in model.params:
            model[name].frozen = False
    return model


def _build_recovered_model(request: BootstrapRequest, candidate: Candidate) -> object:
    model = _load_pint_model(request.par_path)
    _freeze_all_parameters(model)
    for name, value in candidate.hard_parameters.items():
        if name in model.params:
            _set_numeric_model_parameter_value(model, name, float(value))
    for name, value in (candidate.fitted_parameters or {}).items():
        if name in model.params:
            _set_numeric_model_parameter_value(model, name, float(value))
    return model


def _update_fit_metadata(model, toas, residuals: Residuals, chi2: float) -> None:
    metadata_values = {
        "START": toas.first_MJD,
        "FINISH": toas.last_MJD,
        "NTOA": len(toas),
        "EPHEM": toas.ephem,
        "DMDATA": hasattr(residuals, "dm"),
        "CLOCK": (
            f"TT({toas.clock_corr_info['bipm_version']})"
            if toas.clock_corr_info["include_bipm"]
            else "TT(TAI)"
        ),
        "CHI2": chi2,
        "CHI2R": chi2 / residuals.dof,
    }
    for name, value in metadata_values.items():
        if name in model.params:
            if name in {"CHI2", "CHI2R", "START", "FINISH", "NTOA"}:
                model[name].value = value
            else:
                model[name].value = value
    if "TRES" in model.params:
        model["TRES"].quantity = residuals.rms_weighted()


def _normalize_design_matrix(design_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalization = np.sqrt(np.sum(design_matrix**2, axis=0))
    normalization[normalization == 0.0] = 1.0
    return design_matrix / normalization, normalization


def _solve_weighted_least_squares(
    residual_seconds: np.ndarray,
    sigma_seconds: np.ndarray,
    design_matrix: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_residuals = residual_seconds / sigma_seconds
    weighted_design = design_matrix / sigma_seconds[:, None]
    normalized_design, normalization = _normalize_design_matrix(weighted_design)
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        normalized_design,
        full_matrices=False,
    )

    filtered_singular_values = singular_values.copy()
    if filtered_singular_values.size > 0 and filtered_singular_values[0] > 0.0:
        filtered_singular_values[filtered_singular_values <= threshold * filtered_singular_values[0]] = np.inf

    covariance_normalized = (right_vectors_t.T / (filtered_singular_values**2)) @ right_vectors_t
    covariance = (covariance_normalized / normalization).T / normalization
    parameter_updates = (
        right_vectors_t.T @ ((left_vectors.T @ weighted_residuals) / filtered_singular_values)
    ) / normalization
    return parameter_updates, covariance


def _solve_damped_weighted_least_squares(
    residual_seconds: np.ndarray,
    sigma_seconds: np.ndarray,
    design_matrix: np.ndarray,
    threshold: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_residuals = residual_seconds / sigma_seconds
    weighted_design = design_matrix / sigma_seconds[:, None]
    normalized_design, normalization = _normalize_design_matrix(weighted_design)
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        normalized_design,
        full_matrices=False,
    )

    filtered_singular_values = singular_values.copy()
    if filtered_singular_values.size > 0 and filtered_singular_values[0] > 0.0:
        floor = threshold * filtered_singular_values[0]
        filtered_singular_values[filtered_singular_values <= floor] = 0.0

    gains = np.zeros_like(filtered_singular_values)
    usable = filtered_singular_values > 0.0
    if np.any(usable):
        gains[usable] = filtered_singular_values[usable] / (
            filtered_singular_values[usable] ** 2 + max(damping, 0.0)
        )

    parameter_updates = (
        right_vectors_t.T @ ((left_vectors.T @ weighted_residuals) * gains)
    ) / normalization

    inverse_singular_squares = np.zeros_like(filtered_singular_values)
    if np.any(usable):
        inverse_singular_squares[usable] = 1.0 / (filtered_singular_values[usable] ** 2)
    covariance_normalized = (
        right_vectors_t.T * inverse_singular_squares
    ) @ right_vectors_t
    covariance = (covariance_normalized / normalization).T / normalization
    return parameter_updates, covariance


def _apply_linearized_updates(
    model,
    design_params: Sequence[str],
    design_units: Sequence[u.Unit],
    parameter_updates: np.ndarray,
    covariance: np.ndarray,
) -> None:
    uncertainty_updates: dict[str, u.Quantity] = {}
    for index, parameter_name in enumerate(design_params):
        if parameter_name == "Offset":
            continue
        update_unit = u.s / design_units[index]
        parameter = model[parameter_name]
        parameter.quantity = parameter.quantity + parameter_updates[index] * update_unit
        diagonal_entry = max(float(covariance[index, index]), 0.0)
        uncertainty_updates[parameter_name] = np.sqrt(diagonal_entry) * update_unit

    if uncertainty_updates:
        model.set_param_uncertainties(uncertainty_updates)


def _capture_parameter_state(model, design_params: Sequence[str]) -> dict[str, u.Quantity]:
    captured: dict[str, u.Quantity] = {}
    for parameter_name in design_params:
        if parameter_name == "Offset" or parameter_name not in model.params:
            continue
        captured[parameter_name] = model[parameter_name].quantity
    return captured


def _restore_parameter_state(model, captured_state: dict[str, u.Quantity]) -> None:
    for parameter_name, quantity in captured_state.items():
        if parameter_name in model.params:
            model[parameter_name].quantity = quantity


def _model_respects_basic_physical_bounds(model) -> bool:
    for parameter_name in ("F0", "A1", "PB"):
        if parameter_name in model.params:
            value = float(model[parameter_name].value)
            if not np.isfinite(value) or value <= 0.0:
                return False
    if "ECC" in model.params:
        eccentricity = float(model["ECC"].value)
        if not np.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0:
            return False
    return True


class PulseNumberAssignmentError(RuntimeError):
    """Raised when a fit cannot keep a well-defined integer pulse path."""


def _declared_track_mode(model) -> str | None:
    """Return an explicit PINT TRACK directive ('0' or '-2') when present."""

    try:
        if "TRACK" not in model.params:
            return None
        value = model["TRACK"].value
    except (AttributeError, KeyError, TypeError):
        return None
    if value is None:
        return None
    token = str(value).strip()
    return token if token in {"0", "-2"} else None


def _complete_pulse_numbers(toas) -> np.ndarray | None:
    """Return every integer pulse number in the TOA table, or None when incomplete."""

    try:
        if "pulse_number" not in toas.table.colnames:
            return None
        values = np.ma.asarray(toas.table["pulse_number"])
    except (AttributeError, KeyError, TypeError):
        return None
    if np.any(np.ma.getmaskarray(values)):
        return None
    try:
        array = np.asarray(values.data, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (len(toas.table),) or not np.all(np.isfinite(array)):
        return None
    return array


def _fit_residual_track_mode(toas, model) -> str:
    """Choose the PINT residual mode while preserving the integer pulse path.

    With the default ``track_mode``, each ``Residuals`` evaluation derives the
    TOA pulse numbers from the current model. An iterative fit can then change
    pulse numbers between steps and converge to a phase branch that differs
    from the saved path. Fix the pulse numbers and report any relabeling.
    """

    declared = _declared_track_mode(model)
    if declared == "0":
        return "nearest"
    if declared == "-2":
        return "use_pulse_numbers"
    if _complete_pulse_numbers(toas) is not None:
        return "use_pulse_numbers"
    return "nearest"


def pin_fit_pulse_numbers(toas, model) -> dict[str, object]:
    """Pin the integer pulse path once, before a fit iterates.

    Keep pulse numbers supplied in the TIM file. If the file has none,
    compute them from the input model before the first iteration so that PINT
    cannot change the integer path during the fit.
    """

    existing = _complete_pulse_numbers(toas)
    if existing is not None:
        return {
            "source": "tim_pulse_number_flags",
            "track_mode": _fit_residual_track_mode(toas, model),
            "toa_count": int(existing.size),
            "pinned": True,
            "derived_from_input_model": False,
        }
    if _declared_track_mode(model) == "0":
        return {
            "source": "model_TRACK_directive_0",
            "track_mode": "nearest",
            "toa_count": int(len(toas.table)),
            "pinned": False,
            "derived_from_input_model": False,
        }
    if not hasattr(toas, "compute_pulse_numbers"):
        raise PulseNumberAssignmentError(
            "The fit input carries no pulse numbers and this PINT version cannot "
            "derive an integer pulse path."
        )
    toas.compute_pulse_numbers(model)
    derived = _complete_pulse_numbers(toas)
    if derived is None:
        raise PulseNumberAssignmentError(
            "PINT did not produce a complete integer pulse path for the fit input."
        )
    return {
        "source": "computed_from_input_model",
        "track_mode": "use_pulse_numbers",
        "toa_count": int(derived.size),
        "pinned": True,
        "derived_from_input_model": True,
    }


def _relabeling_audit(toas, model) -> dict[str, object]:
    """Report TOAs whose integer cycle the fitted model would now relabel.

    Restore the pinned integers before returning; the audit leaves the
    input pulse path unchanged.
    """

    reference = _complete_pulse_numbers(toas)
    empty = {
        "relabeled_toa_count": 0,
        "relabeled_toa_indices_1based": [],
        "maximum_cycle_shift": 0.0,
    }
    if reference is None or not hasattr(toas, "compute_pulse_numbers"):
        return empty
    try:
        toas.compute_pulse_numbers(model)
        candidate = _complete_pulse_numbers(toas)
    finally:
        toas.table["pulse_number"] = np.array(reference, copy=True)
        toas.table["pulse_number"].unit = u.dimensionless_unscaled
    if candidate is None:
        return empty
    shifts = np.rint(candidate - reference)
    changed = np.flatnonzero(shifts != 0.0)
    if changed.size == 0:
        return empty
    return {
        "relabeled_toa_count": int(changed.size),
        "relabeled_toa_indices_1based": [int(index) + 1 for index in changed],
        "maximum_cycle_shift": float(np.max(np.abs(shifts))),
    }


def _anchored_branch_ambiguity(toas, model) -> dict[str, object]:
    """Report TOAs whose pinned pulse label is not the nearest one.

    PINT's ``subtract_mean`` removes a common phase offset when calculating residuals.
    If a residual relative to that anchor exceeds half a cycle, the input pulse
    label differs from the label preferred by the fitted model. The labels
    differ by a whole pulse, which changes the residual and its chi2
    contribution. Review these ambiguous TOAs before using them to constrain
    or report the timing solution.
    """

    empty = {
        "anchored_branch_ambiguous_toa_count": 0,
        "anchored_branch_ambiguous_toa_indices_1based": [],
        "maximum_anchored_phase_residual_cycles": 0.0,
    }
    if _fit_residual_track_mode(toas, model) != "use_pulse_numbers":
        return empty
    if not hasattr(toas, "get_pulse_numbers") or not hasattr(toas, "get_mjds"):
        return empty
    try:
        residuals = Residuals(toas, model, track_mode="use_pulse_numbers")
    except (AttributeError, KeyError, TypeError, ValueError):
        # If residuals cannot be computed for this diagnostic, leave the fit
        # result unchanged.
        return empty
    phase_residuals = np.asarray(residuals.phase_resids, dtype=float)
    ambiguous = np.flatnonzero(np.abs(phase_residuals) > 0.5)
    return {
        "anchored_branch_ambiguous_toa_count": int(ambiguous.size),
        "anchored_branch_ambiguous_toa_indices_1based": [
            int(index) + 1 for index in ambiguous
        ],
        "maximum_anchored_phase_residual_cycles": float(
            np.max(np.abs(phase_residuals))
        ),
    }


def audit_fit_pulse_number_relabeling(toas, model) -> dict[str, object]:
    """Audit the pulse path a fit used: absolute drift and anchored ambiguity."""

    audit = _relabeling_audit(toas, model)
    audit.update(_anchored_branch_ambiguity(toas, model))
    return audit


def _finalize_fit_diagnostics(
    toas,
    model,
    threshold: float | None,
    fit_diagnostics: dict[str, object],
) -> tuple[Residuals, float, float, dict[str, object]]:
    residuals = Residuals(toas, model, track_mode=_fit_residual_track_mode(toas, model))
    residual_seconds = residuals.time_resids.to(u.s).value
    sigma_seconds = model.scaled_toa_uncertainty(toas).to(u.s).value
    design_matrix, design_params, _ = model.designmatrix(
        toas=toas,
        incfrozen=False,
        incoffset=True,
    )
    svd_threshold = threshold if threshold is not None else 1e-14 * max(design_matrix.shape)
    _, covariance = _solve_weighted_least_squares(
        residual_seconds=residual_seconds,
        sigma_seconds=sigma_seconds,
        design_matrix=design_matrix,
        threshold=svd_threshold,
    )
    chi2 = float(residuals.chi2)
    reduced_chi2 = float(residuals.reduced_chi2)
    _update_fit_metadata(model, toas, residuals, chi2)
    fit_diagnostics["covariance_matrix"] = covariance
    fit_diagnostics["design_parameter_names"] = list(design_params)
    fit_diagnostics["weighted_rms_us"] = float(residuals.rms_weighted().to(u.us).value)
    fit_diagnostics["pulse_number_audit"] = audit_fit_pulse_number_relabeling(
        toas,
        model,
    )
    return residuals, chi2, reduced_chi2, fit_diagnostics


def _run_linearized_timing_fit(
    toas,
    model,
    maxiter: int = 5,
    threshold: float | None = None,
) -> tuple[object, float, float, dict[str, object]]:
    model.validate()
    model.validate_toas(toas)
    fit_diagnostics: dict[str, object] = {
        "covariance_matrix": None,
        "design_parameter_names": None,
        "weighted_rms_us": None,
        "pulse_numbers": pin_fit_pulse_numbers(toas, model),
        "local_solver_diagnostics": {
            "mode": "linearized",
            "iterations": 0,
            "accepted_steps": maxiter,
            "rejected_steps": 0,
            "final_damping": 0.0,
        },
    }
    if not model.free_params:
        residuals = Residuals(toas, model, track_mode=_fit_residual_track_mode(toas, model))
        chi2 = float(residuals.chi2)
        reduced_chi2 = float(residuals.reduced_chi2)
        _update_fit_metadata(model, toas, residuals, chi2)
        fit_diagnostics["weighted_rms_us"] = float(residuals.rms_weighted().to(u.us).value)
        return model, chi2, reduced_chi2, fit_diagnostics

    for _ in range(maxiter):
        residuals = Residuals(toas, model, track_mode=_fit_residual_track_mode(toas, model))
        residual_seconds = residuals.time_resids.to(u.s).value
        sigma_seconds = model.scaled_toa_uncertainty(toas).to(u.s).value
        design_matrix, design_params, design_units = model.designmatrix(
            toas=toas,
            incfrozen=False,
            incoffset=True,
        )
        svd_threshold = threshold if threshold is not None else 1e-14 * max(design_matrix.shape)
        parameter_updates, covariance = _solve_weighted_least_squares(
            residual_seconds=residual_seconds,
            sigma_seconds=sigma_seconds,
            design_matrix=design_matrix,
            threshold=svd_threshold,
        )
        _apply_linearized_updates(
            model,
            design_params=design_params,
            design_units=design_units,
            parameter_updates=parameter_updates,
            covariance=covariance,
        )
    fit_diagnostics["local_solver_diagnostics"]["iterations"] = maxiter
    _, chi2, reduced_chi2, fit_diagnostics = _finalize_fit_diagnostics(
        toas=toas,
        model=model,
        threshold=threshold,
        fit_diagnostics=fit_diagnostics,
    )
    return model, chi2, reduced_chi2, fit_diagnostics


def _run_low_level_timing_fit(
    toas,
    model,
    maxiter: int = 8,
    threshold: float | None = None,
    mode: str = "robust-lm",
    initial_damping: float = 1.0,
    damping_multiplier: float = 8.0,
    min_fractional_improvement: float = 1.0e-6,
) -> tuple[object, float, float, dict[str, object]]:
    if mode == "linearized":
        return _run_linearized_timing_fit(
            toas=toas,
            model=model,
            maxiter=maxiter,
            threshold=threshold,
        )

    model.validate()
    model.validate_toas(toas)
    fit_diagnostics: dict[str, object] = {
        "covariance_matrix": None,
        "design_parameter_names": None,
        "weighted_rms_us": None,
        "pulse_numbers": pin_fit_pulse_numbers(toas, model),
        "local_solver_diagnostics": {
            "mode": mode,
            "iterations": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "final_damping": max(initial_damping, 1.0e-12),
        },
    }
    if not model.free_params:
        residuals = Residuals(toas, model, track_mode=_fit_residual_track_mode(toas, model))
        chi2 = float(residuals.chi2)
        reduced_chi2 = float(residuals.reduced_chi2)
        _update_fit_metadata(model, toas, residuals, chi2)
        fit_diagnostics["weighted_rms_us"] = float(residuals.rms_weighted().to(u.us).value)
        return model, chi2, reduced_chi2, fit_diagnostics

    damping = max(initial_damping, 1.0e-12)
    rejection_cap = 8
    for iteration in range(maxiter):
        residuals = Residuals(toas, model, track_mode=_fit_residual_track_mode(toas, model))
        current_chi2 = float(residuals.chi2)
        residual_seconds = residuals.time_resids.to(u.s).value
        sigma_seconds = model.scaled_toa_uncertainty(toas).to(u.s).value
        design_matrix, design_params, design_units = model.designmatrix(
            toas=toas,
            incfrozen=False,
            incoffset=True,
        )
        svd_threshold = threshold if threshold is not None else 1e-14 * max(design_matrix.shape)
        baseline_state = _capture_parameter_state(model, design_params)
        accepted = False
        trial_damping = damping
        for _ in range(rejection_cap):
            parameter_updates, covariance = _solve_damped_weighted_least_squares(
                residual_seconds=residual_seconds,
                sigma_seconds=sigma_seconds,
                design_matrix=design_matrix,
                threshold=svd_threshold,
                damping=trial_damping,
            )
            _apply_linearized_updates(
                model,
                design_params=design_params,
                design_units=design_units,
                parameter_updates=parameter_updates,
                covariance=covariance,
            )
            try:
                if not _model_respects_basic_physical_bounds(model):
                    raise ValueError("Trial timing model is outside its physical domain.")
                trial_residuals = Residuals(
                    toas,
                    model,
                    track_mode=_fit_residual_track_mode(toas, model),
                )
                trial_chi2 = float(trial_residuals.chi2)
            except Exception:
                _restore_parameter_state(model, baseline_state)
                fit_diagnostics["local_solver_diagnostics"]["rejected_steps"] += 1
                trial_damping *= max(damping_multiplier, 1.0000001)
                continue
            improvement = current_chi2 - trial_chi2
            target_improvement = max(abs(current_chi2) * min_fractional_improvement, 1.0e-12)
            if np.isfinite(trial_chi2) and improvement > 0.0:
                accepted = True
                fit_diagnostics["local_solver_diagnostics"]["accepted_steps"] += 1
                damping = max(trial_damping / max(damping_multiplier, 1.0000001), 1.0e-12)
                if improvement <= target_improvement:
                    fit_diagnostics["local_solver_diagnostics"]["iterations"] = iteration + 1
                    fit_diagnostics["local_solver_diagnostics"]["final_damping"] = damping
                    _, chi2, reduced_chi2, fit_diagnostics = _finalize_fit_diagnostics(
                        toas=toas,
                        model=model,
                        threshold=threshold,
                        fit_diagnostics=fit_diagnostics,
                    )
                    return model, chi2, reduced_chi2, fit_diagnostics
                break
            _restore_parameter_state(model, baseline_state)
            fit_diagnostics["local_solver_diagnostics"]["rejected_steps"] += 1
            trial_damping *= max(damping_multiplier, 1.0000001)

        fit_diagnostics["local_solver_diagnostics"]["iterations"] = iteration + 1
        fit_diagnostics["local_solver_diagnostics"]["final_damping"] = (
            damping if accepted else trial_damping
        )
        if not accepted:
            break

    _, chi2, reduced_chi2, fit_diagnostics = _finalize_fit_diagnostics(
        toas=toas,
        model=model,
        threshold=threshold,
        fit_diagnostics=fit_diagnostics,
    )
    return model, chi2, reduced_chi2, fit_diagnostics


def _run_fit(
    toas,
    model,
    fit_params: list[str],
    local_solver_mode: str = "robust-lm",
    local_solver_max_iterations: int = 8,
    local_solver_initial_damping: float = 1.0,
    local_solver_damping_multiplier: float = 8.0,
    local_solver_min_fractional_improvement: float = 1.0e-6,
) -> tuple[object, float, float, dict[str, object]]:
    if not fit_params:
        residuals = Residuals(
            toas,
            model,
            use_abs_phase=False,
            track_mode=_fit_residual_track_mode(toas, model),
        )
        chi2 = float(residuals.chi2)
        reduced_chi2 = float(residuals.reduced_chi2)
        return model, chi2, reduced_chi2, {
            "covariance_matrix": None,
            "design_parameter_names": None,
            "weighted_rms_us": float(residuals.rms_weighted().to(u.us).value),
            "local_solver_diagnostics": None,
        }

    return _run_low_level_timing_fit(
        toas,
        model,
        maxiter=local_solver_max_iterations,
        mode=local_solver_mode,
        initial_damping=local_solver_initial_damping,
        damping_multiplier=local_solver_damping_multiplier,
        min_fractional_improvement=local_solver_min_fractional_improvement,
    )


def _evaluate_residual_stats(toas, model) -> tuple[float, float, float]:
    residuals = Residuals(
        toas,
        model,
        use_abs_phase=False,
        track_mode=_fit_residual_track_mode(toas, model),
    )
    return (
        float(residuals.chi2),
        float(residuals.reduced_chi2),
        float(residuals.rms_weighted().to(u.us).value),
    )


def _matches_reference_quality(
    quality_ratio_to_reference: float | None,
    reference_weighted_rms_us: float,
    recovered_weighted_rms_us: float,
) -> bool:
    near_rms_threshold_us = max(0.05, reference_weighted_rms_us * 10.0)

    if recovered_weighted_rms_us <= near_rms_threshold_us:
        return True
    if quality_ratio_to_reference is not None and quality_ratio_to_reference <= 10.0:
        return True
    return False


def _parameter_gap_is_small(parameter_deltas: dict[str, float] | None) -> bool:
    if not parameter_deltas:
        return False
    thresholds = {
        "F0": 1.0e-5,
        "F1": 1.0e-12,
        "PB": 1.0e-3,
        "A1": 1.0e-3,
        "TASC": 1.0e-2,
        "EPS1": 1.0e-3,
        "EPS2": 1.0e-3,
        "DM": 1.0,
    }
    for name, delta in parameter_deltas.items():
        threshold = thresholds.get(name, 1.0e-3)
        if abs(delta) > threshold:
            return False
    return True


def classify_likely_error_points(
    stage_results,
    failures: list[FailureState] | None,
    quality_ratio_to_reference: float | None,
    hard_parameter_deltas: dict[str, float] | None,
    fitted_parameter_deltas: dict[str, float] | None,
) -> list[LikelyErrorPoint]:
    likely: list[LikelyErrorPoint] = []
    failure_codes = {failure.code for failure in failures or []}

    if "local-fit-quality-gate" in failure_codes or "local-fit-residual-gate" in failure_codes:
        if _parameter_gap_is_small(hard_parameter_deltas or fitted_parameter_deltas):
            likely.append(
                LikelyErrorPoint(
                    code="full-solution-fit-dimensionality-too-large",
                    message="The official full-solution fit likely keeps too many strongly coupled parameters free at once.",
                    evidence="Parameter deltas are already small, but the official full fit still violates the quality gates by a large margin.",
                )
            )
        else:
            likely.append(
                LikelyErrorPoint(
                    code="remaining-basin-or-model-mismatch",
                    message="The official fit likely still starts in the wrong parameter region, or a model mismatch remains.",
                    evidence="Full-solution quality gates failed before the run reached reference-quality residuals.",
                )
            )

    if "basin-entry-not-established" in failure_codes:
        likely.append(
            LikelyErrorPoint(
                code="precondition-basin-not-reliable",
                message="The search preceding the official fit likely fails to identify a reliable basin for local fitting.",
                evidence="The basin-entry check skipped a local-fit stage before the official fit sequence completed.",
            )
        )

    max_condition_number = None
    for stage in stage_results or []:
        if stage.posterior_summary and stage.posterior_summary.condition_number is not None:
            max_condition_number = max(
                max_condition_number or stage.posterior_summary.condition_number,
                stage.posterior_summary.condition_number,
            )
    if max_condition_number is not None and max_condition_number > 1.0e20:
        likely.append(
            LikelyErrorPoint(
                code="strong-parameter-degeneracy",
                message="Strong parameter degeneracy likely prevents stable full-solution convergence.",
                evidence=f"Observed posterior condition number reached {max_condition_number}.",
            )
        )

    if "posterior-guidance-locked" in failure_codes:
        likely.append(
            LikelyErrorPoint(
                code="official-fit-schedule-misaligned",
                message="The official fit schedule likely requests posterior-guided refinement before a reliable local posterior is available.",
                evidence="Posterior-guided stages remained locked while the debug run still used deterministic fallback.",
            )
        )

    if quality_ratio_to_reference is not None and quality_ratio_to_reference > 1000.0:
        likely.append(
            LikelyErrorPoint(
                code="residual-gap-still-catastrophic",
                message="The candidate is numerically close to the reference, but its residuals are still far from the reference solution.",
                evidence=f"Reference quality ratio is still {quality_ratio_to_reference}.",
            )
        )

    return likely


def _summarize_covariance(
    covariance_matrix: np.ndarray | None,
    design_parameter_names: Sequence[str] | None,
    stage_id: str,
) -> PosteriorSummary:
    if covariance_matrix is None or not design_parameter_names:
        return PosteriorSummary(
            status="unavailable",
            source_stage=stage_id,
            guidance_strategy="deterministic-fallback",
            message="Local fit did not produce a usable covariance matrix.",
        )

    active_indices = [
        index
        for index, name in enumerate(design_parameter_names)
        if name != "Offset"
    ]
    if not active_indices:
        return PosteriorSummary(
            status="unavailable",
            source_stage=stage_id,
            guidance_strategy="deterministic-fallback",
            message="Covariance matrix contained no free timing parameters.",
        )

    reduced_covariance = np.asarray(covariance_matrix)[np.ix_(active_indices, active_indices)]
    parameter_names = [design_parameter_names[index] for index in active_indices]
    diagonal = np.diag(reduced_covariance)
    if diagonal.size == 0 or not np.all(np.isfinite(diagonal)):
        return PosteriorSummary(
            status="ill-conditioned",
            source_stage=stage_id,
            guidance_strategy="deterministic-fallback",
            message="Covariance diagonal is not finite.",
        )

    safe_diagonal = np.maximum(diagonal, 0.0)
    sigmas = np.sqrt(safe_diagonal)
    condition_number = float(np.linalg.cond(reduced_covariance)) if reduced_covariance.size else None
    if (
        condition_number is None
        or not np.isfinite(condition_number)
        or np.any(sigmas <= 0.0)
    ):
        return PosteriorSummary(
            status="ill-conditioned",
            source_stage=stage_id,
            guidance_strategy="deterministic-fallback",
            condition_number=condition_number,
            message="Covariance estimate is too poorly conditioned for posterior-guided adaptation.",
        )

    denominator = np.outer(sigmas, sigmas)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_matrix = np.divide(
            reduced_covariance,
            denominator,
            out=np.zeros_like(reduced_covariance),
            where=denominator > 0.0,
        )
    parameter_sigmas = {
        name: float(value)
        for name, value in zip(parameter_names, sigmas, strict=True)
    }
    parameter_correlations: dict[str, float] = {}
    recommended_joint_parameters: list[list[str]] = []
    for left_index, left_name in enumerate(parameter_names):
        for right_index in range(left_index + 1, len(parameter_names)):
            right_name = parameter_names[right_index]
            correlation = float(correlation_matrix[left_index, right_index])
            parameter_correlations[f"{left_name}:{right_name}"] = correlation
            if abs(correlation) >= 0.7:
                recommended_joint_parameters.append([left_name, right_name])

    return PosteriorSummary(
        status="usable",
        source_stage=stage_id,
        guidance_strategy="posterior-summary",
        parameter_sigmas=parameter_sigmas,
        parameter_correlations=parameter_correlations or None,
        condition_number=condition_number,
        recommended_joint_parameters=recommended_joint_parameters or None,
        message=(
            "The covariance has a large but finite condition number; correlation-based guidance "
            "is informative but not definitive."
            if condition_number is not None and condition_number > 1.0e12
            else None
        ),
    )


def refine_candidate_with_pint(
    request: BootstrapRequest,
    candidate: Candidate,
    fit_params: list[str],
    stage_id: str,
    quality_gate_max_reduced_chi2: float | None = None,
    quality_gate_max_weighted_rms_us: float | None = None,
    local_solver_mode: str = "robust-lm",
    local_solver_max_iterations: int = 8,
    local_solver_initial_damping: float = 1.0,
    local_solver_damping_multiplier: float = 8.0,
    local_solver_min_fractional_improvement: float = 1.0e-6,
) -> tuple[Candidate, PosteriorSummary, list[FailureState], dict[str, float]]:
    seed_model = _load_pint_model(request.par_path)
    toas = _load_pint_toas(
        request.tim_path,
        seed_model=seed_model,
        clock_override_path=(
            request.simulation.reference_target.pint_clock_override_path
            if request.simulation.reference_target is not None
            else None
        ),
    )
    model = _build_candidate_model(request, candidate, [])
    for name in fit_params:
        if name in model.params:
            model[name].frozen = False
    fitted_model, chi2, reduced_chi2, fit_diagnostics = _run_fit(
        toas,
        model,
        fit_params,
        local_solver_mode=local_solver_mode,
        local_solver_max_iterations=local_solver_max_iterations,
        local_solver_initial_damping=local_solver_initial_damping,
        local_solver_damping_multiplier=local_solver_damping_multiplier,
        local_solver_min_fractional_improvement=local_solver_min_fractional_improvement,
    )

    updated_candidate = Candidate(
        candidate_id=candidate.candidate_id,
        hard_parameters=dict(candidate.hard_parameters),
        approximate_score=candidate.approximate_score,
        exact_score=-chi2,
        fitted_parameters=_extract_parameters(
            fitted_model,
            set(candidate.hard_parameters) | set(fit_params),
        ),
        fit_chi2=chi2,
        fit_reduced_chi2=reduced_chi2,
    )
    for name in candidate.hard_parameters:
        if updated_candidate.fitted_parameters and name in updated_candidate.fitted_parameters:
            updated_candidate.hard_parameters[name] = float(updated_candidate.fitted_parameters[name])

    posterior_summary = _summarize_covariance(
        fit_diagnostics.get("covariance_matrix"),
        fit_diagnostics.get("design_parameter_names"),
        stage_id=stage_id,
    )
    failures: list[FailureState] = []
    weighted_rms_us = float(fit_diagnostics.get("weighted_rms_us") or 0.0)
    if (
        quality_gate_max_reduced_chi2 is not None
        and reduced_chi2 > quality_gate_max_reduced_chi2
    ):
        failures.append(
            FailureState(
                stage=stage_id,
                code="local-fit-quality-gate",
                message=(
                    "Local low-level timing fit exceeded the reduced-chi2 quality gate: "
                    f"{reduced_chi2} > {quality_gate_max_reduced_chi2}"
                ),
                suggestion="Narrow the basin identified by coarse search, or reduce the local-fit parameter set.",
            )
        )
    if (
        quality_gate_max_weighted_rms_us is not None
        and weighted_rms_us > quality_gate_max_weighted_rms_us
    ):
        failures.append(
            FailureState(
                stage=stage_id,
                code="local-fit-residual-gate",
                message=(
                    "Local low-level timing fit exceeded the weighted-RMS quality gate: "
                    f"{weighted_rms_us} us > {quality_gate_max_weighted_rms_us} us"
                ),
                suggestion="Broaden the preceding search stage or revisit clock/model normalization inputs.",
            )
        )
    if posterior_summary.status != "usable":
        failures.append(
            FailureState(
                stage=stage_id,
                code="posterior-unavailable",
                message=posterior_summary.message or "Posterior guidance was unavailable.",
                suggestion="Continue with deterministic staged recovery or reduce the free-parameter set.",
            )
        )

    return updated_candidate, posterior_summary, failures, {
        "fit_chi2": chi2,
        "fit_reduced_chi2": reduced_chi2,
        "weighted_rms_us": weighted_rms_us,
        "local_solver_diagnostics": fit_diagnostics.get("local_solver_diagnostics"),
    }


def build_reference_calibration(
    request: BootstrapRequest,
    official_candidate: Candidate | None,
    debug_candidate: Candidate | None = None,
    stage_results=None,
    failures: list[FailureState] | None = None,
) -> CalibrationResult | None:
    reference_target = request.simulation.reference_target
    if reference_target is None:
        return None

    seed_model = _load_pint_model(request.par_path)
    toas = _load_pint_toas(
        request.tim_path,
        seed_model=seed_model,
        clock_override_path=reference_target.pint_clock_override_path,
    )
    reference_model = _load_pint_model(reference_target.par_path)
    reference_fit_chi2, reference_fit_reduced_chi2, reference_weighted_rms_us = _evaluate_residual_stats(
        toas,
        reference_model,
    )
    evaluation_candidate = official_candidate or debug_candidate
    if evaluation_candidate is None:
        return CalibrationResult(
            status="failure",
            evaluation_mode="unavailable",
            debug_candidate_id=None,
            reference_par_path=reference_target.par_path,
            reference_fit_chi2=reference_fit_chi2,
            reference_fit_reduced_chi2=reference_fit_reduced_chi2,
            reference_weighted_rms_us=reference_weighted_rms_us,
        )
    recovered_model = _build_recovered_model(request, evaluation_candidate)
    recovered_fit_chi2, recovered_fit_reduced_chi2, recovered_weighted_rms_us = _evaluate_residual_stats(
        toas,
        recovered_model,
    )
    quality_ratio_to_reference = (
        recovered_fit_reduced_chi2 / reference_fit_reduced_chi2
        if reference_fit_reduced_chi2 > 0.0
        else None
    )

    hard_parameter_deltas: dict[str, float] = {}
    for name, reference_value in _extract_numeric_model_parameters(
        reference_model,
        list(evaluation_candidate.hard_parameters),
    ).items():
        if name in evaluation_candidate.hard_parameters:
            hard_parameter_deltas[name] = evaluation_candidate.hard_parameters[name] - reference_value

    fitted_parameter_deltas: dict[str, float] = {}
    for name, reference_value in _extract_numeric_model_parameters(
        reference_model,
        list((evaluation_candidate.fitted_parameters or {}).keys()),
    ).items():
        if evaluation_candidate.fitted_parameters and name in evaluation_candidate.fitted_parameters:
            fitted_parameter_deltas[name] = float(evaluation_candidate.fitted_parameters[name]) - reference_value

    return CalibrationResult(
        status=(
            "success"
            if official_candidate is not None
            and _matches_reference_quality(
                quality_ratio_to_reference,
                reference_weighted_rms_us,
                recovered_weighted_rms_us,
            )
            else "failure"
        ),
        evaluation_mode="official" if official_candidate is not None else "debug",
        debug_candidate_id=debug_candidate.candidate_id if debug_candidate is not None else None,
        reference_par_path=reference_target.par_path,
        reference_fit_chi2=reference_fit_chi2,
        reference_fit_reduced_chi2=reference_fit_reduced_chi2,
        reference_weighted_rms_us=reference_weighted_rms_us,
        recovered_fit_chi2=recovered_fit_chi2,
        recovered_fit_reduced_chi2=recovered_fit_reduced_chi2,
        recovered_weighted_rms_us=recovered_weighted_rms_us,
        quality_ratio_to_reference=quality_ratio_to_reference,
        reduced_chi2_gap_to_reference=(
            recovered_fit_reduced_chi2 - reference_fit_reduced_chi2
            if recovered_fit_reduced_chi2 is not None
            else None
        ),
        weighted_rms_gap_us_to_reference=(
            recovered_weighted_rms_us - reference_weighted_rms_us
            if recovered_weighted_rms_us is not None
            else None
        ),
        hard_parameter_deltas=hard_parameter_deltas or None,
        fitted_parameter_deltas=fitted_parameter_deltas or None,
        likely_error_points=classify_likely_error_points(
            stage_results=stage_results,
            failures=failures,
            quality_ratio_to_reference=quality_ratio_to_reference,
            hard_parameter_deltas=hard_parameter_deltas or None,
            fitted_parameter_deltas=fitted_parameter_deltas or None,
        )
        or None,
    )


def certify_with_pint(
    request: BootstrapRequest,
    candidates: list[Candidate],
) -> CertificationResult:
    if not candidates:
        raise ValueError("PINT certification requires at least one candidate.")

    metrics: dict[str, float] = {}
    seed_model = _load_pint_model(request.par_path)
    load_start = time.perf_counter()
    toas = _load_pint_toas(
        request.tim_path,
        seed_model=seed_model,
        clock_override_path=(
            request.simulation.reference_target.pint_clock_override_path
            if request.simulation.reference_target is not None
            else None
        ),
    )
    metrics["pint_toa_load_wall_time_seconds"] = time.perf_counter() - load_start
    certification_fit_params = request.simulation.certification_fit_params or []
    certified: list[Candidate] = []

    candidate_eval_start = time.perf_counter()
    for candidate in candidates:
        model = _build_candidate_model(request, candidate, certification_fit_params)
        fitted_model, chi2, reduced_chi2, _ = _run_fit(toas, model, certification_fit_params)
        candidate.exact_score = -chi2
        candidate.fit_chi2 = chi2
        candidate.fit_reduced_chi2 = reduced_chi2
        candidate.fitted_parameters = _extract_parameters(
            fitted_model,
            set(candidate.hard_parameters) | set(certification_fit_params),
        )
        certified.append(candidate)
    metrics["pint_candidate_evaluation_wall_time_seconds"] = (
        time.perf_counter() - candidate_eval_start
    )

    best_candidate = max(
        certified,
        key=lambda item: item.exact_score if item.exact_score is not None else float("-inf"),
    )

    post_fit_params = request.simulation.post_certification_fit_params or []
    if post_fit_params:
        post_fit_start = time.perf_counter()
        post_model = _build_candidate_model(request, best_candidate, [])
        for name in post_fit_params:
            if name in post_model.params:
                post_model[name].frozen = False
        fitted_model, chi2, reduced_chi2, _ = _run_fit(toas, post_model, post_fit_params)
        best_candidate.fit_chi2 = chi2
        best_candidate.fit_reduced_chi2 = reduced_chi2
        best_candidate.fitted_parameters = _extract_parameters(
            fitted_model,
            set(best_candidate.hard_parameters) | set(post_fit_params),
        )
        metrics["pint_postfit_wall_time_seconds"] = time.perf_counter() - post_fit_start
    else:
        metrics["pint_postfit_wall_time_seconds"] = 0.0

    metrics["certification_wall_time_seconds"] = sum(metrics.values())
    return CertificationResult(candidates=certified, best_candidate=best_candidate, metrics=metrics)
