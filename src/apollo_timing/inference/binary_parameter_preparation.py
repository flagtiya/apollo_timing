"""Restore the input parameter contract after a family-specific seed proposal."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import math

from pint.models import get_model

from apollo_timing.backend.pint_backend import (
    _load_pint_model,
    _set_numeric_model_parameter_value,
)
from apollo_timing.inference.bayesian_coarse_binary_bootstrap import (
    inspect_coarse_binary_par,
)
from apollo_timing.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from apollo_timing.io.par_compatibility import normalize_par_text
from apollo_timing.io.timing_input import inspect_timing_system


def prepare_binary_seed(
    *, source_par: Path, proposal_par: Path, output_path: Path
) -> Path:
    """Fill only missing core values; preserve every supplied fixed parameter.

    Bootstrap scores describe unconstrained proposals, not certification under
    this restored contract. The common solver must search and certify each seed.
    """

    spec = inspect_coarse_binary_par(source_par)
    proposal = _load_pint_model(proposal_par)
    text = normalize_par_text(Path(source_par).read_text(encoding="utf-8")).text
    for name in spec.missing_core_parameters:
        text += f"\n{name} {float(proposal[name].value):.18g} 1\n"
    model = get_model(StringIO(text), allow_tcb=True)
    for name in model.free_params:
        if name in proposal.params and proposal[name].value is not None:
            value = float(proposal[name].value)
        elif name == "FB0" and "PB" in proposal.params:
            value = 1.0 / (float(proposal["PB"].value) * 86400.0)
        else:
            continue
        _set_numeric_model_parameter_value(model, name, value)
    return write_pint_model_parfile_roundtrip_safe(model=model, path=output_path)


def prepare_discovered_binary_seed(
    *, source_par: Path, proposal_par: Path, output_path: Path,
    parameter_values: dict[str, float] | None = None,
) -> Path:
    """Apply an orbit proposal without overwriting the original spin contract.

    A declared target model is retained. DD/ELL1 conversion here supplies only
    initial values, never a claim that the two timing approximations are equal.
    """
    declaration = inspect_timing_system(source_par)
    proposal = _load_pint_model(proposal_par)
    target = declaration.binary_model or str(proposal["BINARY"].value)
    text = normalize_par_text(Path(source_par).read_text(encoding="utf-8")).text
    keys = {line.split()[0].upper() for line in text.splitlines() if line.strip()}
    if not declaration.binary_model:
        text += f"\nBINARY {target}\n"
    if "PB" not in keys and "FB0" not in keys:
        if any(key.startswith("FB") and key[2:].isdigit() for key in keys):
            raise ValueError("Frequency-derivative orbital inputs require FB0 before discovery.")
        text += f"\nPB {float(proposal['PB'].value):.18g} 1\n"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_with_period = output_path.with_name(output_path.stem + ".contract.par")
    source_with_period.write_text(text, encoding="utf-8")
    spec = inspect_coarse_binary_par(source_with_period)
    proposal_names = set(proposal.params)
    if any(name not in proposal_names or proposal[name].value is None for name in spec.missing_core_parameters):
        values = dict(parameter_values or {})
        if not values:
            if "ECC" in proposal.params and proposal["ECC"].value is not None:
                values = {name: float(proposal[name].value) for name in ("PB", "A1", "ECC", "OM", "T0")}
            else:
                eps1, eps2 = float(proposal["EPS1"].value), float(proposal["EPS2"].value)
                omega = math.atan2(eps1, eps2)
                pb = float(proposal["PB"].value)
                values = dict(PB=pb, A1=float(proposal["A1"].value), ECC=math.hypot(eps1, eps2),
                              OM=math.degrees(omega), T0=float(proposal["TASC"].value) + omega * pb / (2 * math.pi))
        omega = math.radians(values["OM"])
        values.update(EPS1=values["ECC"] * math.sin(omega), EPS2=values["ECC"] * math.cos(omega),
                      TASC=values["T0"] - omega * values["PB"] / (2 * math.pi))
        complete = text
        for name in spec.missing_core_parameters:
            complete += f"\n{name} {values[name]:.18g} 1\n"
        converted = output_path.with_name(output_path.stem + ".proposal.par")
        converted.write_text(complete, encoding="utf-8")
        proposal_par = converted
    return prepare_binary_seed(source_par=source_with_period, proposal_par=proposal_par, output_path=output_path)
