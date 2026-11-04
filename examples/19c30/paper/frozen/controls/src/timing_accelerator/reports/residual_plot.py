from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from pint.residuals import Residuals

from timing_accelerator.backend.pint_backend import _load_pint_model, _load_pint_toas


def _write_residual_csv(path: Path, mjd: np.ndarray, residual_us: np.ndarray, error_us: np.ndarray) -> None:
    rows = ["mjd,residual_us,error_us"]
    rows.extend(
        f"{float(x):.12f},{float(y):.12g},{float(err):.12g}"
        for x, y, err in zip(mjd, residual_us, error_us, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_residual_series_plot(
    plot_path: Path,
    *,
    mjd: np.ndarray,
    residual_us: np.ndarray,
    error_us: np.ndarray,
    spin_frequency_hz: float,
    pulsar_name: str,
    weighted_rms_us: float,
    reduced_chi2: float,
    inlier_mask: np.ndarray | None = None,
    weighted_rms_label: str = "WRMS",
    reduced_chi2_label: str = "Reduced chi2",
) -> Path:
    """Render the standard APOLLO timing-residual product from aligned arrays."""

    epochs = np.asarray(mjd, dtype=float)
    values = np.asarray(residual_us, dtype=float)
    uncertainties = np.asarray(error_us, dtype=float)
    if (
        epochs.ndim != 1
        or epochs.size == 0
        or values.shape != epochs.shape
        or uncertainties.shape != epochs.shape
    ):
        raise ValueError("Residual epochs, values, and errors must be aligned vectors.")
    if (
        not np.all(np.isfinite(epochs))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(uncertainties))
        or np.any(uncertainties <= 0.0)
    ):
        raise ValueError("Residual series must be finite with positive errors.")
    frequency = float(spin_frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("The residual plot needs a positive spin frequency.")
    retained = (
        np.ones(epochs.size, dtype=bool)
        if inlier_mask is None
        else np.asarray(inlier_mask, dtype=bool)
    )
    if retained.shape != epochs.shape:
        raise ValueError("The residual inlier mask must align with every TOA.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)

    def plot_subset(mask: np.ndarray, *, color: str, label: str | None) -> None:
        if not np.any(mask):
            return
        axis.errorbar(
            epochs[mask],
            values[mask],
            yerr=uncertainties[mask],
            fmt="o",
            markersize=4.5,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            alpha=0.9,
            label=label,
        )

    has_marginalized = bool(np.any(~retained))
    plot_subset(
        retained,
        color="#1f77b4",
        label="Retained TOAs" if has_marginalized else None,
    )
    plot_subset(
        ~retained,
        color="#c43c35",
        label="Marginalized session",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("MJD")
    axis.set_ylabel("Timing residual (us)")
    axis.set_title(f"PSR {str(pulsar_name).strip()} Timing Residuals")
    axis.grid(axis="y", alpha=0.3)
    if has_marginalized:
        axis.legend(loc="upper left", frameon=True)

    phase_axis = axis.secondary_yaxis(
        "right",
        functions=(
            lambda value: np.asarray(value) * 1.0e-6 * frequency,
            lambda value: np.asarray(value) * 1.0e6 / frequency,
        ),
    )
    phase_axis.set_ylabel("Phase residual (cycles)")
    axis.text(
        0.985,
        0.975,
        (
            f"{weighted_rms_label} = {float(weighted_rms_us):.3f} us\n"
            f"{reduced_chi2_label} = {float(reduced_chi2):.3f}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return plot_path


def write_residual_plot(
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    *,
    clock_override_path: Path | None = None,
    artifact_stem: str = "best",
) -> tuple[Path, Path]:
    par_path = Path(par_path).resolve()
    tim_path = Path(tim_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = str(artifact_stem).strip() or "best"
    csv_path = output_dir / f"{artifact_stem}_residuals.csv"
    plot_path = output_dir / f"{artifact_stem}_residuals.png"
    model = _load_pint_model(par_path)
    toas = _load_pint_toas(
        tim_path,
        seed_model=model,
        clock_override_path=clock_override_path,
    )
    residuals = Residuals(toas, model)
    mjd = np.asarray(toas.get_mjds().to_value(u.day), dtype=float)
    residual_us = np.asarray(residuals.time_resids.to_value(u.us), dtype=float)
    error_us = np.asarray(toas.get_errors().to_value(u.us), dtype=float)
    weighted_rms_us = float(residuals.rms_weighted().to_value(u.us))
    reduced_chi2 = float(residuals.reduced_chi2)
    spin_frequency_hz = float(model["F0"].value)
    _write_residual_csv(csv_path, mjd, residual_us, error_us)

    pulsar_name = next(
        (
            str(model[name].value)
            for name in ("PSR", "PSRJ")
            if name in model.params
        ),
        "Pulsar",
    )
    write_residual_series_plot(
        plot_path,
        mjd=mjd,
        residual_us=residual_us,
        error_us=error_us,
        spin_frequency_hz=spin_frequency_hz,
        pulsar_name=pulsar_name,
        weighted_rms_us=weighted_rms_us,
        reduced_chi2=reduced_chi2,
    )
    return csv_path, plot_path
