"""Bounded phase-connection proposals for an isolated spin seed."""

import numpy as np


def propose_spin_lut_path(phase_cycles, epochs_days):
    """Return a pulse-count candidate, not a uniqueness or fit certificate.

    Grid coefficients are cycles in a centered, unit-span time coordinate.
    The quadratic surrogate only ranks integer paths; exact timing must be
    refit with those integers fixed.
    """
    phase = np.asarray(phase_cycles, dtype=np.longdouble)
    x = np.asarray(epochs_days, dtype=float)
    if phase.ndim != 1 or phase.shape != x.shape or x.size < 4:
        raise ValueError("Expected aligned one-dimensional phase and epoch arrays.")
    if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(x)) or np.ptp(x) <= 0:
        raise ValueError("Expected finite phases and a positive epoch span.")
    x = (x - x.mean()) / np.ptp(x)
    design = np.column_stack((np.ones_like(x), x, x*x))
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("Spin LUT requires at least three distinct epochs.")
    inverse = np.linalg.pinv(design)
    base = np.rint(phase)
    y = np.asarray(phase - base, dtype=float)
    best_score = np.inf
    best_counts = None
    for linear in np.linspace(-3, 3, 61):
        for quadratic in np.linspace(-0.25, 0.25, 11):
            trial = y + linear*x + quadratic*x*x
            trial -= np.angle(np.mean(np.exp(2j*np.pi*trial))) / (2*np.pi)
            offsets = np.rint(trial)
            residual = trial - offsets
            residual -= design @ (inverse @ residual)
            score = float(residual @ residual)
            if score < best_score:
                best_score = score
                best_counts = base + offsets
    return best_counts
