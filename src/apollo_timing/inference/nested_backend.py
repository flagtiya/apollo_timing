from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from apollo_timing.models.contracts import NestedEngineConfig


class PyMultiNestBackend:
    def __init__(
        self,
        ndim: int,
        prior_callback,
        loglike_callback,
        config: NestedEngineConfig,
        output_dir: Path,
    ) -> None:
        self.ndim = ndim
        self.prior_callback = prior_callback
        self.loglike_callback = loglike_callback
        self.config = config
        self.output_dir = Path(output_dir)
        self.scratch_dir = self._prepare_scratch_dir()

    def run(self) -> dict[str, Any]:
        try:
            import pymultinest
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "PyMultiNest is required for the FAST nested engine. "
                "Install pymultinest and ensure the MultiNest system library is visible."
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        basename = str((self.scratch_dir / "multinest_").resolve())
        pymultinest.run(
            self.loglike_callback,
            self.prior_callback,
            self.ndim,
            outputfiles_basename=basename,
            n_live_points=self.config.live_points,
            evidence_tolerance=self.config.evidence_tolerance,
            sampling_efficiency=self.config.sampling_efficiency,
            seed=self.config.seed,
            resume=self.config.resume,
            verbose=False,
        )
        analyzer = pymultinest.Analyzer(n_params=self.ndim, outputfiles_basename=basename)
        stats = analyzer.get_stats()
        best_fit = analyzer.get_best_fit()
        equal_weighted = self._load_equal_weighted_posterior(analyzer, basename)
        return {
            "stats": stats,
            "best_fit": best_fit,
            "equal_weighted_posterior": equal_weighted,
            "outputfiles_basename": basename,
        }

    def _prepare_scratch_dir(self) -> Path:
        key = str(self.output_dir.resolve())
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        scratch_dir = Path.home() / ".cache" / "apollo-timing" / "multinest" / digest
        scratch_dir.mkdir(parents=True, exist_ok=True)
        if not self.config.resume:
            for path in scratch_dir.glob("multinest_*"):
                path.unlink()
        return scratch_dir

    def _load_equal_weighted_posterior(self, analyzer, basename: str) -> np.ndarray:
        if hasattr(analyzer, "get_equal_weighted_posterior"):
            return np.asarray(analyzer.get_equal_weighted_posterior(), dtype=float)
        fallback_path = Path(f"{basename}post_equal_weights.dat")
        if fallback_path.exists():
            return np.loadtxt(fallback_path)
        raise RuntimeError("PyMultiNest analyzer did not expose equal-weighted posterior samples.")
