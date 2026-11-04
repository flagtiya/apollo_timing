from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from typing import Any, Callable, Sequence

import numpy as np

from timing_accelerator.inference.active_timing_contracts import ActiveTimingBranch, BranchCertification
from timing_accelerator.inference.alias_manifold_solver import AliasManifoldCandidate
from timing_accelerator.inference.branch_proposal import (
    alias_certification_to_branch_certification,
    branch_to_alias_candidate,
)


_PROCESS_CERTIFIER: Callable[[AliasManifoldCandidate], Any] | None = None


def _coerce_branch_certification(result: Any) -> BranchCertification:
    if isinstance(result, BranchCertification):
        return result
    return alias_certification_to_branch_certification(result)


def _set_process_certifier(certifier: Callable[[AliasManifoldCandidate], Any]) -> None:
    global _PROCESS_CERTIFIER
    _PROCESS_CERTIFIER = certifier


def _failed_certification(branch: ActiveTimingBranch, error: BaseException) -> BranchCertification:
    return BranchCertification(
        branch_id=str(branch.branch_id),
        log_likelihood=float("-inf"),
        fitted_theta={},
        covariance=np.zeros((0, 0), dtype=float),
        reduced_chi2=None,
        weighted_rms_us=None,
        backend="certification-failed",
        log_evidence=float("-inf"),
        chi2=None,
        notes=[f"{type(error).__name__}: {error}"],
    )


def _certify_branch_in_process(branch: ActiveTimingBranch) -> BranchCertification:
    if _PROCESS_CERTIFIER is None:
        raise RuntimeError("Process certification worker was not initialized.")
    return _coerce_branch_certification(
        _PROCESS_CERTIFIER(branch_to_alias_candidate(branch))
    )


def _certify_branch_in_process_capturing(branch: ActiveTimingBranch) -> BranchCertification:
    try:
        return _certify_branch_in_process(branch)
    except Exception as exc:
        return _failed_certification(branch, exc)


def certify_branches(
    *,
    branches: Sequence[ActiveTimingBranch],
    certifier: Callable[[AliasManifoldCandidate], Any],
    max_workers: int = 1,
    parallel_backend: str = "thread",
    capture_failures: bool = False,
) -> list[BranchCertification]:
    def certify_one(branch: ActiveTimingBranch) -> BranchCertification:
        try:
            return _coerce_branch_certification(
                certifier(branch_to_alias_candidate(branch))
            )
        except Exception as exc:
            if not capture_failures:
                raise
            return _failed_certification(branch, exc)

    workers = max(int(max_workers), 1)
    backend = str(parallel_backend).lower()
    if backend not in {"thread", "process"}:
        raise ValueError(f"Unsupported certification parallel backend: {parallel_backend}")
    if workers == 1 or len(branches) <= 1:
        return [certify_one(branch) for branch in branches]
    if backend == "process":
        with ProcessPoolExecutor(
            max_workers=min(workers, len(branches)),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_set_process_certifier,
            initargs=(certifier,),
        ) as executor:
            worker = (
                _certify_branch_in_process_capturing
                if capture_failures
                else _certify_branch_in_process
            )
            return list(executor.map(worker, branches))
    with ThreadPoolExecutor(max_workers=min(workers, len(branches))) as executor:
        return list(executor.map(certify_one, branches))
