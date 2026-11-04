"""Inference implementations and lazy public bootstrap entry points."""

__all__ = ["bootstrap_coarse_binary", "bootstrap_coarse_ell1", "bootstrap_partial_ell1"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    if name == "bootstrap_coarse_binary":
        from timing_accelerator.inference.bayesian_coarse_binary_bootstrap import (
            bootstrap_coarse_binary,
        )

        return bootstrap_coarse_binary
    from timing_accelerator.inference import bayesian_partial_ell1_bootstrap

    return getattr(bayesian_partial_ell1_bootstrap, name)
