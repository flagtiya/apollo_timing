"""Binary pulsar timing pipelines."""

__all__ = [
    "ApolloSourceType",
    "BayesianBinaryLatticeSolverConfig",
    "BayesianBinaryLatticeSolverResult",
    "BayesianELL1DualLatticeSolverConfig",
    "BayesianELL1DualLatticeSolverResult",
    "solve_apollo_timing",
    "solve_bayesian_binary_lattice",
    "solve_bayesian_ell1_dual_lattice",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    if name in {"ApolloSourceType", "solve_apollo_timing"}:
        from timing_accelerator.pipeline import apollo_timing

        return getattr(apollo_timing, name)
    if name in {
        "BayesianBinaryLatticeSolverConfig",
        "BayesianBinaryLatticeSolverResult",
        "solve_bayesian_binary_lattice",
    }:
        from timing_accelerator.pipeline import bayesian_binary_lattice_solver

        return getattr(bayesian_binary_lattice_solver, name)
    from timing_accelerator.pipeline import bayesian_ell1_dual_lattice_solver

    return getattr(bayesian_ell1_dual_lattice_solver, name)
