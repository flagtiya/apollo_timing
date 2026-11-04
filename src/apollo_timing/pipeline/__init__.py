"""Single-star and binary pipeline runners."""

__all__ = [
    "ApolloSourceType",
    "ParameterExplorationConfig",
    "FoldbankBootstrapConfig",
    "FoldbankConfig",
    "FoldbankResult",
    "build_foldbank",
    "inspect_search_psrfits",
    "BayesianBinaryLatticeSolverConfig",
    "BayesianBinaryLatticeSolverResult",
    "BayesianELL1DualLatticeSolverConfig",
    "BayesianELL1DualLatticeSolverResult",
    "IsolatedAnnealingConfig",
    "IsolatedCellSearchConfig",
    "IsolatedPulsarSolverConfig",
    "IsolatedPulsarSolverResult",
    "IsolatedSessionUpdateConfig",
    "IsolatedGlitchSearchConfig",
    "IsolatedUpdateModelSelectionConfig",
    "solve_apollo_timing",
    "solve_bayesian_binary_lattice",
    "solve_bayesian_ell1_dual_lattice",
    "solve_isolated_pulsar",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    if name == "ParameterExplorationConfig":
        from apollo_timing.inference.timing_parameter_exploration import ParameterExplorationConfig
        return ParameterExplorationConfig
    if name == "IsolatedGlitchSearchConfig":
        from apollo_timing.inference.isolated_glitch_search import IsolatedGlitchSearchConfig
        return IsolatedGlitchSearchConfig
    if name == "IsolatedUpdateModelSelectionConfig":
        from apollo_timing.inference.bayesian_isolated_update_model_selection import IsolatedUpdateModelSelectionConfig
        return IsolatedUpdateModelSelectionConfig
    if name in {
        "FoldbankBootstrapConfig", "FoldbankConfig", "FoldbankResult",
        "build_foldbank", "inspect_search_psrfits",
    }:
        from apollo_timing.pipeline import apollo_foldbank

        return getattr(apollo_foldbank, name)
    if name in {
        "BayesianBinaryLatticeSolverConfig",
        "BayesianBinaryLatticeSolverResult",
        "solve_bayesian_binary_lattice",
    }:
        from apollo_timing.pipeline import bayesian_binary_lattice_solver

        return getattr(bayesian_binary_lattice_solver, name)
    if name in {"ApolloSourceType", "solve_apollo_timing"}:
        from apollo_timing.pipeline import apollo_timing

        return getattr(apollo_timing, name)
    if name in {
        "IsolatedAnnealingConfig",
        "IsolatedCellSearchConfig",
        "IsolatedPulsarSolverConfig",
        "IsolatedPulsarSolverResult",
        "IsolatedSessionUpdateConfig",
        "solve_isolated_pulsar",
    }:
        from apollo_timing.pipeline import isolated_pulsar_solver

        return getattr(isolated_pulsar_solver, name)
    from apollo_timing.pipeline import bayesian_ell1_dual_lattice_solver

    return getattr(bayesian_ell1_dual_lattice_solver, name)
