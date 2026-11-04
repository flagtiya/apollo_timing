from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class IntegerLeastSquaresCandidate:
    integer_vector: tuple[int, ...]
    squared_mahalanobis: float


@dataclass(frozen=True)
class IntegerLeastSquaresResult:
    candidates: tuple[IntegerLeastSquaresCandidate, ...]
    decorrelation_iterations: int
    visited_nodes: int


def _nearest_integer(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_HALF_EVEN))


def _decimal_vector(values: Sequence[float]) -> list[Decimal]:
    result = []
    for value in values:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Integer least-squares inputs must be finite.")
        result.append(Decimal(str(value)))
    return result


def _decimal_symmetric_matrix(
    values: Sequence[Sequence[float]],
    dimension: int,
) -> list[list[Decimal]]:
    array = np.asarray(values, dtype=object)
    if array.shape != (dimension, dimension):
        raise ValueError("Covariance must be square and match the center dimension.")
    matrix = [[Decimal(0) for _ in range(dimension)] for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            raw_left = array[row, column]
            raw_right = array[column, row]
            left = (
                raw_left
                if isinstance(raw_left, Decimal)
                else Decimal(str(float(raw_left)))
            )
            right = (
                raw_right
                if isinstance(raw_right, Decimal)
                else Decimal(str(float(raw_right)))
            )
            if not left.is_finite() or not right.is_finite():
                raise ValueError("Integer least-squares inputs must be finite.")
            value = (left + right) / 2
            matrix[row][column] = value
            matrix[column][row] = value
    return matrix


def _invert_decimal_matrix(matrix: list[list[Decimal]]) -> list[list[Decimal]]:
    dimension = len(matrix)
    augmented = [
        row.copy()
        + [Decimal(int(row_index == column)) for column in range(dimension)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot_row = max(
            range(column, dimension),
            key=lambda row_index: abs(augmented[row_index][column]),
        )
        if augmented[pivot_row][column] == 0:
            raise ValueError("Shape Jacobian must have full column rank.")
        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row_index in range(dimension):
            if row_index == column:
                continue
            multiplier = augmented[row_index][column]
            if multiplier == 0:
                continue
            augmented[row_index] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index],
                    augmented[column],
                    strict=True,
                )
            ]
    return [row[dimension:] for row in augmented]


def build_marginalized_integer_covariance(
    shape_jacobian: Sequence[Sequence[float]],
    ambiguity_jacobian: Sequence[Sequence[float]],
    ambiguity_noise_covariance: Sequence[Sequence[float]],
    *,
    decimal_precision: int = 80,
) -> tuple[tuple[Decimal, ...], ...]:
    shape = np.asarray(shape_jacobian, dtype=float)
    ambiguity = np.asarray(ambiguity_jacobian, dtype=float)
    if shape.ndim != 2 or ambiguity.ndim != 2:
        raise ValueError("Shape and ambiguity Jacobians must be two-dimensional.")
    if shape.shape[1] == 0 or ambiguity.shape[1] != shape.shape[1]:
        raise ValueError("Shape and ambiguity Jacobians must share parameters.")
    if not np.all(np.isfinite(shape)) or not np.all(np.isfinite(ambiguity)):
        raise ValueError("Linearized covariance inputs must be finite.")

    ambiguity_count, parameter_count = ambiguity.shape
    noise = _decimal_symmetric_matrix(
        ambiguity_noise_covariance,
        ambiguity_count,
    )
    shape_decimal = [
        [Decimal(str(value)) for value in row]
        for row in shape
    ]
    ambiguity_decimal = [
        [Decimal(str(value)) for value in row]
        for row in ambiguity
    ]
    with localcontext() as context:
        context.prec = max(int(decimal_precision), 32)
        shape_precision = [
            [
                sum(
                    shape_decimal[row][left] * shape_decimal[row][right]
                    for row in range(shape.shape[0])
                )
                for right in range(parameter_count)
            ]
            for left in range(parameter_count)
        ]
        shape_covariance = _invert_decimal_matrix(shape_precision)
        projected = [
            [
                sum(
                    ambiguity_decimal[row][parameter]
                    * shape_covariance[parameter][column]
                    for parameter in range(parameter_count)
                )
                for column in range(parameter_count)
            ]
            for row in range(ambiguity_count)
        ]
        marginalized = [
            [
                noise[row][column]
                + sum(
                    projected[row][parameter]
                    * ambiguity_decimal[column][parameter]
                    for parameter in range(parameter_count)
                )
                for column in range(ambiguity_count)
            ]
            for row in range(ambiguity_count)
        ]
    return tuple(tuple(value for value in row) for row in marginalized)


def build_predictive_integer_covariance(
    ambiguity_jacobian: Sequence[Sequence[float]],
    parameter_covariance: Sequence[Sequence[float]],
    ambiguity_noise_covariance: Sequence[Sequence[float]],
    *,
    decimal_precision: int = 80,
) -> tuple[tuple[Decimal, ...], ...]:
    """Build ``N + B C B.T`` without losing the small noise floor.

    Sparse phase-connection problems commonly have a low-rank parameter term
    many orders of magnitude larger than the TOA phase covariance. Forming the
    sum in float64 can erase the positive diagonal noise that makes the integer
    predictive covariance nonsingular. Decimal construction preserves both
    scales for the integer least-squares factorization.
    """

    ambiguity = np.asarray(ambiguity_jacobian, dtype=float)
    covariance = np.asarray(parameter_covariance, dtype=float)
    if ambiguity.ndim != 2:
        raise ValueError("The ambiguity Jacobian must be two-dimensional.")
    ambiguity_count, parameter_count = ambiguity.shape
    if covariance.shape != (parameter_count, parameter_count):
        raise ValueError("Parameter covariance must match the ambiguity Jacobian.")
    if not np.all(np.isfinite(ambiguity)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Predictive covariance inputs must be finite.")
    noise = _decimal_symmetric_matrix(
        ambiguity_noise_covariance,
        ambiguity_count,
    )
    ambiguity_decimal = [
        [Decimal(str(value)) for value in row]
        for row in ambiguity
    ]
    covariance_decimal = _decimal_symmetric_matrix(covariance, parameter_count)
    with localcontext() as context:
        context.prec = max(int(decimal_precision), 32)
        projected = [
            [
                sum(
                    ambiguity_decimal[row][parameter]
                    * covariance_decimal[parameter][column]
                    for parameter in range(parameter_count)
                )
                for column in range(parameter_count)
            ]
            for row in range(ambiguity_count)
        ]
        result = [
            [
                noise[row][column]
                + sum(
                    projected[row][parameter]
                    * ambiguity_decimal[column][parameter]
                    for parameter in range(parameter_count)
                )
                for column in range(ambiguity_count)
            ]
            for row in range(ambiguity_count)
        ]
    return tuple(tuple(value for value in row) for row in result)


def _ld_factorization(
    covariance: list[list[Decimal]],
) -> tuple[list[list[Decimal]], list[Decimal]]:
    dimension = len(covariance)
    work = [row.copy() for row in covariance]
    lower = [[Decimal(0) for _ in range(dimension)] for _ in range(dimension)]
    diagonal = [Decimal(0) for _ in range(dimension)]

    for row in range(dimension - 1, -1, -1):
        diagonal[row] = work[row][row]
        if diagonal[row] <= 0:
            raise ValueError("Covariance must be positive definite.")
        root = diagonal[row].sqrt()
        for column in range(row + 1):
            lower[row][column] = work[row][column] / root
        for update_row in range(row):
            for column in range(update_row + 1):
                work[update_row][column] -= (
                    lower[row][column] * lower[row][update_row]
                )
        pivot = lower[row][row]
        for column in range(row + 1):
            lower[row][column] /= pivot
    return lower, diagonal


def _integer_gauss_transform(
    lower: list[list[Decimal]],
    transform: list[list[int]],
    inverse_transform: list[list[int]],
    row: int,
    column: int,
) -> None:
    multiplier = _nearest_integer(lower[row][column])
    if multiplier == 0:
        return
    dimension = len(lower)
    for update_row in range(row, dimension):
        lower[update_row][column] -= multiplier * lower[update_row][row]
    for update_row in range(dimension):
        transform[update_row][column] -= multiplier * transform[update_row][row]
    for update_column in range(dimension):
        inverse_transform[row][update_column] += (
            multiplier * inverse_transform[column][update_column]
        )


def _permute_ambiguities(
    lower: list[list[Decimal]],
    diagonal: list[Decimal],
    transform: list[list[int]],
    inverse_transform: list[list[int]],
    column: int,
    updated_diagonal: Decimal,
) -> None:
    dimension = len(lower)
    eta = diagonal[column] / updated_diagonal
    lam = (
        diagonal[column + 1]
        * lower[column + 1][column]
        / updated_diagonal
    )
    old_next = diagonal[column + 1]
    diagonal[column] = eta * old_next
    diagonal[column + 1] = updated_diagonal

    for previous in range(column):
        left = lower[column][previous]
        right = lower[column + 1][previous]
        lower[column][previous] = -lower[column + 1][column] * left + right
        lower[column + 1][previous] = eta * left + lam * right
    lower[column + 1][column] = lam

    for row in range(column + 2, dimension):
        lower[row][column], lower[row][column + 1] = (
            lower[row][column + 1],
            lower[row][column],
        )
    for row in range(dimension):
        transform[row][column], transform[row][column + 1] = (
            transform[row][column + 1],
            transform[row][column],
        )
    inverse_transform[column], inverse_transform[column + 1] = (
        inverse_transform[column + 1],
        inverse_transform[column],
    )


def _decorrelate(
    lower: list[list[Decimal]],
    diagonal: list[Decimal],
) -> tuple[list[list[int]], list[list[int]], int]:
    dimension = len(lower)
    transform = [
        [int(row == column) for column in range(dimension)]
        for row in range(dimension)
    ]
    inverse_transform = [row.copy() for row in transform]
    column = dimension - 2
    last_permuted = column
    iterations = 0
    while column >= 0:
        iterations += 1
        if column <= last_permuted:
            for row in range(column + 1, dimension):
                _integer_gauss_transform(
                    lower,
                    transform,
                    inverse_transform,
                    row,
                    column,
                )
        updated_diagonal = (
            diagonal[column]
            + lower[column + 1][column] ** 2 * diagonal[column + 1]
        )
        if updated_diagonal < diagonal[column + 1]:
            _permute_ambiguities(
                lower,
                diagonal,
                transform,
                inverse_transform,
                column,
                updated_diagonal,
            )
            last_permuted = column
            column = dimension - 2
        else:
            column -= 1
    return transform, inverse_transform, iterations


def _next_step(value: float) -> float:
    return -1.0 if value <= 0.0 else 1.0


def _enumerate_decorrelated(
    lower: list[list[Decimal]],
    diagonal: list[Decimal],
    center: np.ndarray,
    *,
    candidate_count: int,
    max_visited_nodes: int,
    return_partial_on_budget: bool,
) -> tuple[list[np.ndarray], list[float], int]:
    lower_float = np.asarray(
        [[float(value) for value in row] for row in lower],
        dtype=float,
    )
    diagonal_float = np.asarray([float(value) for value in diagonal], dtype=float)
    dimension = center.size
    conditional_sum = np.zeros((dimension, dimension), dtype=float)
    partial_distance = np.zeros(dimension, dtype=float)
    conditional_center = np.zeros(dimension, dtype=float)
    integer = np.zeros(dimension, dtype=float)
    step = np.zeros(dimension, dtype=float)
    vectors: list[np.ndarray] = []
    distances: list[float] = []
    maximum_distance = float("inf")

    level = dimension - 1
    conditional_center[level] = center[level]
    integer[level] = np.rint(conditional_center[level])
    delta = conditional_center[level] - integer[level]
    step[level] = _next_step(delta)

    for visited_nodes in range(1, max_visited_nodes + 1):
        distance = partial_distance[level] + delta * delta / diagonal_float[level]
        if distance < maximum_distance:
            if level > 0:
                level -= 1
                partial_distance[level] = distance
                conditional_sum[level, : level + 1] = (
                    conditional_sum[level + 1, : level + 1]
                    + (integer[level + 1] - conditional_center[level + 1])
                    * lower_float[level + 1, : level + 1]
                )
                conditional_center[level] = center[level] + conditional_sum[level, level]
                integer[level] = np.rint(conditional_center[level])
                delta = conditional_center[level] - integer[level]
                step[level] = _next_step(delta)
                continue

            if len(vectors) < candidate_count:
                vectors.append(integer.astype(np.int64, copy=True))
                distances.append(float(distance))
            else:
                worst = int(np.argmax(distances))
                if distance < distances[worst]:
                    vectors[worst] = integer.astype(np.int64, copy=True)
                    distances[worst] = float(distance)
            if len(vectors) == candidate_count:
                maximum_distance = max(distances)
            integer[0] += step[0]
            delta = conditional_center[0] - integer[0]
            step[0] = -step[0] - _next_step(step[0])
            continue

        if level == dimension - 1:
            return vectors, distances, visited_nodes
        level += 1
        integer[level] += step[level]
        delta = conditional_center[level] - integer[level]
        step[level] = -step[level] - _next_step(step[level])

    if return_partial_on_budget and vectors:
        return vectors, distances, max_visited_nodes
    raise RuntimeError(
        "Integer least-squares enumeration exceeded the configured node budget."
    )


def _solve_float_beam_proposal(
    center: np.ndarray,
    covariance: np.ndarray,
    *,
    candidate_count: int,
    max_visited_nodes: int,
) -> IntegerLeastSquaresResult:
    """Build high-dimensional integer proposals without Decimal decorrelation.

    This is deliberately a proposal kernel, not a replacement for exact
    integer certification.  Cholesky conditioning supplies a cheap Babai-like
    beam; downstream APOLLO targets re-evaluate every retained path in the
    original nonlinear likelihood.
    """

    dimension = int(center.size)
    try:
        cholesky = np.linalg.cholesky(
            0.5 * (covariance + covariance.T)
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance must be positive definite.") from error
    beam_width = max(
        2,
        min(
            64,
            int(max_visited_nodes // max(3 * dimension, 1)),
            max(4, 4 * int(candidate_count)),
        ),
    )
    beam: list[tuple[tuple[int, ...], tuple[float, ...], float]] = [
        ((), (), 0.0)
    ]
    visited_nodes = 0
    offsets = (0, -1, 1, -2, 2)
    budget_exhausted = False
    for index in range(dimension):
        next_beam: list[
            tuple[tuple[int, ...], tuple[float, ...], float]
        ] = []
        diagonal = float(cholesky[index, index])
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            raise ValueError("Covariance Cholesky factor must be positive.")
        for prefix, whitened_prefix, partial_distance in beam:
            if index:
                conditional = float(
                    center[index]
                    + np.dot(
                        cholesky[index, :index],
                        np.asarray(whitened_prefix, dtype=float),
                    )
                )
            else:
                conditional = float(center[index])
            nearest = int(np.rint(conditional))
            options = (0,) if budget_exhausted else offsets
            for offset in options:
                if not budget_exhausted and visited_nodes >= max(
                    int(max_visited_nodes), 1
                ):
                    budget_exhausted = True
                    break
                value = nearest + int(offset)
                candidate = prefix + (value,)
                whitened_value = float(
                    (value - conditional) / diagonal
                )
                distance = partial_distance + whitened_value * whitened_value
                next_beam.append(
                    (
                        candidate,
                        whitened_prefix + (whitened_value,),
                        float(distance),
                    )
                )
                if not budget_exhausted:
                    visited_nodes += 1
            if visited_nodes >= max(int(max_visited_nodes), 1):
                budget_exhausted = True
        if not next_beam:
            break
        next_beam.sort(key=lambda item: item[2])
        beam = next_beam[:beam_width]

    unique: dict[tuple[int, ...], float] = {}
    for vector, _, distance in beam:
        previous = unique.get(vector)
        if previous is None or distance < previous:
            unique[vector] = distance
    candidates = tuple(
        IntegerLeastSquaresCandidate(
            integer_vector=vector,
            squared_mahalanobis=distance,
        )
        for vector, distance in sorted(unique.items(), key=lambda item: item[1])[
            : max(int(candidate_count), 1)
        ]
    )
    return IntegerLeastSquaresResult(
        candidates=candidates,
        decorrelation_iterations=0,
        visited_nodes=int(visited_nodes),
    )


def solve_integer_least_squares(
    center: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    candidate_count: int,
    max_visited_nodes: int = 10_000_000,
    decimal_precision: int = 80,
    return_partial_on_budget: bool = False,
) -> IntegerLeastSquaresResult:
    center_decimal = _decimal_vector(center)
    dimension = len(center_decimal)
    if dimension == 0:
        raise ValueError("At least one integer ambiguity is required.")
    if candidate_count <= 0:
        raise ValueError("Candidate count must be positive.")
    if max_visited_nodes <= 0:
        raise ValueError("Maximum visited nodes must be positive.")

    center_array = np.asarray(center, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    if (
        dimension >= 64
        and covariance_array.shape == (dimension, dimension)
        and np.all(np.isfinite(center_array))
        and np.all(np.isfinite(covariance_array))
    ):
        return _solve_float_beam_proposal(
            center_array,
            0.5 * (covariance_array + covariance_array.T),
            candidate_count=candidate_count,
            max_visited_nodes=max_visited_nodes,
        )

    covariance_decimal = _decimal_symmetric_matrix(covariance, dimension)
    with localcontext() as context:
        context.prec = max(int(decimal_precision), 32)
        lower, diagonal = _ld_factorization(covariance_decimal)
        transform, inverse_transform, iterations = _decorrelate(lower, diagonal)

        transformed_center = []
        transformed_origin = []
        for column in range(dimension):
            value = sum(
                Decimal(transform[row][column]) * center_decimal[row]
                for row in range(dimension)
            )
            origin = _nearest_integer(value)
            transformed_origin.append(origin)
            transformed_center.append(float(value - origin))

    vectors, distances, visited_nodes = _enumerate_decorrelated(
        lower,
        diagonal,
        np.asarray(transformed_center, dtype=float),
        candidate_count=candidate_count,
        max_visited_nodes=max_visited_nodes,
        return_partial_on_budget=bool(return_partial_on_budget),
    )

    unique: dict[tuple[int, ...], float] = {}
    for transformed_vector, distance in zip(vectors, distances, strict=True):
        shifted = [
            int(transformed_vector[index]) + transformed_origin[index]
            for index in range(dimension)
        ]
        integer_vector = tuple(
            sum(
                inverse_transform[row][column] * shifted[row]
                for row in range(dimension)
            )
            for column in range(dimension)
        )
        current = unique.get(integer_vector)
        if current is None or distance < current:
            unique[integer_vector] = float(distance)

    candidates = tuple(
        IntegerLeastSquaresCandidate(
            integer_vector=integer_vector,
            squared_mahalanobis=distance,
        )
        for integer_vector, distance in sorted(unique.items(), key=lambda item: item[1])
    )
    return IntegerLeastSquaresResult(
        candidates=candidates,
        decorrelation_iterations=iterations,
        visited_nodes=visited_nodes,
    )
