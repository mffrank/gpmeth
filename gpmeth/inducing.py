"""Utility functions to produce inducing points based on input point locations."""
from gpflow.models.training_mixins import InputData
import numpy as np
import itertools


def make_grid(num_points: int, lower: int = -1, upper: int = 1, input_dim: int = 2):

    # Make sure bounds have the same dimensions as input dim
    if input_dim > 1:
        if np.isscalar(lower):
            lower = np.repeat(lower, input_dim)
        elif len(lower) != input_dim:
            raise ValueError("lower must be a scalar or a list with input_dim elements")
        if np.isscalar(upper):
            upper = np.repeat(upper, input_dim)
        elif len(upper) != input_dim:
            raise ValueError("upper must be a scalar or a list with input_dim elements")
        if np.isscalar(num_points):
            num_points = np.repeat(num_points, input_dim)
        elif len(num_points) != input_dim:
            raise ValueError(
                "num_points must be a scalar or a list with input_dim elements"
            )

        a = [np.linspace(lower[i], upper[i], num_points[i]) for i in range(input_dim)]
        grid = np.array([x for x in itertools.product(*a)])
    else:
        grid = np.linspace(lower, upper, num_points)[:, None]
    return grid


def make_grid_inducing_points(X: InputData, num_points: int = 100, extend: int = 0):
    input_dim = X.shape[1]
    n_points_axis = int(num_points ** (1 / input_dim))
    minima = np.amin(X, axis=0)
    maxima = np.amax(X, axis=0)
    extensions = extend * (maxima - minima)
    minima = minima - extensions
    maxima = maxima + extensions
    grid = make_grid(
        num_points=n_points_axis,
        lower=minima,
        upper=maxima,
        input_dim=input_dim,
    )
    return grid
