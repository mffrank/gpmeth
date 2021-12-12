"""Utility functions to produce inducing points based on input point locations."""
from typing import List, Optional
from .util import InputData
import numpy as np
import itertools
import scipy


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
        grid = np.linspace(lower, upper, num_points).reshape((-1, 1))
    return grid


def make_grid_inducing_points(
    X: InputData,
    num_points: int = 144,
    extend: int = 0,
    active_dims: Optional[np.array] = None,
    *args,
    **kwargs
):
    if active_dims is not None:
        ndims = X.shape[1]
        X = X[:, active_dims]

    input_dim = X.shape[1]
    n_points_axis = int(num_points ** (1 / max(input_dim, 1)))
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
    if active_dims is not None:
        Zfull = np.zeros(shape=(grid.shape[0], ndims))
        Zfull[:, active_dims] = grid
        grid = Zfull
    return grid


def make_categorical_grid_inducing_points(
    X: InputData, num_points: int = 144, extend: int = 0, *args, **kwargs
):
    categories = np.unique(X[:, -1])
    Xgr = make_grid_inducing_points(
        X=X[:, :-1],
        active_dims = [0],
        num_points=num_points // len(categories),
        extend=extend,
    )
    Z = np.vstack([np.hstack((Xgr, np.zeros_like(Xgr) + i)) for i in categories])

    return Z

def make_kmeans_inducing_points(
    X: InputData, num_points: int = 144, *args, **kwargs
):
    """
    Initialize inducing inputs using kmeans(++)
    :param X:  An array of training inputs X ⊂ 𝑋, with |X| = N < ∞. We frequently assume X= ℝ^D
    and this is [N,D]
    :param num_points: integer, number of inducing points to return. Equiv. "k" to use in kmeans
    :return: Z, None, num_points inducing inputs
    """
    N = X.shape[0]
    # normalize data
    X_stds = np.std(X, axis=0)
    if np.min(X_stds) < 1e-13:
        warnings.warn("One feature of training inputs is constant")
    X = X / X_stds

    centroids, _ = scipy.cluster.vq.kmeans(X, num_points)
    # Some times K-num_pointseans returns fewer than K centroids, in this case we sample remaining point from data
    if len(centroids) < num_points:
        num_extra_points = num_points - len(centroids)
        indices = np.random.choice(N, size=num_extra_points, replace=False)
        additional_points = X[indices]
        centroids = np.concatenate([centroids, additional_points], axis=0)
    return centroids * X_stds
