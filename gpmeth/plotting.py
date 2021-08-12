"""Functions to plot results of model fits"""

from typing import Optional
from gpflow.models.model import GPModel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .models import RegressionData


# Colors
met_color_dict = {
    1: (0.5, 0.0, 0.0, 1.0),
    0: (0.0, 0.0, 0.5, 1.0),
}  # Extremes of the jet colorscale


def plot_input_data(data: RegressionData, ax=None):
    """Scatterplot of the input data with pseudotime on y- and genome on x-axis."""
    X, Y = data
    if ax is None:
        fig, ax = plt.subplots()
    g = sns.scatterplot(
        x=X[:, 1].flatten(),
        y=X[:, 0].flatten(),
        hue=Y.flatten(),
        palette=met_color_dict,
        s=10,
        ax=ax,
    )
    return g


def plot_prediction_contours(
    X_gr: np.array, p: np.array, fixed_clevels: bool = True, ax=None
):
    """Plot predictions of GP models in 2d as contours."""

    if ax is None:
        fig, ax = plt.subplots()

    if fixed_clevels:
        clevels = np.linspace(0, 1, 9)

        cs = ax.tricontour(
            X_gr[:, 1],
            X_gr[:, 0],
            p[:, 0],
            levels=clevels[clevels != 0.5],
            linewidths=2,
            cmap="jet",
        )
        ax.clabel(cs)
        cs = ax.tricontour(X_gr[:, 1], X_gr[:, 0], p[:, 0], levels=[0.5], linewidths=4)
        ax.clabel(cs)
        # Fill with color
        ax.tricontourf(
            X_gr[:, 1],
            X_gr[:, 0],
            p[:, 0],
            levels=clevels,
            cmap="jet",
            alpha=0.2,
        )
    else:
        cs = ax.tricontour(
            X_gr[:, 1],
            X_gr[:, 0],
            p[:, 0],
            linewidths=2,
            cmap="jet",
        )
        ax.tricontourf(
            X_gr[:, 1],
            X_gr[:, 0],
            p[:, 0],
            cmap="jet",
            alpha=0.2,
        )


def plot_inducing_points(model: GPModel, ax=None):
    """Plot the locations of the inducing points of a model"""
    Z = model.inducing_variable.Z.numpy()
    g = ax.scatter(Z[:, 1], Z[:, 0], marker="x", c="grey")
    return g
