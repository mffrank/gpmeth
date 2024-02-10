"""Functions to plot results of model fits"""

from typing import List, Optional
from gpflow.models.model import GPModel
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import warnings

import numpy as np
import seaborn as sns
from .util import InputData, OutputData, RegressionData

# Colors
met_color_dict = {
    1: (0.5, 0.0, 0.0, 1.0),
    0: (0.0, 0.0, 0.5, 1.0),
}  # Extremes of the jet colorscale


def plot_input_data(
    data: RegressionData,
    genome_dim: Optional[int] = None,
    pseudotime_dim: Optional[int] = None,
    category_dim: Optional[int] = None,
    ax=None,
    *args,
    **kwargs,
):
    """Scatterplot of the input data with pseudotime on y- and genome on x-axis."""
    X, Y = data

    if ax is None:
        fig, ax = plt.subplots()

    if pseudotime_dim is None:
        if category_dim is None:
            if genome_dim is None:
                raise ValueError("Must specify at least one dim")
            else:
                g = plot_input_data_genome(X=X, Y=Y, genome_dim=genome_dim, ax=ax)
        else:
            if genome_dim is None:
                # raise ValueError("Not implemented.")
                print("Not implemented")
                return
            else:
                g = plot_input_data_categorical_genome(
                    X=X, Y=Y, genome_dim=genome_dim, category_dim=category_dim, ax=ax
                )
    else:
        if genome_dim is None:
            g = plot_input_data_pseudotime(
                X=X, Y=Y, pseudotime_dim=pseudotime_dim, ax=ax
            )
        else:
            g = sns.scatterplot(
                x=X[:, genome_dim].flatten(),
                y=X[:, pseudotime_dim].flatten(),
                hue=Y.flatten(),
                palette=met_color_dict,
                s=10,
                ax=ax,
                *args,
                **kwargs,
            )
    return g


def plot_input_data_categorical_genome(
    X: InputData,
    Y: OutputData,
    genome_dim: int,
    category_dim: int,
    ax,    
    group_names=None,
    *args,
    **kwargs,
):
    means = []
    cov = []
    positions = []
    categories = np.unique(X[:, category_dim])
    for tp in categories:
        filter = X[:, -1] == tp
        x, y = X[filter, :], Y[filter, :]
        for pos in np.unique(x[:, genome_dim]):
            f = x[:, genome_dim] == pos
            means.append(y[f].mean())
            cov.append(y[f].shape[0])
            positions.append((pos, tp))

    X = np.array(positions, dtype=np.int)
    Y = np.array(means)

    g = sns.scatterplot(
        x=X[:, 0],
        y=Y,
        size = np.array(cov),
        hue=X[:, category_dim],
        palette=sns.color_palette("tab10", len(categories)),
        ax=ax,
        *args,
        **kwargs,
    )

    return g


def plot_input_data_pseudotime(
    X: InputData, Y: OutputData, pseudotime_dim: int, ax, *args, **kwargs
):
    means = []
    positions = []
    for t in np.unique(X[:, pseudotime_dim]):
        f = X[:, pseudotime_dim] == t
        means.append(Y[f].mean())
        positions.append(t)

    g = sns.scatterplot(
        x=positions,
        y=means,
        ax=ax,
        *args,
        **kwargs,
    )
    return g


def plot_input_data_genome(
    X: InputData, Y: OutputData, genome_dim: int, ax, *args, **kwargs
):
    means = []
    positions = []
    cov = []
    for pos in np.unique(X[:, genome_dim]):
        f = X[:, genome_dim] == pos
        means.append(Y[f].mean())
        cov.append(Y[f].shape[0])
        positions.append(pos)

    g = sns.scatterplot(
        x=positions,
        y=means,
        size=cov,
        ax=ax,
        *args,
        **kwargs,
    )
    g.legend().set_title("coverage")
    return g


def plot_model_output(
    X_gr: np.array,
    p: np.array,
    genome_dim: Optional[int] = None,
    pseudotime_dim: Optional[int] = None,
    category_dim: Optional[int] = None,
    ax=None,
    group_names=None,
    *args,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    if pseudotime_dim is None:
        if category_dim is None:
            if genome_dim is None:
                raise ValueError("Must specify at least one dim")
            else:
                g = plot_model_genome(
                    X_gr=X_gr,
                    p=p,
                    genome_dim=genome_dim,
                    ax=ax,
                    *args,
                    **kwargs,
                )
        else:
            if genome_dim is None:
                # raise ValueError("Not implemented.")
                print("Not implemented")
                pass
            else:
                g = plot_model_categorical_genome(
                    X_gr=X_gr,
                    p=p,
                    genome_dim=genome_dim,
                    category_dim=category_dim,
                    ax=ax,
                    group_names=group_names,
                    *args,
                    **kwargs,
                )
    else:
        if category_dim is None:
            if genome_dim is None:
                g = plot_model_pseudotime(
                    X_gr=X_gr,
                    p=p,
                    pseudotime_dim=pseudotime_dim,
                    ax=ax,
                    *args,
                    **kwargs,
                )
            else:
                g = plot_prediction_contours(
                    X_gr=X_gr,
                    p=p,
                    pseudotime_dim=pseudotime_dim,
                    genome_dim=genome_dim,
                    ax=ax,
                    *args,
                    **kwargs,
                )
        else:
            # raise ValueError("Not implemented.")
            print("Not implemented")
            pass


def plot_model_categorical_genome(
    X_gr: np.array,
    p: np.array,
    category_dim: int,
    genome_dim: int,
    ax,
    group_names=None,
    *args,
    **kwargs,
):
    g = sns.lineplot(
        x=X_gr[:, genome_dim],
        y=p.numpy().flatten(),
        hue=X_gr[:, category_dim].astype(int),
        palette=sns.color_palette("tab10", len(np.unique(X_gr[:, category_dim]))),
        ax=ax,
        *args,
        **kwargs,
    )
    if group_names is not None:
        ax.legend(ax.get_legend_handles_labels()[0], group_names)

    return g


def plot_model_genome(
    X_gr: np.array,
    p: np.array,
    genome_dim: int,
    ax,
    *args,
    **kwargs,
):
    g = sns.lineplot(
        x=X_gr[:, genome_dim],
        y=p.numpy().flatten(),
        linewidth=3,
        ax=ax,
        *args,
        **kwargs,
    )
    return g


def plot_model_pseudotime(
    X_gr: np.array,
    p: np.array,
    pseudotime_dim: int,
    ax,
    *args,
    **kwargs,
):
    g = sns.lineplot(
        x=X_gr[:, pseudotime_dim],
        y=p.numpy().flatten(),
        linewidth=3,
        ax=ax,
        *args,
        **kwargs,
    )
    return g


def plot_prediction_contours(
    X_gr: np.array,
    p: np.array,
    pseudotime_dim: int,
    genome_dim: int,
    fixed_clevels: bool = True,
    n_contours: int = 9,
    ax=None,
    *args,
    **kwargs,
):
    """Plot predictions of GP models in 2d as contours."""

    if ax is None:
        fig, ax = plt.subplots()

    if fixed_clevels:
        clevels = np.linspace(0, 1, n_contours)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
            cs = ax.tricontour(
                X_gr[:, genome_dim],
                X_gr[:, pseudotime_dim],
                p.numpy().flatten(),
                levels=clevels[clevels != 0.5],
                linewidths=1,
                cmap="jet",
            )
            # ax.clabel(cs)
            cs = ax.tricontour(
                X_gr[:, genome_dim],
                X_gr[:, pseudotime_dim],
                p.numpy().flatten(),
                levels=[0.5],
                linewidths=2,
            )
            ax.clabel(cs)
            # Fill with color
            cs=ax.tricontourf(
                X_gr[:, genome_dim],
                X_gr[:, pseudotime_dim],
                p.numpy().flatten(),
                levels=clevels,
                cmap="jet",
                alpha=0.2,
            )
            plt.colorbar(cs)
    else:
        cs = ax.tricontour(
            X_gr[:, genome_dim],
            X_gr[:, pseudotime_dim],
            p.numpy().flatten(),
            linewidths=2,
            cmap="jet",
        )
        ax.tricontourf(
            X_gr[:, genome_dim],
            X_gr[:, pseudotime_dim],
            p[:, 0],
            cmap="jet",
            alpha=0.2,
        )


def plot_inducing_points(model: GPModel, ax=None):
    """Plot the locations of the inducing points of a model"""
    Z = model.inducing_variable.Z.numpy()
    g = ax.scatter(Z[:, 1], Z[:, 0], marker="x", c="grey")
    return g

def plot_minmax(
    X_gr: np.array,
    p: np.array,
    genome_dim: int,
    pseudotime_dim: int,
    minmax_threshold: Optional[float] = None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    n_grid = int(X_gr.shape[0]**0.5) # Always square grid
    psq = p.numpy().reshape((n_grid, -1))
    x = np.unique(X_gr[:,genome_dim])
    mins = np.min(psq, axis=genome_dim)
    maxs = np.max(psq, axis=genome_dim)
    _plot_minmax(x, maxs-mins, minmax_threshold, ax)

    
def _plot_minmax(x, y, minmax_threshold, ax):
    ax.plot(x, y)
    ax.set_ylim(0,1)
    ax.set_ylabel("minmax")
    if minmax_threshold is not None:
        ax.axhline(y=minmax_threshold, color="black", linestyle="--")
        
        collection = collections.BrokenBarHCollection.span_where(
            x, ymin=0, ymax=1, where=y > minmax_threshold, facecolor='grey', alpha=0.5)
        ax.add_collection(collection)
