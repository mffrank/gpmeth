from gpflow import inducing_variables
import tensorflow as tf
import gpflow
import tensorflow_probability as tfp
import numpy as np
from typing import Callable, Iterator, Optional, Tuple, TypeVar, Union
from gpflow.models.model import (
    BayesianModel,
    MeanAndVariance,
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from . import util
from . import inducing
from . import plotting

InputData = Union[tf.Tensor]
OutputData = Union[tf.Tensor]
RegressionData = Tuple[InputData, OutputData]


def get_data(data: Union[RegressionData, OutputData]):
    """Return Tuple of input and output even if only output is given."""
    if isinstance(data, tuple):
        X, Y = data
    else:
        X, Y = None, data
    return X, Y


class Model(BayesianModel):
    def to_dict(self):
        return gpflow.utilities.read_values(self)

    @classmethod
    def from_dict(cls, param_dict: dict):
        obj = cls()
        gpflow.utilities.multiple_assign(obj, param_dict)
        return obj


class Constant(Model):
    """Models outputs with a constant rate"""

    def __init__(self, mu=0.5):
        super().__init__()
        self._mu = gpflow.Parameter(mu, transform=util.InvProbit())

    @property
    def n_parameters(self) -> int:
        return 1

    def optimize(self, data: np.ndarray) -> optimize.OptimizeResult:
        _, Y = get_data(data)
        self._mu.assign(Y.mean())
        return optimize.OptimizeResult(success=True)

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        _, Y = get_data(data)
        return tf.reduce_sum(tfp.distributions.Bernoulli(probs=self._mu).log_prob(Y))

    def predict_y(self, Xnew: InputData) -> MeanAndVariance:
        mu = tf.ones_like(Xnew[:, 0]) * self._mu
        return mu, mu - tf.square(mu)


class ConstantCategorical(Model):
    """Models output with a constant rate per category. Assumes the last column of the input is specifying a category."""

    def __init__(self, mu=[0], n_categories=1):
        super().__init__()
        self._mu = gpflow.Parameter(mu)
        self.n_categories = n_categories
        self.categories = np.arange(n_categories)

    @classmethod
    def from_dict(cls, param_dict: dict):
        # Init model with correct shapes for multiple_assign to work
        obj = cls(mu=np.zeros_like(param_dict["._mu"]))
        gpflow.utilities.multiple_assign(obj, param_dict)
        return obj

    def get_categories_from_data(self, data: RegressionData) -> Tuple:
        X, Y = data
        cat = X[:, -1]
        self.categories = np.unique(cat)
        self.n_categories = len(self.categories)
        if self._mu.shape != self.categories.shape:
            self._mu = gpflow.Parameter(np.zeros_like(self.categories))
        return X, Y, cat

    @property
    def n_parameters(self) -> int:
        return self.n_categories

    def optimize(self, data: RegressionData) -> optimize.OptimizeResult:
        X, Y, cat = self.get_categories_from_data(data)
        mu = self._mu.numpy()
        for i in self.categories:
            mu[self.categories == i] = np.mean(Y[cat == i, -1])
        self._mu.assign(mu)

        return optimize.OptimizeResult(success=True)

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        X, Y, cat = self.get_categories_from_data(data)
        cat_likelihoods = np.empty_like(self._mu)
        for i in self.categories:
            cat_likelihoods[i] = tf.reduce_sum(
                tfp.distributions.Bernoulli(
                    probs=self._mu[self.categories == i]
                ).log_prob(Y[cat == i, -1])
            )
        print(cat_likelihoods)

        return tf.reduce_sum(cat_likelihoods)

    def predict_y(self, Xnew: InputData) -> MeanAndVariance:
        cat = Xnew[:, -1:]
        mu = np.empty_like(cat, dtype=gpflow.config.default_float())
        for i in self.categories:
            mu[cat == i] = self._mu[self.categories == i]
        return tf.convert_to_tensor(mu), mu - tf.square(mu)


class GPmodel(gpflow.models.SVGP, Model):
    """Handles the commonalities of all gp models in gpmeth."""

    def __init__(
        self,
        inducing_variable: Optional[np.array] = None,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        *args,
        **kwargs,
    ):
        likelihood = gpflow.likelihoods.Bernoulli()
        if inducing_variable is None:
            inducing_variable = gpflow.inducing_variables.InducingPoints(
                Z=np.array([0])
            )
        if mean_function is None:
            mean_function = gpflow.mean_functions.Constant(0)
            gpflow.utilities.set_trainable(mean_function, False)
        elif isinstance(mean_function, Model):
            # Take mean prediction of model as mean function
            mean_model = mean_function
            mean_function = lambda X: mean_model.predict_f(X)[0]

        super().__init__(
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            *args,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, param_dict: dict):
        # Init model with correct shapes for multiple_assign to work
        obj = cls(inducing_variable=np.zeros_like(param_dict[".inducing_variable.Z"]))
        gpflow.utilities.multiple_assign(obj, param_dict)
        return obj

    def optimize(
        self,
        data: RegressionData,
        initialize_parameters: bool = True,
        *args,
        **kwargs,
    ) -> optimize.OptimizeResult:
        """Fit the model to data."""

        if initialize_parameters:
            self.initialize_parameters_from_data(data, *args, **kwargs)

        # Do the optimization
        o = gpflow.optimizers.Scipy()
        training_loss = self.training_loss_closure(data)
        fitres = o.minimize(training_loss, variables=self.trainable_variables)
        return fitres

    def initialize_parameters_from_data(
        self,
        data: RegressionData,
        compute_inducing_points: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize parameters according to the input data in a sensible way."""
        X, Y = data

        # Compute inducing points
        if compute_inducing_points:
            self.inducing_variable = gpflow.models.util.inducingpoint_wrapper(
                self.get_inducing_points(X, *args, **kwargs)
            )
            num_inducing = self.inducing_variable.num_inducing
            self._init_variational_parameters(
                num_inducing=num_inducing, q_mu=None, q_sqrt=None, q_diag=None
            )

        # Set the mean function to the mean of the data
        self.mean_function.c.assign(util.InvProbit()._inverse(Y.mean()))

        # Initialize lengthscales according to data
        self.initialize_lengthscales(X=X, *args, **kwargs)

    def get_inducing_points(
        self,
        X: InputData,
        inducing_point_function: Callable = inducing.make_grid_inducing_points,
        *args,
        **kwargs,
    ):
        return inducing_point_function(X, *args, **kwargs)

    def initialize_lengthscales(self, X: InputData, span_fraction: float = 0.1):
        util.initialize_kernel_lengthscales(self, X, span_fraction)

    def plot_predictions(
        self,
        data: RegressionData,
        n_grid: int = 1000,
        plot_inducing: bool = False,
        *args,
        **kwargs,
    ):
        X, Y = data
        fig, ax = plt.subplots()
        # Plot data
        g = plotting.plot_input_data(data, ax)
        # Plot predictions
        X_gr = inducing.make_grid_inducing_points(X, n_grid, *args, **kwargs)
        p, _ = self.predict_y(X_gr)
        g = plotting.plot_prediction_contours(X_gr=X_gr, p=p, ax=ax, fixed_clevels=True)
        # Plot inducing points
        if plot_inducing:
            g = plotting.plot_inducing_points(self, ax=ax)

        return g


class ConstantLinear(GPmodel):
    def __init__(self, pseudotime_dims=[0], *args, **kwargs):
        kernel = gpflow.kernels.Constant() + gpflow.kernels.Linear(
            active_dims=pseudotime_dims
        )
        super().__init__(kernel=kernel, *args, **kwargs)


class RBFLinear(GPmodel):
    def __init__(self, pseudotime_dims=[0], genome_dims=[1], *args, **kwargs):
        pskern = gpflow.kernels.RBF(active_dims=pseudotime_dims)
        gnkern = gpflow.kernels.Constant() + gpflow.kernels.Linear(
            active_dims=genome_dims
        )
        kernel = pskern * gnkern
        super().__init__(kernel=kernel, *args, **kwargs)
