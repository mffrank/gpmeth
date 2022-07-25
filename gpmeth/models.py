import tensorflow as tf
import gpflow
import tensorflow_probability as tfp
import numpy as np
from typing import Callable, Optional, Tuple, Union, List
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
from .util import InputData, OutputData, RegressionData


def get_data(data: Union[RegressionData, OutputData]):
    """Return Tuple of input and output even if only output is given."""
    if isinstance(data, tuple):
        X, Y = data
    else:
        X, Y = None, data
    return X, Y


class Model(BayesianModel):
    """Base Class for GPmeth models"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init_params = {}

    def to_dict(self):
        return gpflow.utilities.read_values(self)

    @classmethod
    def from_dict(cls, param_dict: dict, *args, **kwargs):
        obj = cls(*args, **kwargs)
        gpflow.utilities.multiple_assign(obj, param_dict)
        return obj

    def initialize_parameters_from_data(self, *args, **kwargs):
        pass

    def plot_predictions(self, data: RegressionData, *args, **kwargs):
        pass


class Constant(Model):
    """Models outputs with a constant rate"""

    def __init__(self, mu=0.5, *args, **kwargs):
        super().__init__()
        self._mu = gpflow.Parameter(mu, transform=util.InvProbit())

    @property
    def n_parameters(self) -> int:
        return 1

    def optimize(self, data: np.ndarray, *args, **kwargs) -> optimize.OptimizeResult:
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

    def __init__(self, mu=[0], n_categories=1, *args, **kwargs):
        super().__init__()
        self.category_dim = -1
        self._mu = gpflow.Parameter(mu)
        self.n_categories = n_categories
        self.categories = np.arange(n_categories)

    @classmethod
    def from_dict(cls, param_dict: dict, *args, **kwargs):
        # Init model with correct shapes for multiple_assign to work
        obj = cls(mu=np.zeros_like(param_dict["._mu"]), *args, **kwargs)
        gpflow.utilities.multiple_assign(obj, param_dict)
        return obj

    def get_categories_from_data(self, data: RegressionData) -> Tuple:
        X, Y = data
        cat = X[:, self.category_dim]
        self.categories = np.unique(cat).astype(int)
        self.n_categories = len(self.categories)
        if self._mu.shape != self.categories.shape:
            self._mu = gpflow.Parameter(np.zeros_like(self.categories))
        return X, Y, cat

    @property
    def n_parameters(self) -> int:
        return self.n_categories

    def optimize(
        self, data: RegressionData, *args, **kwargs
    ) -> optimize.OptimizeResult:
        X, Y, cat = self.get_categories_from_data(data)
        mu = self._mu.numpy()
        for i in self.categories:
            mu[self.categories == i] = np.mean(Y[cat == i, -1])
        self._mu.assign(mu)

        return optimize.OptimizeResult(success=True)

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        X, Y, cat = self.get_categories_from_data(data)
        cat_likelihoods = np.empty_like(self._mu.numpy())
        for i in self.categories:
            cat_likelihoods[i] = tf.reduce_sum(
                tfp.distributions.Bernoulli(
                    probs=self._mu[self.categories == i]
                ).log_prob(Y[cat == i, -1])
            )
        # print(cat_likelihoods)

        return tf.reduce_sum(cat_likelihoods)

    def predict_y(self, Xnew: InputData) -> MeanAndVariance:
        cat = Xnew[:, self.category_dim]
        mu = np.empty_like(cat, dtype=gpflow.config.default_float())
        for i in self.categories:
            mu[cat == i] = self._mu[self.categories == i]
        return tf.convert_to_tensor(mu), mu - tf.square(mu)

    def plot_predictions(
        self,
        data: RegressionData,
        genome_dim: Optional[int] = None,
        pseudotime_dim: Optional[int] = None,
        n_grid: int = 1000,
        plot_title: bool = True,
        *args,
        **kwargs,
    ):
        pass
        # X, Y, cat = self.get_categories_from_data(data)
        # fig, ax = plt.subplots()
        # if plot_title:
        #     fig.suptitle(
        #         f"Model: {self._name}, Max_LL_objective: {self.maximum_log_likelihood_objective(data)}"
        #     )
        # # Plot data

        # g = plotting.plot_input_data(
        #     data,
        #     genome_dim=genome_dim,
        #     pseudotime_dim=pseudotime_dim,
        #     category_dim=self.category_dim,
        #     ax=ax,
        #     *args,
        #     **kwargs,
        # )
        # # Plot predictions
        # X_gr = inducing.make_categorical_grid_inducing_points(
        #     X, category_dim=self.category_dim, n_grid=n_grid, *args, **kwargs
        # )
        # p, _ = self.predict_y(X_gr)

        # g = plotting.plot_model_output(
        #     X_gr=X_gr,
        #     p=p,
        #     genome_dim=genome_dim,
        #     pseudotime_dim=pseudotime_dim,
        #     category_dim=self.category_dim,
        #     ax=ax,
        #     *args,
        #     **kwargs,
        # )

        # return g


class GPmodel(gpflow.models.SVGP, Model):
    """Handles the commonalities of all gp models in gpmeth."""

    def __init__(
        self,
        pseudotime_dims: Optional[List[int]] = None,
        genome_dim: Optional[int] = None,
        inducing_variable: Optional[np.array] = None,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        *args,
        **kwargs,
    ):
        self.pseudotime_dims = pseudotime_dims
        self.genome_dim = genome_dim
        self.init_params = {
            "pseudotime_dims": pseudotime_dims,
            "genome_dim": genome_dim,
        }

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
    def from_dict(cls, param_dict: dict, *args, **kwargs):
        # Init model with correct shapes for multiple_assign to work
        obj = cls(
            inducing_variable=np.zeros_like(param_dict[".inducing_variable.Z"]),
            *args,
            **kwargs,
        )
        # print(obj)
        gpflow.utilities.multiple_assign(obj, param_dict)
        return obj

    def subset_input_data_to_active_dims(self, data: RegressionData) -> RegressionData:
        X, Y = data
        # Subset X to active dimensions
        ad = self.get_active_dims()
        ad = np.unique(np.concatenate(list(ad.values())))
        X = X[:, ad]

        return X, Y

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
        gpflow.utilities.set_trainable(self.inducing_variable.Z, False)

        # Set the mean function to the mean of the data
        self.mean_function.c.assign(util.InvProbit()._inverse(Y.mean()))

        # Initialize lengthscales according to data
        self.initialize_lengthscales(X=X, *args, **kwargs)

    def get_inducing_points(
        self,
        X: InputData,
        inducing_point_function: Callable = inducing.make_grid_inducing_points,
        inducing_dimensions: str = "fixed",
        *args,
        **kwargs,
    ):
        if inducing_dimensions == "active_dims":
            ad = self.get_active_dims()
            ad = np.unique(np.concatenate(list(ad.values())))
            Z = inducing_point_function(X, active_dims=ad, *args, **kwargs)
        elif inducing_dimensions == "all":
            Z = inducing_point_function(X, *args, **kwargs)
        elif inducing_dimensions == "fixed":
            ad = []
            if self.genome_dim:
                ad.append(self.genome_dim)
            if self.pseudotime_dims:
                ad.extend(self.pseudotime_dims)
            Z = inducing_point_function(X, active_dims=ad, *args, **kwargs)
        else:
            raise ValueError(
                "inducing dimensions must be one of: [active_dims, all, fixed] "
            )
        return Z

    def initialize_lengthscales(
        self, X: InputData, span_fraction: float = 0.1, *args, **kwargs
    ):
        util.initialize_kernel_lengthscales(self, X, span_fraction)

    def initialize_kernel_variances(
        self, variance_value: float = 1, only_trainable: bool = True
    ):
        for k, v in gpflow.utilities.parameter_dict(self).items():
            if k.endswith(".variance") and (v.trainable or not only_trainable):
                v.assign(variance_value)

    # def calculate_genome_variance(
    #     self,
    #     n_grid: int = 1000,
    #     *args,
    #     **kwargs,
    # ):
    #     X_gr = self.get_inducing_points(
    #         X,
    #         inducing_point_function=inducing.make_grid_inducing_points,
    #         num_points=n_grid,
    #         *args,
    #         **kwargs,
    #     )
    #     p, _ = self.predict_y(X_gr)

    def plot_predictions(
        self,
        data: RegressionData,
        pseudotime_dim=None,
        genome_dim=None,
        n_grid: int = 1000,
        plot_inducing: bool = False,
        plot_title: bool = True,
        *args,
        **kwargs,
    ):
        X, Y = data
        genome_dim = genome_dim or self.genome_dim
        pseudotime_dim = pseudotime_dim or self.pseudotime_dims
        if isinstance(pseudotime_dim, list):
            pseudotime_dim = pseudotime_dim[0]

        fig, ax = plt.subplots()
        if plot_title:
            fig.suptitle(
                f"Model: {self._name}, Max_LL_objective: {self.maximum_log_likelihood_objective(data)}"
            )

        # Plot data
        g = plotting.plot_input_data(
            data=data,
            genome_dim=genome_dim,
            pseudotime_dim=pseudotime_dim,
            ax=ax,
        )
        # Plot predictions
        X_gr = self.get_inducing_points(
            X,
            inducing_point_function=inducing.make_grid_inducing_points,
            num_points=n_grid,
            *args,
            **kwargs,
        )
        p, _ = self.predict_y(X_gr)
        g = plotting.plot_model_output(
            X_gr=X_gr,
            p=p,
            genome_dim=genome_dim,
            pseudotime_dim=pseudotime_dim,
            ax=ax,
            *args,
            **kwargs,
        )
        if plot_inducing:
            g = plotting.plot_inducing_points(self, ax=ax)

        return g

    def get_active_dims(self):
        """Returns the active dimensions of the model"""

        target_types = (
            gpflow.kernels.Polynomial,
            gpflow.kernels.Linear,
            gpflow.kernels.Stationary,
            gpflow.kernels.Periodic,
            gpflow.kernels.Coregion,
        )
        input_name, state = self.__class__.__name__, dict()
        accumulator = (input_name, state)

        def update_state(kernel, path, state):
            state[path] = kernel.active_dims
            return state

        ad = gpflow.utilities.traversal.traverse_module(
            self, accumulator, update_state, target_types
        )
        return ad


class ConstantLinear(GPmodel):
    def __init__(self, pseudotime_dims=[0], *args, **kwargs):
        kernel = gpflow.kernels.Linear(active_dims=pseudotime_dims)
        super().__init__(
            kernel=kernel, pseudotime_dims=pseudotime_dims, *args, **kwargs
        )
        gpflow.set_trainable(self.mean_function, True)


class ConstantMatern(GPmodel):
    def __init__(self, pseudotime_dims=[0], *args, **kwargs):
        kernel = gpflow.kernels.Matern32(active_dims=pseudotime_dims)
        super().__init__(
            kernel=kernel, pseudotime_dims=pseudotime_dims, *args, **kwargs
        )


class RBFConstant(GPmodel):
    def __init__(self, genome_dim=1, fixed_genome_lengthscale=False, *args, **kwargs):
        gnkern = gpflow.kernels.RBF(active_dims=[genome_dim], lengthscales=100)
        if fixed_genome_lengthscale:
            gpflow.set_trainable(gnkern.lengthscales, False)
        kernel = gnkern
        super().__init__(kernel=kernel, genome_dim=genome_dim, *args, **kwargs)


class GPFullModel(GPmodel):
    """Implements from_null to initialize with a pretrained null model"""

    @classmethod
    def from_null(
        cls, null_model: GPmodel, null_kernel_trainable: bool = False, *args, **kwargs
    ):
        """Creates this model from a null model (must have the same kernel as this models nullkern)"""

        # Adjust Null model parameters to fit in full model
        param_dict = gpflow.utilities.parameter_dict(null_model)
        param_dict = {
            k.replace("kernel", "kernel.kernels[0]"): v for k, v in param_dict.items()
        }

        # Init model with correct shapes for multiple_assign to work
        obj = cls(
            inducing_variable=np.zeros_like(param_dict[".inducing_variable.Z"]),
            *args,
            **kwargs,
        )

        gpflow.utilities.multiple_assign(obj, param_dict)

        if not null_kernel_trainable:
            gpflow.utilities.set_trainable(obj.kernel.kernels[0], False)

        # Set remaining variances low in the beginning
        obj.initialize_kernel_variances(variance_value=0.0001, only_trainable=True)

        return obj

    def copy_null_parameters(
        self, null_model: GPmodel, null_kernel_trainable: bool = False, *args, **kwargs
    ):

        self.inducing_variable = null_model.inducing_variable
        self.q_mu = null_model.q_mu
        self.q_sqrt = null_model.q_sqrt
        self.q_diag = null_model.q_diag

        # Adjust Null model parameters to fit in full model
        self.kernel.kernels[0] = null_model.kernel
        self.mean_function = null_model.mean_function

        if not null_kernel_trainable:
            gpflow.utilities.set_trainable(self.kernel.kernels[0], False)

        # Set remaining variances low in the beginning
        self.initialize_kernel_variances(variance_value=0.01, only_trainable=True)


class RBFLinear(GPFullModel):
    """Model with a product kernel: RBF(genome) * Linear(pseudotime)"""

    def __init__(
        self,
        pseudotime_dims=[0],
        genome_dim=1,
        fixed_genome_lengthscale=False,
        *args,
        **kwargs,
    ):
        nullkern = gpflow.kernels.RBF(active_dims=[genome_dim], lengthscales=100)
        if fixed_genome_lengthscale:
            gpflow.set_trainable(nullkern.lengthscales, False)

        gnkern = gpflow.kernels.RBF(active_dims=[genome_dim])
        pskern = gpflow.kernels.Linear(active_dims=pseudotime_dims)
        # In a product kernel the variances become connected
        gpflow.utilities.set_trainable(gnkern.variance, False)

        kernel = nullkern + pskern * gnkern
        super().__init__(
            kernel=kernel,
            genome_dim=genome_dim,
            pseudotime_dims=pseudotime_dims,
            *args,
            **kwargs,
        )
        # gpflow.set_trainable(self.mean_function, True)  # Linear needs flexible baseline

    def copy_null_parameters(
        self, null_model: GPmodel, null_kernel_trainable: bool = False, *args, **kwargs
    ):
        super().copy_null_parameters(null_model, null_kernel_trainable, *args, **kwargs)
        gpflow.set_trainable(self.mean_function, True)  # Linear needs flexible baseline


class RBFMatern(GPFullModel):
    """Model with a product kernel: RBF(genome) * Linear(pseudotime)"""

    def __init__(
        self,
        pseudotime_dims=[0],
        genome_dim=1,
        fixed_genome_lengthscale=False,
        *args,
        **kwargs,
    ):
        nullkern = gpflow.kernels.RBF(active_dims=[genome_dim], lengthscales=100)
        if fixed_genome_lengthscale:
            gpflow.set_trainable(nullkern.lengthscales, False)

        gnkern = gpflow.kernels.RBF(active_dims=[genome_dim])
        pskern = gpflow.kernels.Matern32(active_dims=pseudotime_dims)
        # In a product kernel the variances become connected
        gpflow.utilities.set_trainable(gnkern.variance, False)

        kernel = nullkern + pskern * gnkern
        super().__init__(
            kernel=kernel,
            pseudotime_dims=pseudotime_dims,
            genome_dim=genome_dim,
            *args,
            **kwargs,
        )


class RBFRBF(GPFullModel):
    """Model with a product kernel: RBF(genome) * Linear(pseudotime)"""

    def __init__(
        self,
        pseudotime_dims=[0],
        genome_dim=1,
        fixed_genome_lengthscale=False,
        *args,
        **kwargs,
    ):
        nullkern = gpflow.kernels.RBF(active_dims=[genome_dim], lengthscales=100)
        if fixed_genome_lengthscale:
            gpflow.set_trainable(nullkern.lengthscales, False)

        gnkern = gpflow.kernels.RBF(active_dims=[genome_dim])
        pskern = gpflow.kernels.RBF(active_dims=pseudotime_dims)
        # In a product kernel the variances become connected
        gpflow.utilities.set_trainable(gnkern.variance, False)

        kernel = nullkern + pskern * gnkern
        super().__init__(
            kernel=kernel,
            pseudotime_dims=pseudotime_dims,
            genome_dim=genome_dim,
            *args,
            **kwargs,
        )


class RBFCategorical(GPFullModel):
    """Model with a MultiOutput Kernel that has a shared lengthscale between outputs."""

    def __init__(
        self,
        genome_dim: int = 0,
        output_dim: int = 1,
        fixed_genome_lengthscale: bool = False,
        *args,
        **kwargs,
    ):

        self.category_dim = -1
        self.init_params = {
            "genome_dim": genome_dim,
            "fixed_genome_lengthscale": fixed_genome_lengthscale,
        }

        nullkern = gpflow.kernels.RBF(active_dims=[genome_dim], lengthscales=100)
        genkern = gpflow.kernels.RBF(active_dims=[genome_dim], lengthscales=100)
        if fixed_genome_lengthscale:
            gpflow.set_trainable(nullkern.lengthscales, False)
            gpflow.set_trainable(genkern.lengthscales, False)

        # Coregion kernel
        coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=1, active_dims=[1])
        coreg.W.assign(np.zeros_like(coreg.W.numpy()))
        gpflow.utilities.set_trainable(coreg.W, False)
        gpflow.utilities.set_trainable(genkern.variance, False)

        kernel = nullkern + genkern * coreg
        super().__init__(kernel=kernel, genome_dim=genome_dim, *args, **kwargs)

    def copy_null_parameters(
        self, null_model: GPmodel, null_kernel_trainable: bool = False, *args, **kwargs
    ):

        super().copy_null_parameters(
            null_model=null_model,
            null_kernel_trainable=null_kernel_trainable,
            *args,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, param_dict, *args, **kwargs):
        obj = cls(
            inducing_variable=np.zeros_like(param_dict[".inducing_variable.Z"]),
            output_dim=param_dict[".kernel.kernels[1].kernels[1].kappa"].shape[0],
            *args,
            **kwargs,
        )
        gpflow.utilities.multiple_assign(obj, param_dict)
        return obj

    def initialize_parameters_from_data(
        self,
        data: RegressionData,
        compute_inducing_points: bool = True,
        *args,
        **kwargs,
    ):
        X, Y = data
        n_categories = len(np.unique(X[:, self.category_dim]))

        coreg = gpflow.kernels.Coregion(
            output_dim=n_categories, rank=1, active_dims=[X.shape[1] - 1]
        )
        coreg.W.assign(np.zeros_like(coreg.W.numpy()))
        gpflow.utilities.set_trainable(coreg.W, False)
        self.kernel.kernels[1].kernels[1] = coreg

        super().initialize_parameters_from_data(
            data,
            compute_inducing_points=compute_inducing_points,
            *args,
            **kwargs,
        )

    def get_inducing_points(
        self,
        X: InputData,
        # grid_dims: List[int] = [1],
        inducing_point_function: Callable = inducing.make_categorical_grid_inducing_points,
        *args,
        **kwargs,
    ):
        return super().get_inducing_points(
            X,
            inducing_point_function=inducing_point_function,
            active_dims_only=False,
            *args,
            **kwargs,
        )
        # return inducing_point_function(X, grid_dims, *args, **kwargs)

    def plot_predictions(
        self,
        data: RegressionData,
        genome_dim: Optional[int] = None,
        pseudotime_dim: Optional[int] = None,
        n_grid: int = 1000,
        plot_title: bool = True,
        *args,
        **kwargs,
    ):
        X, Y = data
        genome_dim = genome_dim or self.genome_dim
        pseudotime_dim = pseudotime_dim or self.pseudotime_dims
        if isinstance(pseudotime_dim, list):
            pseudotime_dim = pseudotime_dim[0]

        fig, ax = plt.subplots()
        if plot_title:
            fig.suptitle(
                f"Model: {self._name}, Max_LL_objective: {self.maximum_log_likelihood_objective(data)}"
            )

        # Plot data
        g = plotting.plot_input_data(
            data,
            genome_dim=genome_dim,
            pseudotime_dim=pseudotime_dim,
            category_dim=self.category_dim,
            ax=ax,
        )
        # Plot predictions
        X_gr = self.get_inducing_points(X, num_points=n_grid, *args, **kwargs)
        p, _ = self.predict_y(X_gr)
        g = plotting.plot_model_output(
            X_gr=X_gr,
            p=p,
            genome_dim=genome_dim,
            pseudotime_dim=pseudotime_dim,
            category_dim=self.category_dim,
            ax=ax,
        )
        return g
