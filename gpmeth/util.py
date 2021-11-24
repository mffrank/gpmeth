import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import numpy as np

from typing import Callable, Iterator, Optional, Tuple, TypeVar, Union

InputData = Union[tf.Tensor]
OutputData = Union[tf.Tensor]
RegressionData = Tuple[InputData, OutputData]


class InvProbit(tfp.bijectors.Bijector):
    """Bijector class for the probit transformation"""

    def __init__(self, validate_args=False, name="probit"):
        super().__init__(
            validate_args=validate_args, forward_min_event_ndims=0, name=name
        )

    def _forward(self, x):
        jitter = 1e-3  # ensures output is strictly between 0 and 1
        return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter

    def _inverse(self, y):
        return np.sqrt(2.0) * tf.math.erfinv(2 * y - 1)


def initialize_kernel_lengthscales(
    input_module: tf.Module, X: InputData, span_fraction: float = 0.1
):
    """Visit all kernels with lengtscale parameters and update with a fraction of the input data span."""
    target_types = (gpflow.kernels.Stationary, gpflow.kernels.Periodic)
    input_name, state = input_module.__class__.__name__, dict()
    accumulator = (input_name, state)

    def update_state(kernel, _, state):
        if kernel.lengthscales.trainable:
            ad = kernel.active_dims
            ls = span_fraction * (X[:, ad].max(axis=0) - X[:, ad].min(axis=0))
            ls = max(ls)
            # if len(ls) == 1:
            #     ls = ls[0]
            kernel.lengthscales.assign(ls)
        return state

    _ = gpflow.utilities.traversal.traverse_module(
        input_module, accumulator, update_state, target_types
    )
    return
