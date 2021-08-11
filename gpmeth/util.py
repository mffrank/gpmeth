import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class InvProbit(tfp.bijectors.Bijector):
    """Bijector class for the probit transformation"""

    def __init__(self, validate_args=False, name="probit"):
        super(InvProbit, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=0, name=name
        )

    def _forward(self, x):
        jitter = 1e-3  # ensures output is strictly between 0 and 1
        return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter

    def _inverse(self, y):
        return np.sqrt(2.0) * tf.math.erfinv(2 * y - 1)
